import os
import json
from six import iteritems

import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from vdgnn.dataset.readers import DenseAnnotationsReader, ImageFeaturesHdfReader

TRAIN_VAL_SPLIT = {'0.9': 80000, '1.0': 123287}

class VisDialDataset(Dataset):
    def __init__(self, args, split, isTrain=True):
        r"""
            Initialize the dataset with split taken from ['train', 'val', 'test']
            We follow the protocal as specified in `https://arxiv.org/pdf/1611.08669.pdf`, namely

            For VisDial v1.0:
                train split:
                    img_feat: train split
                    dialog_data: trainval split (top 123287)
                val split:
                    img_feat: val split
                    dialog_data: trainval split (last 2064)
                test split:
                    img_feat: test split
                    dialog_data: test split
            For VisDial v0.9:
                train split:
                    img_feat: train split
                    dialog_data: trainval split (top 80000)
                val split (isTrain=True):
                    img_feat: train split
                    dialog_data: trainval split (last 2783)
                val split (isTrain=False):
                    img_feat: val split
                    dialog_data: val split
        """
        super(VisDialDataset, self).__init__()
        self.args = args
        self.__split = split
        self.__in_memory = args.in_memory
        self.__version = args.version
        self.isTrain = isTrain
        if self.__split == 'val' and self.__version == '0.9' and self.isTrain:
            input_img_path = args.img_train
            img_split = 'train'
            self.img_start_idx = TRAIN_VAL_SPLIT[self.__version]
        else:
            input_img_path = getattr(args, 'img_%s' % split)
            img_split = self.__split
            self.img_start_idx = 0
        if self.__split == 'val' and self.isTrain:
            self.data_start_idx = TRAIN_VAL_SPLIT[self.__version]
            data_split = 'train'
        else:
            self.data_start_idx = 0
            data_split = self.__split

        self.input_img = os.path.join(args.dataroot, input_img_path)
        self.input_json = os.path.join(args.dataroot, args.visdial_params)
        self.input_ques = os.path.join(args.dataroot, args.visdial_data)
        self.input_dialog = os.path.join(
            args.dataroot, getattr(args, 'dialog_%s' % split))
        self.dense_annotations_jsonpath = os.path.join(
            args.dataroot, args.dense_annotations)
        self.num_data = getattr(args, 'num_%s' % split)
        self.use_img_id_idx = None

        # preprocessing split
        print("\nProcessing split [{}]...".format(self.__split))
        
        print("Dataloader loading json file: {}".format(self.input_json))
        with open(self.input_json, 'r') as info_file:
            info = json.load(info_file)
            # possible keys: {'ind2word', 'word2ind', 'unique_img_(split)'}
            for key, value in iteritems(info):
                setattr(self, key, value)

        # add <START> and <END> to vocabulary
        word_count = len(self.word2ind)
        self.word2ind['<START>'] = word_count + 1
        self.word2ind['<END>'] = word_count + 2
        self.start_token = self.word2ind['<START>']
        self.end_token = self.word2ind['<END>']

        # padding + <START> + <END> token
        self.vocab_size = word_count + 3
        print("Vocab size with <START>, <END>: {}".format(self.vocab_size))

        # construct reverse of word2ind after adding tokens
        self.ind2word = {
            int(ind): word_count
            for word, ind in iteritems(self.word2ind)
        }

        print("Dataloader loading image h5 file: {}".format(self.input_img))
        # Either img_feats or img_reader will be set.
        if self.__version == '0.9':
            # trainval image features
            with h5py.File(self.input_img, 'r') as img_hdf5:
                img_feats_h5 = img_hdf5.get('images_%s' % img_split)
                self.num_data_points = len(img_feats_h5) - self.img_start_idx
                self.img_reader = None
                if self.__split == 'train':
                    self.num_data_points = min(self.num_data_points, TRAIN_VAL_SPLIT[self.__version])
        else:
            # split image features
            self.use_img_id_idx = True
            self.img_reader = ImageFeaturesHdfReader(
                self.input_img, in_memory=self.__in_memory)
            self.num_data_points = len(self.img_reader)

        if self.num_data is not None:
            self.num_data_points = min(self.num_data, self.num_data_points)

        self.img_end_idx = self.img_start_idx + self.num_data_points
        self.data_end_idx = self.data_start_idx + self.num_data_points

        if self.img_reader is None:
            with h5py.File(self.input_img, 'r') as img_hdf5:
                img_feats_h5 = img_hdf5.get('images_%s' % img_split)
                self.img_feats = torch.from_numpy(
                    np.array(img_feats_h5[self.img_start_idx:self.img_end_idx]))
        
        if 'val' == self.__split and os.path.exists(self.dense_annotations_jsonpath):
            self.use_img_id_idx = True
            self.annotations_reader = DenseAnnotationsReader(
                self.dense_annotations_jsonpath)
        else:
            self.annotations_reader = None

        if self.use_img_id_idx:
            print('Loading input dialog json: {}'.format(self.input_dialog))
            with open(self.input_dialog, 'r') as dialog_json:
                visdial_data = json.load(dialog_json)
                self.idx2imgid = [dialog_for_image['image_id']
                                  for dialog_for_image in visdial_data['data']['dialogs']]

        print("Dataloader loading h5 file: {}".format(self.input_ques))
        ques_file = h5py.File(self.input_ques, 'r')

        # load all data mats from ques_file into this
        self.data = {}

        self.img_norm = args.img_norm
        img_fnames = getattr(self, 'unique_img_' + data_split)
        self.data[self.__split + '_img_fnames'] = img_fnames[self.data_start_idx:self.data_end_idx]

        # map from load to save labels
        io_map = {
            'ques_{}': '{}_ques',
            'ques_length_{}': '{}_ques_len',
            'ans_{}': '{}_ans',
            'ans_length_{}': '{}_ans_len',
            'img_pos_{}': '{}_img_pos',
            'cap_{}': '{}_cap',
            'cap_length_{}': '{}_cap_len',
            'opt_{}': '{}_opt',
            'opt_length_{}': '{}_opt_len',
            'opt_list_{}': '{}_opt_list',
            'num_rounds_{}': '{}_num_rounds',
            'ans_index_{}': '{}_ans_ind'
        }

        # read the question, answer, option related information
        for load_label, save_label in iteritems(io_map):
            label = load_label.format(data_split)
            if load_label.format(data_split) not in ques_file:
                continue
            if label.startswith('opt_list') or label.startswith('opt_length'):
                if self.__version == '1.0' and self.__split == 'val':
                    label = load_label.format('test')
                self.data[save_label.format(self.__split)] = torch.from_numpy(
                    np.array(ques_file[label], dtype='int64'))
            else:
                self.data[save_label.format(self.__split)] = torch.from_numpy(
                np.array(ques_file[label][self.data_start_idx:self.data_end_idx], dtype='int64'))

        ques_file.close()

        # record some stats, will be transferred to encoder/decoder later
        # assume similar stats across multiple data subsets
        # maximum number of questions per image, ideally 10
        self.max_ques_count = self.data[self.__split + '_ques'].size(1)
        # maximum length of question
        self.max_ques_len = self.data[self.__split + '_ques'].size(2)
        # maximum length of answer
        self.max_ans_len = self.data[self.__split + '_ans'].size(2)

        print("[{0}] no. of data points: {1}".format(
            self.__split, self.num_data_points))
        print("\tMax no. of rounds: {}".format(self.max_ques_count))
        print("\tMax ques len: {}".format(self.max_ques_len))
        print("\tMax ans len: {}".format(self.max_ans_len))

        # prepare history
        self._process_history(self.__split)
        # 1 indexed to 0 indexed
        self.data[self.__split + '_opt'] -= 1
        if self.__split + '_ans_ind' in self.data:
            self.data[self.__split + '_ans_ind'] -= 1

    @property
    def split(self):
        return self.__split

    # ------------------------------------------------------------------------
    # methods to override - __len__ and __getitem__ methods
    # ------------------------------------------------------------------------

    def __len__(self):
        return self.num_data_points

    def __getitem__(self, idx):
        dtype = self.__split
        item = {'index': idx}
        item['num_rounds'] = self.data[dtype + '_num_rounds'][idx]

        # get image features
        if self.use_img_id_idx:
            image_id = self.idx2imgid[idx]
            item['image_id'] = torch.tensor(image_id).long()
        if self.img_reader is None:
            img_feats = self.img_feats[idx]
        else:
            img_feats = torch.tensor(self.img_reader[image_id])
        if self.img_norm:
            img_feats = F.normalize(img_feats, dim=0, p=2)
        item['img_feat'] = img_feats
        item['img_fnames'] = self.data[dtype + '_img_fnames'][idx]

        # get question tokens
        item['ques'] = self.data[dtype + '_ques'][idx]
        item['ques_len'] = self.data[dtype + '_ques_len'][idx]

        # get history tokens
        item['hist_len'] = self.data[dtype + '_hist_len'][idx]
        item['hist'] = self.data[dtype + '_hist'][idx]

        # get caption tokens
        item['cap'] = self.data[dtype + '_cap'][idx]
        item['cap_len'] = self.data[dtype + '_cap_len'][idx]

        # get answer tokens
        item['ans'] = self.data[dtype + '_ans'][idx]
        item['ans_len'] = self.data[dtype + '_ans_len'][idx]

        # get options tokens
        opt_inds = self.data[dtype + '_opt'][idx]
        opt_size = list(opt_inds.size())
        new_size = torch.Size(opt_size + [-1])
        ind_vector = opt_inds.view(-1)
        option_in = self.data[dtype + '_opt_list'].index_select(0, ind_vector)
        option_in = option_in.view(new_size)

        opt_len = self.data[dtype + '_opt_len'].index_select(0, ind_vector)
        opt_len = opt_len.view(opt_size)

        item['opt'] = option_in
        item['opt_len'] = opt_len
        if dtype != 'test':
            ans_ind = self.data[dtype + '_ans_ind'][idx]
            item['ans_ind'] = ans_ind.view(-1)

        if dtype == 'val' and self.annotations_reader is not None:
            dense_annotations = self.annotations_reader[image_id]
            item['gt_relevance'] = torch.tensor(
                dense_annotations["gt_relevance"]).float()
            item['round_id'] = torch.tensor(
                dense_annotations['round_id']).long()

        # convert zero length sequences to one length
        # this is for handling empty rounds of v1.0 test, they will be dropped anyway
        if dtype == 'test':
            item['ques_len'][item['ques_len'] == 0] += 1
            item['opt_len'][item['opt_len'] == 0] += 1
            item['hist_len'][item['hist_len'] == 0] += 1
        return item

    # -------------------------------------------------------------------------
    # collate function utilized by dataloader for batching
    # -------------------------------------------------------------------------

    def collate_fn(self, batch):
        dtype = self.__split
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}
        for key in merged_batch:
            if key in {'index', 'num_rounds', 'img_fnames'}:
                out[key] = merged_batch[key]
            elif key in {'cap_len'}:
                out[key] = torch.Tensor(merged_batch[key]).long()
            else:
                out[key] = torch.stack(merged_batch[key], 0)

        # Dynamic shaping of padded batch
        out['hist'] = out['hist'][:, :, :torch.max(out['hist_len'])].contiguous()
        out['ques'] = out['ques'][:, :, :torch.max(out['ques_len'])].contiguous()
        out['ans'] = out['ans'][:, :, :torch.max(out['ans_len'])].contiguous()
        out['cap'] = out['cap'][:, :torch.max(out['cap_len'])].contiguous()

        out['opt'] = out['opt'][:, :, :, :torch.max(out['opt_len'])].contiguous()

        batch_keys = ['num_rounds', 'img_feat', 'img_fnames', 'hist', 'hist_len', 'ques', 'ques_len',
                      'ans', 'ans_len', 'cap', 'cap_len', 'opt', 'opt_len']
        if dtype != 'test':
            batch_keys.append('ans_ind')

        if dtype == 'val' and self.annotations_reader is not None:
            batch_keys.append('gt_relevance')
            batch_keys.append('round_id')
        return {key: out[key] for key in batch_keys}

    # -------------------------------------------------------------------------
    # preprocessing functions
    # -------------------------------------------------------------------------

    def _process_history(self, dtype):
        """
        Process caption as well as history. Optionally, concatenate history
        for lf-encoder.
        """
        captions = self.data[dtype + '_cap']
        questions = self.data[dtype + '_ques']
        ques_len = self.data[dtype + '_ques_len']
        cap_len = self.data[dtype + '_cap_len']
        max_ques_len = questions.size(2)

        answers = self.data[dtype + '_ans']
        ans_len = self.data[dtype + '_ans_len']
        num_convs, num_rounds, max_ans_len = answers.size()

        if self.args.concat_history:
            self.max_hist_len = min(
                num_rounds * (max_ques_len + max_ans_len), 300)
            history = torch.zeros(num_convs, num_rounds,
                                  self.max_hist_len).long()
        else:
            history = torch.zeros(num_convs, num_rounds,
                                  max_ques_len + max_ans_len).long()
        hist_len = torch.zeros(num_convs, num_rounds).long()

        # go over each question and append it with answer
        for th_id in range(num_convs):
            clen = cap_len[th_id]
            hlen = min(clen, max_ques_len + max_ans_len)
            for round_id in range(num_rounds):
                if round_id == 0:
                    # first round has caption as history
                    history[th_id][round_id][:max_ques_len + max_ans_len] \
                        = captions[th_id][:max_ques_len + max_ans_len]
                else:
                    qlen = ques_len[th_id][round_id - 1]
                    alen = ans_len[th_id][round_id - 1]
                    # if concat_history, string together all previous question-answer pairs
                    if self.args.concat_history:
                        history[th_id][round_id][:hlen] = history[th_id][round_id - 1][:hlen]
                        history[th_id][round_id][hlen] = self.word2ind['<END>']
                        if qlen > 0:
                            history[th_id][round_id][hlen + 1:hlen + qlen + 1] \
                                = questions[th_id][round_id - 1][:qlen]
                        if alen > 0:
                            # print(round_id, history[th_id][round_id][:10], answers[th_id][round_id][:10])
                            history[th_id][round_id][hlen + qlen + 1:hlen + qlen + alen + 1] \
                                = answers[th_id][round_id - 1][:alen]
                        hlen = hlen + qlen + alen + 1
                    # else, history is just previous round question-answer pair
                    else:
                        if qlen > 0:
                            history[th_id][round_id][:qlen] = questions[th_id][round_id - 1][:qlen]
                        if alen > 0:
                            history[th_id][round_id][qlen:qlen + alen] \
                                = answers[th_id][round_id - 1][:alen]
                        hlen = alen + qlen
                # save the history length
                hist_len[th_id][round_id] = hlen

        self.data[dtype + '_hist'] = history
        self.data[dtype + '_hist_len'] = hist_len
