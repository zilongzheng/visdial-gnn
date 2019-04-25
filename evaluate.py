import argparse
import datetime
import gc
import json
import math
import os
from tqdm import tqdm
import datetime
import torch
from torch.utils.data import DataLoader

from vdgnn.dataset.dataloader import VisDialDataset
from vdgnn.models.encoder import GCNNEncoder
from vdgnn.models.decoder import DiscriminativeDecoder
from vdgnn.options.test_options import TestOptions
from vdgnn.utils.eval_utils import process_ranks, scores_to_ranks, get_gt_ranks
from vdgnn.utils.metrics import NDCG


def eval(encoder, decoder, dataloader, args):
    print("Evaluation start time: {}".format(
        datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')))
    encoder.eval()
    decoder.eval()

    ndcg = NDCG()

    all_ranks = []

    if args.save_rank_path is not None:
        ranks_json = []

    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                if args.use_cuda:
                    batch[key] = batch[key].cuda()

        batch_size, max_num_rounds = batch['ques'].size()[:2]

        enc_output = torch.zeros(batch_size, max_num_rounds, model_args.message_size, requires_grad=True)

        if args.use_cuda:
            enc_output = enc_output.cuda()
        # iterate over dialog rounds
        with torch.no_grad():
            for rnd in range(max_num_rounds):
                round_info = {}

                round_info['img_feat'] = batch['img_feat']

                round_info['ques'] = batch['ques'][:, rnd, :]
                round_info['ques_len'] = batch['ques_len'][:, rnd]

                round_info['hist'] = batch['hist'][:,:rnd+1, :]
                round_info['hist_len'] = batch['hist_len'][:, :rnd+1]
                round_info['round'] = rnd

                pred_adj_mat, enc_out = encoder(round_info, args)
                enc_output[:, rnd, :] = enc_out
            
            dec_out = decoder(enc_output.contiguous().view(-1, model_args.message_size), batch)
            ranks = scores_to_ranks(dec_out.data)

        if args.split == 'val':
            gt_ranks = get_gt_ranks(ranks, batch['ans_ind'].data)
            all_ranks.append(gt_ranks)
            if 'gt_relevance' in batch:
                output = dec_out.view(batch_size, max_num_rounds, dec_out.size(1))
                output = output[torch.arange(batch_size), batch['round_id'] - 1, :]
                ndcg.observe(output, batch['gt_relevance'])
                
        if args.save_rank_path is not None:
            ranks = ranks.view(-1, 10, 100)

            for i in range(len(batch['img_fnames'])):
                # cast into types explicitly to ensure no errors in schema
                if args.split == 'test':
                    ranks_json.append({
                        'image_id': int(batch['img_fnames'][i][-16:-4]),
                        'round_id': int(batch['num_rounds'][i].item()),
                        'ranks': [rank.item() for rank in ranks[i][batch['num_rounds'][i] - 1]]
                    })
                else:
                    for j in range(batch['num_rounds'][i]):
                        ranks_json.append({
                            'image_id': int(batch['img_fnames'][i][-16:-4]),
                            'round_id': int(j + 1),
                            'ranks': list(ranks[i][j].cpu().numpy())
                        })
                        
            gc.collect()

    if args.split == 'val':    
        all_ranks = torch.cat(all_ranks, 0)
        res = process_ranks(all_ranks)
        res['ndcg'] = ndcg.retrieve(reset=True)
        print("\tNo. questions: {}".format(res['num_ques']))
        print('\tndcg: {}'.format(res['ndcg']))
        print("\tr@1: {}".format(res['r_1']))
        print("\tr@5: {}".format(res['r_5']))
        print("\tr@10: {}".format(res['r_10']))
        print("\tmeanR: {}".format(res['mr']))
        print("\tmeanRR: {}".format(res['mrr']))
        gc.collect()

    if args.save_rank_path is not None:
        print("Writing ranks to {}".format(args.save_rank_path))
        save_dir = os.path.dirname(args.save_rank_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        json.dump(ranks_json, open(args.save_rank_path, 'w'))

if __name__ == '__main__':
    # For reproducibility
    RANDOM_SEED = 0
    torch.manual_seed(RANDOM_SEED)
    
    args = TestOptions().parse()
    if args.gpuid >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.cuda.set_device(args.gpuid)
        args.use_cuda = True

    # ----------------------------------------------------------------------------
    # read saved model and args
    # ----------------------------------------------------------------------------
    components = torch.load(args.ckpt)
    model_args = components['model_args']
    model_args.gpuid = args.gpuid
    args.batch_size = model_args.batch_size

    # this is required by dataloader
    args.img_norm = model_args.img_norm

    TestOptions().print_options(args, log_to_file=False)

    # ----------------------------------------------------------------------------
    # loading dataset wrapping with a dataloader
    # ----------------------------------------------------------------------------

    dataset = VisDialDataset(args, args.split, isTrain=True)
    dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn)

    # ----------------------------------------------------------------------------
    # setup the model
    # ----------------------------------------------------------------------------

    encoder = GCNNEncoder(model_args)
    encoder.load_state_dict(components['encoder'])

    decoder = DiscriminativeDecoder(model_args, encoder)
    decoder.load_state_dict(components['decoder'])
    print("Loaded model from {}".format(args.ckpt))

    if args.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    eval(encoder, decoder, dataloader, args)

