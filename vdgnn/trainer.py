import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import gc
import os
import math
import time
from tqdm import tqdm
from vdgnn.utils.eval_utils import process_ranks, scores_to_ranks, get_gt_ranks
from vdgnn.utils.metrics import NDCG

class Trainer(object):
    def __init__(self, dataloader, dataloader_val, model_args):
        self.args = model_args
        self.output_dir = model_args.save_path

        self.num_epochs = model_args.num_epochs
        self.lr = model_args.lr
        self.lr_decay_rate = model_args.lr_decay_rate
        self.min_lr = model_args.min_lr
        self.ckpt = model_args.ckpt
        self.use_cuda = model_args.use_cuda
        self.log_step = model_args.log_step

        self.dataloader = dataloader
        self.dataloader_val = dataloader_val

        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

    def train(self, encoder, decoder):

        criterion = nn.CrossEntropyLoss()
        running_loss = None

        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                lr=self.lr)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay_rate)

        if self.ckpt != None:
            components = torch.load(self.ckpt)
            print('Loaded checkpoint from: ' + self.ckpt)
            encoder.load_state_dict(components['encoder'])
            decoder.load_state_dict(components['decoder'])

        if self.use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            criterion = criterion.cuda()

        for epoch in range(1, self.num_epochs+1):
            epoch_time = time.time()

            encoder.train()
            decoder.train()
            iter_time = time.time()

            for iter, batch in enumerate(self.dataloader):
                optimizer.zero_grad()

                for key in batch:
                    if not isinstance(batch[key], list):
                        if self.use_cuda:
                            batch[key] = batch[key].cuda()

                batch_size, max_num_rounds = batch['ques'].size()[:2]

                enc_output = torch.zeros(batch_size, max_num_rounds, self.args.message_size, requires_grad=True)

                if self.use_cuda:
                    enc_output = enc_output.cuda()

                # iterate over dialog rounds
                for rnd in range(max_num_rounds):
                    round_info = {}

                    round_info['img_feat'] = batch['img_feat']

                    round_info['ques'] = batch['ques'][:, rnd, :]
                    round_info['ques_len'] = batch['ques_len'][:, rnd]

                    round_info['hist'] = batch['hist'][:,:rnd+1, :]
                    round_info['hist_len'] = batch['hist_len'][:, :rnd+1]
                    round_info['round'] = rnd

                    pred_adj_mat, enc_out = encoder(round_info, self.args)

                    enc_output[:, rnd, :] = enc_out

                dec_out = decoder(enc_output.contiguous().view(-1, self.args.message_size), batch)

                cur_loss = criterion(dec_out, batch['ans_ind'].view(-1))
                cur_loss.backward()

                optimizer.step()
                gc.collect()

                if running_loss is not None:
                    running_loss = 0.95 * running_loss + 0.05 * cur_loss.data
                else:
                    running_loss = cur_loss.data

                if optimizer.param_groups[0]['lr'] > self.min_lr:
                    scheduler.step()

                # --------------------------------------------------------------------
                # Logging
                # --------------------------------------------------------------------
                if (iter+1) % self.log_step == 0:
                    print("[Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:6f}][Duration: {:6.2f}s]".format(
                        epoch, iter+1, running_loss, optimizer.param_groups[0]['lr'], time.time() - iter_time))
                    iter_time = time.time()

            print("[Epoch: {:3d}][Loss: {:6f}][lr: {:6f}][Time: {:6.2f}s]".format(
                        epoch, running_loss, optimizer.param_groups[0]['lr'], time.time() - epoch_time))

            # --------------------------------------------------------------------
            # Save checkpoints
            # --------------------------------------------------------------------
            if epoch % 1 == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)

                torch.save({
                    'encoder':encoder.state_dict(),
                    'decoder':decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': self.args
                }, os.path.join(self.model_dir, 'model_epoch_{:06d}.pth'.format(epoch)))

                val_res = self.validate(encoder, decoder, epoch)

        torch.save({
            'encoder':encoder.state_dict(),
            'decoder':decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': self.args
        }, os.path.join(self.model_dir, 'model_epoch_final.pth'))

    def validate(self, encoder, decoder, epoch):
        print('Evaluating...')
        encoder.eval()
        decoder.eval()
        ndcg = NDCG()

        eval_time = time.time()
        all_ranks = []

        for i, batch in enumerate(tqdm(self.dataloader_val)):

            for key in batch:
                if not isinstance(batch[key], list):
                    if self.use_cuda:
                        batch[key] = batch[key].cuda()

            batch_size, max_num_rounds = batch['ques'].size()[:2]

            enc_output = torch.zeros(batch_size, max_num_rounds, self.args.message_size, requires_grad=True)

            if self.use_cuda:
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

                    pred_adj_mat, enc_out = encoder(round_info, self.args)

                    enc_output[:, rnd, :] = enc_out

                dec_out = decoder(enc_output.contiguous().view(-1, self.args.message_size), batch)
                ranks = scores_to_ranks(dec_out.data)
                gt_ranks = get_gt_ranks(ranks, batch['ans_ind'].data)
                all_ranks.append(gt_ranks)
                num_opts = dec_out.size(1)
                output = dec_out.view(batch_size, max_num_rounds, num_opts)
                output = output[torch.arange(batch_size), batch['round_id']-1, :]
                if 'gt_relevance' in batch:
                    ndcg.observe(output, batch['gt_relevance'])

        all_ranks = torch.cat(all_ranks, 0)
        eval_res = process_ranks(all_ranks)
        eval_res['ndcg'] = ndcg.retrieve(reset=True)

        print("[Epoch: {:3d}][R@1: {:6f}][R@5: {:6f}][R@10: {:6f}][MR: {:6f}][MRR: {:6f}][NDCG: {:6f}][Time: {:6.2f}s]".format(epoch,
                        eval_res['r_1'],
                        eval_res['r_5'],
                        eval_res['r_10'],
                        eval_res['mr'],
                        eval_res['mrr'],
                        eval_res['ndcg'],
                        time.time() - eval_time))
        gc.collect()

        return eval_res


