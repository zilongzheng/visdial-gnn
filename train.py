import torch
import argparse
import time
import os
from vdgnn.dataset.dataloader import VisDialDataset
from torch.utils.data import DataLoader
from vdgnn.models.encoder import GCNNEncoder
from vdgnn.models.decoder import DiscriminativeDecoder
from vdgnn.options.train_options import TrainOptions
from vdgnn.trainer import Trainer

if __name__ == '__main__':
    # For reproducibility
    RANDOM_SEED = 0
    torch.manual_seed(RANDOM_SEED)
    
    opts = TrainOptions().parse()
    if opts.gpuid >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.cuda.set_device(opts.gpuid)
        opts.use_cuda = True
    
    TrainOptions().print_options(opts)
    
    dataset = VisDialDataset(opts, 'train', isTrain=True)
    dataset_val = VisDialDataset(opts, 'val', isTrain=True)
    dataloader = DataLoader(dataset,
                            batch_size=opts.batch_size,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=dataset.collate_fn)

    dataloader_val = DataLoader(dataset_val,
                                batch_size=opts.batch_size,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=dataset_val.collate_fn)

    # transfer all options to model
    model_args  = opts

    for key in {'num_data_points', 'vocab_size', 'max_ques_count',
            'max_ques_len', 'max_ans_len'}:
        setattr(model_args, key, getattr(dataset, key))

    encoder = GCNNEncoder(model_args)

    decoder = DiscriminativeDecoder(model_args, encoder)

    trainer = Trainer(dataloader, dataloader_val, model_args)

    trainer.train(encoder, decoder)

