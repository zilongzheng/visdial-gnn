import argparse
import os
import time
import torch

class BaseOptions():
    """Basic model options"""

    def __init__(self):
        self.initialized = False
        self.isTrain = None

    def initialize(self, parser):
        parser.add_argument_group('Dataloader specific arguments')
        parser.add_argument('--dataroot', default='./data/v1.0', help='path to dataroot')
        parser.add_argument('--img_train', default='features_faster_rcnn_x101_train.h5', help='HDF5 file with image features')
        parser.add_argument('--img_val', default='features_faster_rcnn_x101_val.h5', help='HDF5 file with image features')
        parser.add_argument('--visdial_data', default='visdial_data_trainval.h5', help='HDF5 file with preprocessed questions')
        parser.add_argument('--visdial_params', default='visdial_params_trainval.json', help='JSON file with image paths and vocab')
        parser.add_argument('--dialog_train', default='visdial_1.0_train.json', help='JSON file with image paths and vocab')
        parser.add_argument('--dialog_val', default='visdial_1.0_val.json', help='JSON file with image paths and vocab')
        parser.add_argument('--dense_annotations', default='visdial_1.0_val_dense_annotations.json', help='path to visdial val dense annotations (if evaluating on val split)')
        parser.add_argument('--version', default='1.0', choices=['0.9', '1.0'], help='dataset version [0.9|1.0]')
        parser.add_argument('--img_norm', default=1, choices=[1, 0], help='normalize the image feature. 1=yes, 0=no')
        parser.add_argument('--in_memory', action='store_true', help='whether to load data in memory')
        parser.add_argument('--concat_history', action='store_true', help='whether to concat history')
        parser.add_argument('--num_train', default=None, type=int, help='number of datapoints to train [None for all data]')
        parser.add_argument('--num_val', default=None, type=int, help='number of datapoints to validate [None for all data]')
        parser.add_argument('--gpuid', default=0, type=int, help='GPU id to use, use -1 for CPU')
        # Model settings
        parser.add_argument_group('Model related arguments')
        parser.add_argument('--img_feat_size', default=2048, type=int, help='channel size of input image feature')
        parser.add_argument('--embed_size', default=300, type=int, help='size of the input word embedding')
        parser.add_argument('--rnn_hidden_size', default=512, type=int, help='size of the multimodal embedding')
        parser.add_argument('--num_layers', default=2, type=int, help='number of layers in LSTM')
        parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
        parser.add_argument('--message_size', default=512, type=int, help='message passing size')
        parser.add_argument('--m_step', default=3, type=int, help='number of steps for EM-Learning')
        parser.add_argument('--e_step', default=1, type=int, help='number of propagation steps for message passing')

        # Logging settings
        parser.add_argument_group('Checkpointing related arguments')
        parser.add_argument('--save_path', default='outputs/', help='path to save outputs')
        parser.add_argument('--ckpt', default=None, help='path to load checkpoints')
        self.initialized = True
        return parser

    def parse(self):
        """Initialize and parse arguments"""
        if not self.initialized:
            parser = argparse.ArgumentParser(description='Structural Visual Dialog Reasoning Model')
            parser = self.initialize(parser)

        opt = parser.parse_args()
        opt.isTrain = self.isTrain
        opt.save_path = os.path.join(opt.save_path, time.strftime('%Y-%m-%d_%H-%M-%S'))
        
        self.opt = opt
        return opt

    def print_options(self, opt, log_to_file=True):
        """Print options"""
        message = ''
        message += '===========================================\n'
        for k, v in sorted(vars(opt).items()):
            message += '{:<20}: {}\n'.format(str(k), str(v))
        message += '===========================================\n'

        print(message)

        if log_to_file:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)

            log_file = os.path.join(opt.save_path, 'log.txt')
            with open(log_file, 'wt') as log_file:
                log_file.write(message)
                log_file.write('\n')

            opt.log_file = log_file
        return opt
