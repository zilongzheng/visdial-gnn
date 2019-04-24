from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """Testing Options"""

    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument_group('Testing specific arguments')
        parser.add_argument('--input_img_test', default='features_faster_rcnn_x101_test.h5', help='HDF5 fiile with testing image features')
        parser.add_argument('--num_test', default=None, type=int, help='number of datapoints to test [None for all data]')
        parser.add_argument('--split', default='test', choices=['val', 'test'])
        parser.add_argument('--save_rank_path', default='logs/ranks.json', help='path of json files to save ranks')
        self.isTrain = False
        return parser
