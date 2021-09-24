import os
import argparse
import datetime


parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('-data_dir', help='data directory', default='kitti')
parser.add_argument('-image_dir', help='image directory', default='kitti/images/')
parser.add_argument('-pose_dir', help='pose directory', default='kitti/pose_GT/')
parser.add_argument('-train_video', help='list of training sequences', default=['00', '02', '08', '09'])
parser.add_argument('-valid_video', help='list of validation sequences', default=['01'])
parser.add_argument('-test_video', help='list of validation sequences', default=['04', '05', '06', '07', '10', '03'])
parser.add_argument('-n_workers', help='num_workers', type=int, default=8)
parser.add_argument('-im_size', help='image size', default=(384, 1280))
parser.add_argument('-seq_len', help='length of train sub-sequence', default=(5, 7))
parser.add_argument('-val_seq_len', help='length of valid sub-sequence', default=(6, 6))
parser.add_argument('-sample_times', help='number of times sub-sequences are sampled', type=int, default=1)
parser.add_argument('-overlap', help='no. of overlaping frame between two sub-sequences', type=int, default=4)

# Model
parser.add_argument('-conv_dropout', help='dropout used in CNNs', default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2])
parser.add_argument('-rnn_dropout', help='dropout used after LSTMs', type=float, default=0.3)
parser.add_argument('-batch_norm', help='if use batch norm', type=bool, default=False)

# Pretrained
parser.add_argument('-pretrained', help='path to pretrained flownet',
                    choices = ['pretrained/flownets_EPE1.951.pth.tar', 'pretrained/flownets_bn_EPE2.459.pth.tar'],
                    default='pretrained/flownets_EPE1.951.pth.tar')

# Hyper parameter
parser.add_argument('-epochs', help='epochs', type=int, default=200)
parser.add_argument('-batch_size', help='batch size', type=int, default=16)
parser.add_argument('-lr', help='Learning rate', type=float, default=1e-4)

par = parser.parse_args()

# Load/Save model
model_path = 'models/deepvo_'
timestamp = datetime.datetime.now().strftime("%m_%d")
model_path = '{}{}'.format(model_path, timestamp)
if not os.path.isdir(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
par.model_path = model_path



