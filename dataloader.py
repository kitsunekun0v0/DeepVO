import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
from params import par
from utils import create_pose_data, cal_rel_pose


def get_data_info(folder_list, seq_len, overlap, sample_times=1, shuffle=False):
    X_path, Y = [], []

    def sample_index(seq_len, n_frames, stride):
        n_samples = (n_frames + stride - seq_len) / stride
        n_samples = int(n_samples)
        end = int(n_samples * stride)
        return end

    for folder in folder_list:
        start_t = time.time()
        poses = create_pose_data(par.pose_dir, folder)
        fpaths = glob.glob('{}{}/*.png'.format(par.image_dir, folder))
        fpaths.sort()

        if sample_times > 1:
            sample_interval = int(np.ceil(seq_len / sample_times))
            start_frames = list(range(0, seq_len, sample_interval))
            print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for st in start_frames:
            n_frames = len(fpaths) - st
            jump = seq_len - overlap
            end = sample_index(seq_len, n_frames, jump)
            x_segs = [np.asarray(fpaths[i:i + seq_len]) for i in range(st, end, jump)]
            y_segs = [poses[i:i + seq_len] for i in range(st, end, jump)]
            Y += y_segs
            X_path += x_segs
        print('Folder {} finish in {} sec'.format(folder, time.time() - start_t))

    # Convert to pandas dataframes
    data = {'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns=['image_path', 'pose'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    return df


class Kitti(Dataset):
    def __init__(self, video_ls, seq_len, overlap, sample_times=1, shuffle=False, new_size=None, include_global=False):
        """
        info_dataframe: data information
        new_size: resize image, tuple (h, w)
        include_global: if include global poses
        return: seq_len, image seq, relative pose seq, global pose seq (if include_global=True)
        """
        # image process
        transform_ops = []
        transform_ops.append((transforms.Resize((new_size[0], new_size[1]))))
        transform_ops.append(transforms.ToTensor())
        transform_ops.append(transforms.Normalize(mean=[0.45, 0.432, 0.411], std=[1, 1, 1]))
        self.transformer = transforms.Compose(transform_ops)

        self.data_info = get_data_info(video_ls, seq_len, overlap, sample_times, shuffle)
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)
        self.include_grobal = include_global

    def __getitem__(self, index):
        groundtruth_sequence = self.groundtruth_arr[index]
        R_seq = groundtruth_sequence[:, 3:]
        tr_seq = groundtruth_sequence[:, :3]

        # R = R1.T * R2, t = R1.T(t2 - t1)
        # relative poses.
        R1 = R_seq[0].reshape((3, 3)).T
        t1 = tr_seq[0]
        pose = np.zeros((len(R_seq), 6))

        for i in range(len(R_seq)):
            R2 = R_seq[i].reshape((3, 3))
            t2 = tr_seq[i]
            euler, t = cal_rel_pose(R1, R2, t1, t2)
            pose[i, :3] = euler
            pose[i, 3:] = t
            R1 = R2.T
            t1 = t2

        # absolute poses.
        R1 = R_seq[0].reshape((3, 3)).T
        t1 = tr_seq[0]
        abs_pose = np.zeros((len(R_seq), 6))

        for i in range(len(R_seq)):
            R2 = R_seq[i].reshape((3, 3))
            t2 = tr_seq[i]
            euler, t = cal_rel_pose(R1, R2, t1, t2)
            abs_pose[i, :3] = euler
            abs_pose[i, 3:] = t

        pose = torch.FloatTensor(pose)
        abs_pose = torch.FloatTensor(abs_pose)

        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])

        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            image_sequence.append(img_as_tensor.unsqueeze(0))
        image_sequence = torch.cat(image_sequence, 0)

        if self.include_grobal:
            return sequence_len, image_sequence, pose, abs_pose
        else:
            return sequence_len, image_sequence, pose

    def __len__(self):
        return len(self.data_info.index)


