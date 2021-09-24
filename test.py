from params import par
from model import DeepVO
import numpy as np
import glob
import os
import time
import torch
from dataloader import Kitti
from torch.utils.data import DataLoader
from utils import euler2rotation, rotation2euler

if __name__ == '__main__':

    videos_to_test = par.test_video

    # Path
    load_model_path = par.load_model_path  # choose the model you want to load
    save_dir = 'result/'  # directory to save prediction answer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    M_deepvo.load_state_dict(torch.load(load_model_path))
    use_cuda = torch.cuda.is_available()
    M_deepvo = M_deepvo.cuda()
    M_deepvo.load_state_dict(torch.load(load_model_path))
    print('Load model from: ', load_model_path)
    M_deepvo.eval()

    # Data
    n_workers = 8
    seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)
    overlap = seq_len - 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))
    batch_size = par.batch_size

    for test_video in videos_to_test:
        dataset = Kitti([test_video], seq_len, overlap, shuffle=False, new_size=par.im_size, include_global=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        # Predict
        answer = [[0., 0., 0., 0., 0., 0.]]
        st_t = time.time()
        n_batch = len(dataloader)

        R1 = np.eye(3)
        t1 = np.zeros(3)

        for i, batch in enumerate(dataloader):
            print('{} / {}'.format(i, n_batch), end='\r', flush=True)
            _, x, y = batch
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            batch_predict_pose = M_deepvo.forward(x)
            batch_predict_pose = batch_predict_pose.data.cpu().numpy()

            # R = R1 * R2, t = R1 * t2 + t1
            for rel_seq in batch_predict_pose:
                for pose in rel_seq:
                    t12 = pose[3:]
                    R12 = euler2rotation(pose[:3])
                    R2 = R1.dot(R12)
                    t2 = R1.dot(t12) + t1
                    Rt2 = np.hstack((rotation2euler(R2), t2))
                    answer.append(Rt2.tolist())
                    R1 = R2
                    t1 = t2


        print('len(answer): ', len(answer))
        print('expect len: ', len(glob.glob('{}{}/*.png'.format(par.image_dir, test_video))))
        print('Predict use {} sec'.format(time.time() - st_t))
        print('=' * 50)

        # Save answer
        with open('{}/pred_{}.txt'.format(save_dir, test_video), 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(', '.join([str(p) for p in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')
