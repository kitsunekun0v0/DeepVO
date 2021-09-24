import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from params import par
from utils import EarlyStopping  #, adjust_lr
from model import DeepVO
from dataloader import Kitti


def trainer(model, dataloader, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss = 0
    for i, (_, x, y) in enumerate(dataloader):
        print('{} / {}'.format(i, len(dataloader)), end='\r', flush=True)
        x = x.cuda()
        y = y.cuda()
        ls = model.get_loss(x, y)

        if train:
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        loss += float(ls.data.cpu().numpy())
    return loss


def get_dataloader(video, seq_len, overlap, sample_times, batch_size, im_size, n_workers, shuffle=True):
    dataset = Kitti(video, seq_len, overlap, sample_times, shuffle=shuffle, new_size=im_size, include_global=False)
    return DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle), dataset


# Load Data
print('Create training sequences')
train_dl, train_dataset = get_dataloader(par.train_video, par.seq_len, par.overlap, par.sample_times,
                          par.batch_size, par.im_size, par.n_workers)
valid_dl, valid_dataset = get_dataloader(par.train_video, par.seq_len, par.val_seq_len[0]-1, 1,
                          par.batch_size, par.im_size, par.n_workers)

print('Number of samples in training dataset: ', len(train_dataset))
print('Number of samples in validation dataset: ', len(valid_dataset))
print('=' * 50)

# Model
model = DeepVO(par.img_h, par.img_w, par.batch_norm)
model = model.cuda()

if par.pretrained:
    print('Loading pretrained FlowNet')
    flownet_w = torch.load(par.pretrained)
    model_dict = model.state_dict()
    update_dict = {k: v for k, v in flownet_w['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=par.lr)

es = EarlyStopping()
min_loss_t = 1e8

for ep in range(par.epochs):
    print('=' * 50)
    print('epoch {} / {}'.format(ep, par.epochs))

    # Train
    st_t = time.time()
    loss_mean = trainer(model, train_dl, optimizer, train=True)
    print('Train time: {:.1f} sec'.format(time.time() - st_t))
    loss_mean /= len(train_dl)

    # Validation
    st_t = time.time()
    model.eval()
    loss_mean_valid = trainer(model, valid_dl, optimizer, train=False)
    print('Validation time: {:.1f} sec'.format(time.time() - st_t))
    loss_mean_valid /= len(valid_dl)

    print('train loss mean: ', loss_mean)
    print('validation loss mean: ', loss_mean_valid)
    if es.step(loss_mean_valid):  # early stop if validation loss doesn't decrease
        print('early stop')
        break

    # Save model
    if loss_mean < min_loss_t:
        min_loss_t = loss_mean
        print('Model saved.')
        torch.save(model.state_dict(), par.save_model_path)
