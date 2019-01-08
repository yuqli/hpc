import pandas as pd
import os
from PIL import Image
import time
import csv
import random
import pandas as pd
import numpy as np
import pdb

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
import argparse



class KaggleAmazonDataset(Dataset):

    def __init__(self, csv_path, img_path, img_ext, transform=None):

        tmp_df = pd.read_csv(csv_path)
        #assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \"Some images referenced in the CSV file were not found"

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform
        
        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags']
        
        self.num_labels = 17

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label_ids = self.y_train[index].split()
        label_ids = [ int(s) for s in label_ids ]
        label=torch.zeros(self.num_labels)
        label[label_ids] = 1
        return img, label

    def __len__(self):
        return len(self.X_train)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x=self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x=self.conv2(x)
        x=self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x , 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, train_loader, model, criterion, optimizer):
    loader_times = AverageMeter()
    batch_times = AverageMeter()
    losses = AverageMeter()
    precisions_1 = AverageMeter()
    precisions_k = AverageMeter()
    topk=3

    model.train()

    t_train = time.monotonic()
    t_batch = time.monotonic()
    for batch_idx, (data, target) in enumerate(train_loader):
        loader_time = time.monotonic() - t_batch
        loader_times.update(loader_time)
        # data = data.to(device=device)
        # target = target.to(device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        
        _, predicted = output.topk(topk, 1, True, True)
        batch_size = target.size(0)
        prec_k=0
        prec_1=0
        count_k=0
        for i in range(batch_size):
            prec_k += target[i][predicted[i]].sum()
            prec_1 += target[i][predicted[i][0]]
            count_k+=topk # min(target[i].sum(), topk), to have a fair topk precision
        prec_k/=(count_k)
        prec_1/=batch_size

        #Update of averaged metrics
        losses.update(loss.item(), 1)
        precisions_1.update(prec_1, 1)
        precisions_k.update(prec_k, 1)
        batch_times.update(time.monotonic() - t_batch)

        if (batch_idx+1) % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f} ({:.3f}),\tPrec@1: {:.3f} ({:.3f}),\tPrec@3: {:.3f} ({:.3f}),\tTimes: Batch: {:.4f} ({:.4f}),\tDataLoader: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), losses.val, losses.avg, precisions_1.val, precisions_1.avg , precisions_k.val, precisions_k.avg ,
                batch_times.val, batch_times.avg, loader_times.avg))

        t_batch = time.monotonic()

    train_time = time.monotonic() - t_train
    print('Training Epoch: {} done. \tLoss: {:.3f},\tPrec@1: {:.3f},\tPrec@3: {:.3f}\tTimes: Total: {:.3f}, Avg-Batch: {:.4f}, Avg-Loader: {:.4f}\n'.format(epoch, losses.avg, precisions_1.avg, precisions_k.avg, train_time, batch_times.avg, loader_times.avg))
    return  train_time, batch_times.avg, loader_times.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--use_cuda', type=str, default='true', help='Use CUDA if available')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='/scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon2/',
                        help='Data path')
    parser.add_argument('--opt', type=str, default='adam',
                        help='NN optimizer (Examples: adam, rmsprop, adadelta, ...)')

    args = parser.parse_args()
    """
    device = None
    if args.use_cuda.lower()=='true' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    """

    # print ('cuda_available:', torch.cuda.is_available(), 'device: ',device)
    DATA_PATH=args.data_path
    #DATA_PATH='data/'
    IMG_PATH = DATA_PATH+'train-jpg/'
    IMG_EXT = '.jpg'
    TRAIN_DATA = DATA_PATH+'train.csv'    
    batch_size=250

    # Timing variables.
    AggLoadBatchTime = 0.0;
    AggIOTime = 0.0;
    PreprocessingTime = 0.0;

    torch.manual_seed(123)

    model = Net() #.to(device=device)
    # pdb.set_trace()
    for param in model.parameters():
        param.grad = torch.zeros(param.size(), requires_grad=True)
        param.grad.data.zero_()

    print ('Optimizer:',args.opt)
    if args.opt=='adam':
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif args.opt=='adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    elif args.opt=='adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    elif args.opt=='nesterov':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=False)

    criterion = nn.BCELoss().to(device=device)

    transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    dset_train = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)
    train_loader = DataLoader(dset_train,
          batch_size=batch_size,
          shuffle=True,
          num_workers=args.workers # 1 for CUDA
         # pin_memory=True # CUDA only
         )
    print ('Dataset Loaded, tot samples: ', len(dset_train))
    # Send a zero gradient and receive the inital model.
    optimizer.zero_grad()
    
    # Do the actual training.
    TotalStrtTime = time.monotonic();
    train_times=[]
    batch_times=[]
    loader_times=[]
    for epoch in range(5):
        train_time, batch_time, loader_time = train(epoch, train_loader, model, criterion, optimizer)
        train_times.append(train_time)
        batch_times.append(batch_time)
        loader_times.append(loader_time)
        

    TotalTime = (time.monotonic()-TotalStrtTime)
    print('Time to perform 5 epochs is: %f' % TotalTime)

    print('Final Average Times: Total: {:.3f}, Avg-Batch: {:.4f}, Avg-Loader: {:.4f}\n'.format(np.average(train_times), np.average(batch_times), np.average(loader_times)))

