#!/anaconda/envs/nlp/bin/python

### 20181027
### Yuqiong Li
### HPC Lab2

from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
import pandas as pd

import argparse
import sys
import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image  # read images

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import time

############################################################################################
#                                   Data setup
############################################################################################
class PicturesDataset(Dataset):
    """Pictures dataset."""

    def __init__(self, csv_file, root_dir, transform, num_classes):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on sample.
        """
        self.pic_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes

    def resize(self, image):
        # Resize
        transformer = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor()])
        image = transformer(image)
        return image

    def __len__(self):
        """
        Return: how many pictures are in the dataset.
        """
        return len(self.pic_frame)

    def __getitem__(self, idx):
        # print(self.pic_frame.iloc[idx, 0])
        img_name = os.path.join(self.root_dir,
                                self.pic_frame.iloc[idx, 0] + '.jpg')

        # image = io.imread(img_name)
        image = Image.open(img_name)
        tags = self.pic_frame.iloc[idx, 1].split()
        tags = [int(x) for x in tags]
        tags = multi_hot(tags, self.num_classes)
        sample = {'image': image, 'tags': tags}

        if self.transform:
            sample['image'] = self.resize(sample['image'])

        return sample['image'], sample['tags']


############################################################################################
#                                   Network setup
############################################################################################

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(2304, 256),
            nn.Dropout(),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(-1, 2304)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def multi_hot(labels, n):
    """
    @labels : a 2D numpy array of indices for multi-hot encoding
    @n: number of labels available
    Output: assume m labels input. return a m by n tensor.
    """
    size = len(labels)
    labels = torch.LongTensor(labels).view(1, -1)  # labels have to be 2D so need the view() function
    a = torch.zeros(1, size).long()
    i = torch.cat((a, labels))  # indices is a 2D vector..
    v = torch.ones(size)
    out = torch.sparse.FloatTensor(i, v, torch.Size([1,n])).to_dense()
    return out


def precision(out, labels):
    """
    A function to calcualte top1 and top3 precision of predictions.
    @out: the output of the final layer of the network. n by 17 tensor, n is the batch size
    @labels: the original labels. n by 17 tensor where n is the batch size
    """

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    true = labels.nonzero().numpy().tolist()  # all the nonzero values. list of pairs, 0th is the row id and 1th is col id

    rows = np.arange(250).tolist()  # list of row ids

    top1_ids = torch.topk(out, 1, dim=1)[1].squeeze().numpy().tolist()
    top1_pred = [[a, b] for (a, b) in zip(rows, top1_ids)]  # top1 predictions
    top1_correct = intersection(true, top1_pred)
    top1_precision = len(top1_correct) / len(top1_pred)

    top3_ids = torch.topk(out, 3, dim=1)[1].squeeze().numpy().tolist()

    top3_pred = []
    count = 0
    for x in top3_ids:
        r = [count] * 3  # row id
        pairs = [[a, b] for (a, b) in zip(r, x)]  # make pairs
        top3_pred.extend(pairs)  # add to results
        count += 1

    top3_correct = intersection(true, top3_pred)
    top3_precision = len(top3_correct) / len(top3_pred)

    return top1_precision, top3_precision


############################################################################################
#                                   Training setup
############################################################################################


def main(argv):


    # parse arguments
    parser = argparse.ArgumentParser(add_help=False, description=('Begin running your computer vision experiments.'))
    parser.add_argument('--help', '-h', action='help', default=argparse.SUPPRESS, help='Show this help message and exit')
    parser.add_argument('--device', '-d', help='Speicfy the type of device to use')
    parser.add_argument('--path', '-p', help='Path of data folder')
    parser.add_argument('--workers', '-w', type=int, help='Number of dataloader workers')
    parser.add_argument('--optimizer', '-o', help='Type of optimizer to use')


    try:
        args = parser.parse_args(argv)

        device = args.device
        path = args.path
        num_workers = args.workers
        optimizer = args.optimizer

        if not path:
            parser.print_usage()
            raise ValueError('you need to specify an input path for pictures')

        if not device:
            parser.print_usage()
            raise ValueError('you need to specify a device to use for training')

        if not num_workers:
            parser.print_usage()
            raise ValueError('how many workers do you want to laod data for you?')

        if not optimizer:
            parser.print_usage()
            raise ValueError('you need to speicify the type of optimizer you want to use')


        # hardware hyperparameters
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # model hyperparameters
        num_classes = 17

        # training hyperparameters
        num_epochs = 1

        csv_file_path = os.path.join(path, 'train.csv')
        root_dir_path = os.path.join(path, 'train-jpg')

        # load data
        complete_dataset = PicturesDataset(csv_file=csv_file_path,
                                           root_dir=root_dir_path,
                                           transform=True,
                                           num_classes=num_classes)

        train_loader = torch.utils.data.DataLoader(dataset=complete_dataset,
                                                   batch_size=250,
                                                   shuffle=True,
                                                   num_workers=num_workers)

        model = ConvNet(num_classes)
        learning_rate = 0.01
        criterion = torch.nn.BCELoss()

        if(optimizer=='sgd'):
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


        # Train the model
        total_step = len(train_loader)

        for epoch in range(num_epochs):
            epoch_time = 0
            waiting_time = 0
            compute_time = 0

            start = time.monotonic()
            for i, (images, labels) in enumerate(train_loader):
                waiting_time += time.monotonic() - start  # aggregate on waiting time
                flag = time.monotonic()  # reset flag

                images = images.to(device)
                images = images[:, :3, :, :]  # drop the 4th channel
                labels = labels.squeeze()
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                precision1, precision3 = precision(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                compute_time += time.monotonic() - flag  # aggregate compute time
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Precision@1: {:.4f}, Precision@3: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), precision1, precision3))

            print('\nDone!')
            epoch_time += time.monotonic() - start
            # round time
            epoch_time = round((epoch_time) * 10 ** (-9), 10)
            waiting_time = round((waiting_time) * 10 ** (-9), 10)
            compute_time = round((compute_time) * 10 ** (-9), 10)
            print("Waiting time : " + str(waiting_time) + " secs")
            print("Compute time : " + str(compute_time) + " secs")
            print("Epoch time : " + str(epoch_time) + " secs")

    except Exception as e:
        print('Error:', str(e))
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])