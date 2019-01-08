import pandas as pd
import os
from PIL import Image
import time
import csv
import random
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

from math import ceil
from random import Random


class Partition(object):
    
    def __init__(self, data, index):
        self.data = data
        self.index = index
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
    
    
class DataPartition(object):
    
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        
        rng = Random()  # generate random seed
        rng.seed(seed)
        # shuffle data by reorganizing all indexes 
        all_indexes = [x for x in range(0, len(data))]
        rng.shuffle(all_indexes)   # get a random list of indexes 
        
        # store partitions into a new list of list. 
        # every sublist is indexes for minibatch assigned to a rank. 
        # the sublist length is according to sizes 
        self.list_of_ids = []  
        curr_start_id = 0
        for i in range(len(sizes)):
            curr_batch_size = ceil(len(data) * sizes[i])
            curr_mini_batch = all_indexes[curr_start_id: curr_start_id + curr_batch_size]
            self.list_of_ids.append(curr_mini_batch)
            curr_start_id += curr_batch_size  # change the start id for the next batch
        return
    
     
    def get(self, rank_id):
        """
        This function takes in a rank id and returns its share of minibatch
        """
        return Partition(self.data, self.list_of_ids[rank_id])
                          
         
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


def get_partition_dataloader(dataset, batch_size, num_workers, myrank):
    """
    Partition the whole dataset into smaller sets for each rank.
    
    @param dataset: complete the dataset. We will 1) split it evenly to every worker and 2) build dataloaders on splitted dataset
    @batch_size: batch_size for the data loader! i.e. every worker will load this many samples 
    @param num_workers: the number of workers. Used to decide partition ratios
    @param myrank: the rank of this particular process
    rvalue: training set for this particular rank
    """
    partition_ratio = [1.0/num_workers for _ in range(num_workers)]
    partitioner = DataPartition(dataset, partition_ratio)  # partitioner is in charge of producing shuffled id lists
    curr_rank_dataset = partitioner.get(myrank-1)  # get the data partitioned for current rank, 0 is the server so -1
    # build a dataloader based on the partitioned dataset for current rank
    train_set = torch.utils.data.DataLoader(curr_rank_dataset, batch_size=batch_size, shuffle=True)
    return train_set
                             
    
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



if __name__ == '__main__':
    
    # initialize MPI environment 
    dist.init_process_group(backend="mpi")
    myrank = dist.get_rank()
    wsize = dist.get_world_size()  # number of processes = num_workers + 1   
    server_rank = 0 # this process is the server
    
    # print hello world to test MPI
    if (myrank == server_rank):
        print("Hello from {}. I am the server.\n".format(myrank))
    else:
        print("Hello from {} of world size {}".format(myrank, wsize)) 
    
    #############################################################################################################################
    #                                          setup code shared by workers and server
    #############################################################################################################################

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--use_cuda', type=str, default='true', help='Use CUDA if available')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--data_path', type=str, default='/home/yl5090/hpc/lab4/lab4/kaggleamazon/',
                        help='Data path')
    parser.add_argument('--opt', type=str, default='adam',
                        help='NN optimizer (Examples: adam, rmsprop, adadelta, ...)')

    args = parser.parse_args()
    
    # Timing variables.
    AggLoadBatchTime = 0.0;
    AggIOTime = 0.0;
    PreprocessingTime = 0.0;
    
    torch.manual_seed(123)
    model = Net() # workers and servers all have a model! 
    
    DATA_PATH=args.data_path
    IMG_PATH = DATA_PATH+'train-jpg/'
    IMG_EXT = '.jpg'
    TRAIN_DATA = DATA_PATH+'train.csv'
    transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    dset_train = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)
    
    num_workers = args.workers
    batch_size=250
    num_epochs = 5
       
    #############################################################################################################################
    #                                                 workers' setup code
    ############################################################################################################################# 
    if (myrank != server_rank):
        train_loader = get_partition_dataloader(dset_train, batch_size, num_workers, myrank)
        print('Dataset Loaded and Partitioned, total samples for worker rank {}: '.format(myrank), len(train_loader.dataset))
        # initialize criterion
        criterion = nn.BCELoss()  #.to(device=device)
        worker_loss_hist = []  # used in the end
        num_samples_seen = 0   # used in the end 
    
    #############################################################################################################################
    #                                                 server's setup code
    #############################################################################################################################
    elif (myrank == server_rank):
        # initialize parameters
        for param in model.parameters():
            param.grad = torch.zeros(param.size(), requires_grad=True)
            param.grad.data.zero_()
            
        # initialize optimizers
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
            
    
    # training starts!!
    TotalStrtTime = time.monotonic();
    
    for epoch_id in range(num_epochs):
        EpochStrtTime = time.monotonic();
        if (myrank == server_rank):
            # server code 
            # server stops when this epoch finishes
            num_server_updates = int(num_workers * len(dset_train) / batch_size)   # max number of updates for the server 
            count_updates = 0
            loss_hist = torch.zeros([num_server_updates])  # stores loss history
            
            while(count_updates < num_server_updates):
                count_updates += 1
                # initialize optimizer
                optimizer.zero_grad()

                # receive handshake message + identify sender for the current round 
                hello_buffer = torch.zeros([1])
                # print("Server prepares to receive hello from a worker for this round")
                sender = dist.recv(tensor=hello_buffer, src=None, tag=99)  
                #print("Server finishes receiving hello from worker {0}".format(sender))
                #print("Server prepares to send hello to worker {0}".format(sender))
                dist.send(hello_buffer, dst=sender, tag=98)  # say hello back
               # print("Server finishes sending hello to worker {0}".format(sender))         

                # receive loss 
                loss_buffer = torch.zeros([1])
                # print("Server prepare to receive loss from worker {0}\n".format(sender))
                dist.recv(tensor=loss_buffer, src=sender)
                # print("Server finishes receiving loss from worker {0}\n".format(sender))
                loss_hist[count_updates-1] = loss_buffer  # stores this round loss 
                # print("Server finishes saving loss to its history\n")

                # receive gradients 
                for name, param in model.named_parameters():
                    grad_buffer = torch.zeros(param.grad.size())
                    #  print("Server prepares to receive {0} gradient from worker {1}".format(name, sender))
                    dist.recv(tensor=grad_buffer, src=sender)
                    # print("Server finishes receiving {0} gradient from worker {1}".format(name, sender))
                    param.grad.data = grad_buffer

                # update 
                optimizer.step()

                # send the information back to the worker who send in gradients
                for name, param in model.named_parameters():
                    param_tensor = param.data  # parameter buffer 
                    # print("Server prepares to send {0} parameter to worker {1}".format(name, sender))
                    dist.send(tensor=param_tensor, dst=sender)  # send back parameter information 
                    # print("Server finishes sending {0} parameter to worker {1}".format(name, sender))
                                
        else:
            # worker code 
            for batch_idx, (data, target) in enumerate(train_loader):  
          
                # establish connection with the server
                hello_buffer = torch.zeros([1])
                # print("Worker {0} prepares to say hello to the server".format(myrank))
                dist.send(tensor=hello_buffer, dst=server_rank, tag=99) # send handshake message
                # print("Worker {0} finishes saying hello to the server".format(myrank))
                # print("Worker {0} prepares to receive hello from the server".format(myrank))
                dist.recv(tensor=hello_buffer, src=server_rank, tag=98)
                # print("Worker {0} finishes receiving hello from the server".format(myrank))
                # print("Connection with worker {0} is established!".format(myrank))

                # compute the gradient first or else gradient will be None.
                # the caveat is the model will use the parameters fetched last time
                output = model(data)
                loss = criterion(output, target)
                # print("Current loss is {0}\n".format(loss))
                # print("Worker {0} prepare to send loss to the server\n".format(myrank))
                dist.send(tensor=loss, dst=server_rank)
                # print("Worker {0} finishes sending loss to the server\n".format(myrank))
                loss.backward()
                worker_loss_hist.append(loss.item)
                num_samples_seen += len(target)

                # Now begin sending gradients to the server 
                for name, param in model.named_parameters():
                    # print("Worker {0} prepare to send {1} gradient to the server\n".format(myrank, name))
                    # print("The gradient size is {}".format(param.grad.data.size()))
                    dist.send(tensor=param.grad.data, dst=server_rank) 
                    # print("Worker {0} finishes sending {1}  gradient to the server\n".format(myrank, name))

                # wait to receive updated parameters to use for the next round
                for name, param in model.named_parameters():
                    param_buffer = torch.zeros(param.size())
                    # print("Now worker {0} prepare to receive {1} parameter from the server\n".format(myrank, name))
                    # print("The parameter size is {}".format(param.data.size()))
                    dist.recv(tensor=param_buffer, src=server_rank)
                    param.data = param_buffer
                    # print("Now worker {0} finishes receiving {1} parameter from the server\n".format(myrank, name))
                    
        # now finishes one epoch. Synchronize the workers
        workers_list = list(range(1, num_workers+1))
        labor_union = dist.new_group(ranks=workers_list) 
        dist.barrier(labor_union)  # synchronize all workers 
        
        # the server send parameters to all workers 
        for name, param in model.named_parameters():
            param_buffer = torch.zeros(param.size())
            print("Finishes one epoch. The server broadcasts parameter to everyone.")
            dist.broadcast(param.data, server_rank)
        
        # workers print results
        exec_time = time.monotonic() - EpochStrtTime
        print("{0}, {1}, {2}".format(myrank, loss, exec_time))
        
    # now finishes all epoches, do an all_reduce between workers 
    dist.barrier(labor_union)  # synchronize all workers 
    # worker code 
    if (myrank != server_rank):
        my_avg_loss = sum(worker_loss_hist) / len(worker_loss_hist)
        my_loss_sum = torch.tensor(num_samples_seen * my_avg_loss)
        # all reduce phase 1
        dist.all_reduce(my_loss_sum, op=ReduceOp.SUM, group=label_union)
        # all reduce phase 2
        total_samples_seen = torch.tensor(num_samples_seen)
        dist.all_reduce(total_samples_seen, op=ReduceOp.SUM, group=label_union)
        # finally get the result
        final_weighted_loss = torch.div(my_loss_sum, total_samples_seen)
    
    TotalTime = (time.monotonic()-TotalStrtTime)
    print("{0}, {1}, {2}".format(myrank, final_weighted_loss, exec_time))

