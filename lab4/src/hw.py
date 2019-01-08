import os
import torch
import torch.distributed as dist

#initialize MPI
dist.init_process_group(backend='mpi')
rank = dist.get_rank()
wsize = dist.get_world_size()

# print hello world
print("Hello from rank {} of world {}!".format(rank, wsize))

# blocking p2p communication
def run_blk(rank, size):
    dat = torch.zeros(1)
    if (rank == 0):
        dat += 1
        # send the tensor to process 1 
        dist.send(tensor=dat, dst=1)
    else:
        dist.recv(tensor=dat, src=0)
    print("rank ", rank, "has data ", dat[0])
    return 


# run_blk(rank, wsize)
    

def run_non_blk(rank, size):
    """
    Non blocking MPI p2p send + receive
    """
    dat = torch.zeros(1)
    req = None
    if (rank == 0):
        dat += 1
        # send the tensor to processor 1
        req = dist.isend(tensor=dat, dst=1)
        print("Rank 0 starts sending...\n")
    else:
        # receiver tensor from processor 0
        req = dist.irecv(tensor=dat, src=0)
        print("Rank 1 starts receiving...\n")
    req.wait()
    print("rank ", rank, "has data ", dat[0])
    return 


def run_all_reduce(rank, size):
    """
    all reduce fashion send
    """
    group = dist.new_group([0, 1])
    dat = torch.ones(1)
    dist.all_reduce(dat, op=dist.ReduceOp.SUM, group=group)
    print("rank ", rank, "has data ", dat[0])
    return 


run_all_reduce(rank, wsize)

