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

