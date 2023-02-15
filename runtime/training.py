from tqdm import tqdm
import os
from time import perf_counter_ns
import time

import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler

from runtime.distributed_utils import get_rank, reduce_tensor, get_world_size
from runtime.inference import evaluate
from runtime.logging import mllog_event, mllog_start, mllog_end, CONSTANTS


# def get_optimizer(params, flags):
#     if flags.optimizer == "adam":
#         optim = Adam(params, lr=flags.learning_rate, weight_decay=flags.weight_decay)
#     elif flags.optimizer == "sgd":
#         optim = SGD(params, lr=flags.learning_rate, momentum=flags.momentum, nesterov=True,
#                     weight_decay=flags.weight_decay)
#     elif flags.optimizer == "lamb":
#         import apex
#         optim = apex.optimizers.FusedLAMB(params, lr=flags.learning_rate, betas=flags.lamb_betas, 
#                                           weight_decay=flags.weight_decay)
#     else:
#         raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
#     return optim


# def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
#     scale = current_epoch / warmup_epochs
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = init_lr + (lr - init_lr) * scale


# def busy_wait(dt):   
#     current_time = time.time()
#     while (time.time() < current_time+dt):
#         pass


def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks, is_distributed, skip_step_7=False):
    rank = get_rank()

    filename=os.path.join("/results", f'cases_read_{rank}.log')
    logfile = open(filename, "w")
    mllog_start(f"Rank {rank} opened {logfile} for writing\n")

    # world_size = get_world_size()
    # torch.backends.cudnn.benchmark = flags.cudnn_benchmark
    # torch.backends.cudnn.deterministic = flags.cudnn_deterministic

    # optimizer = get_optimizer(model.parameters(), flags)
    # if flags.lr_decay_epochs:
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                      milestones=flags.lr_decay_epochs,
    #                                                      gamma=flags.lr_decay_factor)
    # # Model an loss function are on GPU
    # model.to(device)
    # loss_fn.to(device)

    # if is_distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                       device_ids=[rank],
    #                                                       output_device=rank)

    # for callback in callbacks:
    #     callback.on_fit_start()


    for epoch in range(1, flags.epochs + 1):
        logfile.write(f"Epoch {epoch}\n")

        # Necessary for DDP
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        # optimizer.zero_grad()
        
        # start timers
        for iteration, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
            cases = batch

            for i in cases:
                logfile.write(f'{i}\n')

        logfile.flush()

    logfile.flush()
    logfile.close()


