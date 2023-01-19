from tqdm import tqdm
import os
from time import perf_counter_ns

import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler

from runtime.distributed_utils import get_rank, reduce_tensor, get_world_size
from runtime.inference import evaluate
from runtime.logging import mllog_event, mllog_start, mllog_end, CONSTANTS


def get_optimizer(params, flags):
    if flags.optimizer == "adam":
        optim = Adam(params, lr=flags.learning_rate, weight_decay=flags.weight_decay)
    elif flags.optimizer == "sgd":
        optim = SGD(params, lr=flags.learning_rate, momentum=flags.momentum, nesterov=True,
                    weight_decay=flags.weight_decay)
    elif flags.optimizer == "lamb":
        import apex
        optim = apex.optimizers.FusedLAMB(params, lr=flags.learning_rate, betas=flags.lamb_betas, 
                                          weight_decay=flags.weight_decay)
    else:
        raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
    return optim


def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = init_lr + (lr - init_lr) * scale


def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks, is_distributed):
    rank = get_rank()

    world_size = get_world_size()
    torch.backends.cudnn.benchmark = flags.cudnn_benchmark
    torch.backends.cudnn.deterministic = flags.cudnn_deterministic

    optimizer = get_optimizer(model.parameters(), flags)
    if flags.lr_decay_epochs:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=flags.lr_decay_epochs,
                                                         gamma=flags.lr_decay_factor)
    scaler = GradScaler()
    # Model an loss function are on GPU
    model.to(device)
    loss_fn.to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank],
                                                          output_device=rank)

    is_successful = False
    diverged = False
    next_eval_at = flags.start_eval_at
    model.train()
    for callback in callbacks:
        callback.on_fit_start()
    for epoch in range(1, flags.epochs + 1):
        # cumulative_loss = []

        if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
            lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
        mllog_start(key=CONSTANTS.BLOCK_START, sync=False,
                    metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1})
        mllog_start(key=CONSTANTS.EPOCH_START, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)
        # Necessary for DDP
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        loss_value = None
        optimizer.zero_grad()
        
        # start timers
        t_iter = t0 = perf_counter_ns()
        for iteration, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
            image, label = batch
            mllog_end(key="load_batch_mem", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata = {CONSTANTS.EPOCH_NUM: epoch})

            continue
            
            t0 = perf_counter_ns()
            image, label = image.to(device), label.to(device)
            mllog_end(key="load_batch_gpu", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata = {CONSTANTS.EPOCH_NUM: epoch})

            for callback in callbacks:
                callback.on_batch_start()

            t0 = perf_counter_ns()
            with autocast(enabled=flags.amp):
                output = model(image)
                mllog_end(key="model_forward_pass", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})

                t0 = perf_counter_ns()
                loss_value = loss_fn(output, label)
                loss_value /= flags.ga_steps
                mllog_end(key="loss_tensor_calc", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            
            # https://pytorch.org/docs/stable/notes/ddp.html#internal-design
            # When gradients in one bucket are all ready, the Reducer kicks off an asynchronous allreduce on that bucket to 
            # calculate mean of gradients across all processes. When all buckets are ready, the Reducer will block waiting 
            # for all allreduce operations to finish. When this is done, averaged gradients are written to the param.grad field of all parameters. 
            # After the backward pass, the grad field on the same corresponding parameter across different DDP processes should be the same.
            #
            # --> Parameter syncing happens throughout the backward() operation, and at the granularity of gradient buckets
            t0 = perf_counter_ns()
            if flags.amp:
                scaler.scale(loss_value).backward()
            else:
                loss_value.backward()
            mllog_end(key="model_backward_pass", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            
            # From the optimizer’s perspective, it is optimizing a local model. Model replicas on all DDP processes can keep in sync because 
            # they all start from the same state and they have the same averaged gradients in every iteration.
            t0 = perf_counter_ns()
            if (iteration + 1) % flags.ga_steps == 0:
                if flags.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            mllog_end(key="model_optim_step", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            
            # REMOVING THIS LAST STEP
            # t0 = perf_counter_ns()
            # # Calls an explicit all_reduce on the batch's loss_tensor
            # # detach returns a cpy of the tensor, detached from graph
            # # cpu moves it from GPU to CPU
            # loss_value = reduce_tensor(loss_value, world_size).detach().cpu().numpy()
            # cumulative_loss.append(loss_value)
            # mllog_end(key="cum_loss_fn_calc", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})

            mllog_end(key="step_end", value={"start": t_iter, "duration": perf_counter_ns() - t_iter}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            # Restart counters for next iteration
            t_iter = t0 = perf_counter_ns()


        mllog_end(key=CONSTANTS.EPOCH_STOP, sync=False,
                  metadata={CONSTANTS.EPOCH_NUM: epoch, 'current_lr': optimizer.param_groups[0]['lr']})

        if flags.lr_decay_epochs:
            scheduler.step()

        if epoch == next_eval_at:
            next_eval_at += flags.evaluate_every
            del output
            mllog_start(key=CONSTANTS.EVAL_START, value=epoch, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

            continue

            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, epoch)
            # eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)

            mllog_event(key=CONSTANTS.EVAL_ACCURACY, 
                        value=eval_metrics["mean_dice"], 
                        metadata={CONSTANTS.EPOCH_NUM: epoch}, 
                        sync=False)
            mllog_end(key=CONSTANTS.EVAL_STOP, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

            for callback in callbacks:
                callback.on_epoch_end(epoch=epoch, metrics=eval_metrics, model=model, optimizer=optimizer)
            model.train()
            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                is_successful = True
            
            # Commented out for experiments
            # elif eval_metrics["mean_dice"] < 1e-6:
            #     print("MODEL DIVERGED. ABORTING.")
            #     diverged = True

        mllog_end(key=CONSTANTS.BLOCK_STOP, sync=False,
                  metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1})

        if is_successful or diverged:
            break

    mllog_end(key=CONSTANTS.RUN_STOP, sync=True,
              metadata={CONSTANTS.STATUS: CONSTANTS.SUCCESS if is_successful else CONSTANTS.ABORTED})

    for callback in callbacks:
        callback.on_fit_end()
