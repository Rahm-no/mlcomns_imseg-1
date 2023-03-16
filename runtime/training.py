from tqdm import tqdm
import os
from time import perf_counter_ns
import time
from numpy import random

import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler

from runtime.distributed_utils import get_rank, reduce_tensor, get_world_size
from runtime.inference import evaluate
from runtime.logging import mllog_event, mllog_start, mllog_end, CONSTANTS, all_workers_print


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


def busy_wait(dt):   
    current_time = time.time()
    while (time.time() < current_time+dt):
        pass

def emulate_compute(device, sec):
    if (str(device).find("GPU")!=-1):
        # print("Putting GPU into sleep for %10.5f sec"%sec)
        cuda.nanosleep(sec*1_000_000_000)
    else:
        time.sleep(sec)

UNET_MEASURED_SLEEP_TIMES = {
    "1": {
        "1": {
            "mean": 0.5482973824075986,
            "std": 0.006451229114864745,
            "median": 0.54807,
            "q1": 0.54686,
            "q3": 0.549425
        },
        "2": {
            "mean": 0.8371057694225177,
            "std": 0.009275869060140768,
            "median": 0.837069,
            "q1": 0.835032,
            "q3": 0.838983
        },
        "3": {
            "mean": 1.1072114821316614,
            "std": 0.004123831664074876,
            "median": 1.106964,
            "q1": 1.104516,
            "q3": 1.109849
        },
        "4": {
            "mean": 1.3650318132884778,
            "std": 0.009053563385589896,
            "median": 1.364309,
            "q1": 1.3614625,
            "q3": 1.3681815
        },
        "5": {
            "mean": 1.6445071627155172,
            "std": 0.004611577140197176,
            "median": 1.644256,
            "q1": 1.64116375,
            "q3": 1.64746425
        }
    },
    "2": {
        "1": {
            "mean": 0.5543537014998771,
            "std": 0.007084158775550645,
            "median": 0.554058,
            "q1": 0.552858,
            "q3": 0.5554
        },
        "2": {
            "mean": 0.8411210119462419,
            "std": 0.01599091791006479,
            "median": 0.840282,
            "q1": 0.838247,
            "q3": 0.8425229999999999
        },
        "3": {
            "mean": 1.1137939947089948,
            "std": 0.020116374715517974,
            "median": 1.112456,
            "q1": 1.109795,
            "q3": 1.115467
        },
        "4": {
            "mean": 1.3699684285714286,
            "std": 0.02219152290008795,
            "median": 1.3688544999999999,
            "q1": 1.36537775,
            "q3": 1.37272775
        },
        "5": {
            "mean": 1.6521917537414965,
            "std": 0.05265444278643964,
            "median": 1.649398,
            "q1": 1.645883,
            "q3": 1.652338
        }
    },
    "4": {
        "1": {
            "mean": 0.5541295460428074,
            "std": 0.0061521714181549,
            "median": 0.553736,
            "q1": 0.552469,
            "q3": 0.5550305
        },
        "2": {
            "mean": 0.8417280255102041,
            "std": 0.021750728267810833,
            "median": 0.8394189999999999,
            "q1": 0.837021,
            "q3": 0.842297
        },
        "3": {
            "mean": 1.1156181020408162,
            "std": 0.03607033718812949,
            "median": 1.112112,
            "q1": 1.108768,
            "q3": 1.1162174999999999
        },
        "4": {
            "mean": 1.3734651247165532,
            "std": 0.04623434494650203,
            "median": 1.368067,
            "q1": 1.3641955000000001,
            "q3": 1.372322
        },
        "5": {
            "mean": 1.6539953148688047,
            "std": 0.03880151520478443,
            "median": 1.648738,
            "q1": 1.645458,
            "q3": 1.652732
        }
    },
    "6": {
        "1": {
            "mean": 0.5566394678760394,
            "std": 0.011434649156769746,
            "median": 0.555719,
            "q1": 0.554198,
            "q3": 0.557348
        },
        "2": {
            "mean": 0.8476239293563579,
            "std": 0.026587468344471465,
            "median": 0.844392,
            "q1": 0.8411795,
            "q3": 0.8483335000000001
        },
        "3": {
            "mean": 1.1259871096938776,
            "std": 0.0700788100287033,
            "median": 1.1139255000000001,
            "q1": 1.1089885000000002,
            "q3": 1.12085175
        },
        "4": {
            "mean": 1.3956025272108843,
            "std": 0.09999913881815674,
            "median": 1.3705435000000001,
            "q1": 1.365761,
            "q3": 1.3832185000000001
        },
        "5": {
            "mean": 1.6887698214285713,
            "std": 0.12499219048914872,
            "median": 1.6512924999999998,
            "q1": 1.6468927500000001,
            "q3": 1.65907275
        }
    },
    "8": {
        "1": {
            "mean": 0.5593211989795919,
            "std": 0.035050401411968767,
            "median": 0.554577,
            "q1": 0.55282725,
            "q3": 0.55666725
        },
        "2": {
            "mean": 0.8545178548752834,
            "std": 0.059188740640250925,
            "median": 0.844172,
            "q1": 0.83985,
            "q3": 0.8504674999999999
        },
        "3": {
            "mean": 1.1270917074829931,
            "std": 0.0731983365453498,
            "median": 1.112901,
            "q1": 1.1073217499999999,
            "q3": 1.12251325
        },
        "4": {
            "mean": 1.4088261530612245,
            "std": 0.14763950262456094,
            "median": 1.3733434999999998,
            "q1": 1.3673712499999997,
            "q3": 1.383303
        },
        "5": {
            "mean": 1.6942649727891157,
            "std": 0.1452261457771531,
            "median": 1.650639,
            "q1": 1.644439,
            "q3": 1.662866
        }
    }
}


def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks, is_distributed, skip_step_7=False):
    rank = get_rank()

    # filename=os.path.join("/results", f'cases_read_{rank}.log')
    # logfile = open(filename, "w")
    # mllog_start(f"Rank {rank} opened {logfile} for writing\n")
    
    batch_size = flags.batch_size
    world_size = get_world_size()
    torch.backends.cudnn.benchmark = flags.cudnn_benchmark
    torch.backends.cudnn.deterministic = flags.cudnn_deterministic

    compute_time_mean = UNET_MEASURED_SLEEP_TIMES[str(world_size)][str(batch_size)]['mean']
    compute_time_std = UNET_MEASURED_SLEEP_TIMES[str(world_size)][str(batch_size)]['std']

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

    eval_num = 0

    cases_seen = set()

    for epoch in range(1, flags.epochs + 1):
        # logfile.write(f"Starting epoch {epoch}\n")
        cumulative_loss = []

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

            t_compute = perf_counter_ns()

            time.sleep(random.normal(compute_time_mean, compute_time_std))

            # image, label = image.to(device), label.to(device)
            # # mllog_end(key="load_batch_gpu", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata = {CONSTANTS.EPOCH_NUM: epoch})

            # # t0 = perf_counter_ns()
            # for callback in callbacks:
            #     callback.on_batch_start()

            # with autocast(enabled=flags.amp):
            #     output = model(image)
            #     # mllog_end(key="model_forward_pass", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})

            #     # t0 = perf_counter_ns()
            #     loss_value = loss_fn(output, label)
            #     loss_value /= flags.ga_steps
            #     # mllog_end(key="loss_tensor_calc", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            
            # # https://pytorch.org/docs/stable/notes/ddp.html#internal-design
            # # When gradients in one bucket are all ready, the Reducer kicks off an asynchronous allreduce on that bucket to 
            # # calculate mean of gradients across all processes. When all buckets are ready, the Reducer will block waiting 
            # # for all allreduce operations to finish. When this is done, averaged gradients are written to the param.grad field of all parameters. 
            # # After the backward pass, the grad field on the same corresponding parameter across different DDP processes should be the same.
            # #
            # # --> Parameter syncing happens throughout the backward() operation, and at the granularity of gradient buckets
            # # t0 = perf_counter_ns()
            # if flags.amp:
            #     scaler.scale(loss_value).backward()
            # else:
            #     loss_value.backward()
            # # mllog_end(key="model_backward_pass", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            
            # # From the optimizer’s perspective, it is optimizing a local model. Model replicas on all DDP processes can keep in sync because 
            # # they all start from the same state and they have the same averaged gradients in every iteration.
            # # t0 = perf_counter_ns()
            # if (iteration + 1) % flags.ga_steps == 0:
            #     if flags.amp:
            #         scaler.step(optimizer)
            #         scaler.update()
            #     else:
            #         optimizer.step()
            #     optimizer.zero_grad()
            # # mllog_end(key="model_optim_step", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            
            # # t0 = perf_counter_ns()
            # if not skip_step_7:
            #     # Calls an explicit all_reduce on the batch's loss_tensor
            #     # detach returns a cpy of the tensor, detached from graph
            #     # cpu moves it from GPU to CPU
            #     loss_value = reduce_tensor(loss_value, world_size).detach().cpu().numpy()
            #     cumulative_loss.append(loss_value)
            
            # mllog_end(key="cum_loss_fn_calc", value={"start": t0, "duration": perf_counter_ns() - t0}, metadata={CONSTANTS.EPOCH_NUM: epoch})

            mllog_end(key="all_compute", value={"start": t_iter, "duration": perf_counter_ns() - t_compute}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            mllog_end(key="step_end", value={"start": t_iter, "duration": perf_counter_ns() - t_iter}, metadata={CONSTANTS.EPOCH_NUM: epoch})
            # Restart counters for next iteration
            t_iter = t0 = perf_counter_ns()


        # logfile.flush()

        mllog_end(key=CONSTANTS.EPOCH_STOP, sync=False,
                  metadata={CONSTANTS.EPOCH_NUM: epoch, 'current_lr': optimizer.param_groups[0]['lr']})

        # if flags.lr_decay_epochs:
        #     scheduler.step()

        # if epoch == next_eval_at:
            # mllog_start(key=CONSTANTS.EVAL_START, value=epoch, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

            # # logfile.write(f"Starting eval {epoch}\n")
            # eval_num += 1

            # next_eval_at += flags.evaluate_every

            # del output

            # # eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, epoch, logfile)
            # eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, epoch)

            # if not skip_step_7:
            #     eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)

            # mllog_event(key=CONSTANTS.EVAL_ACCURACY, 
            #             value=eval_metrics["mean_dice"], 
            #             metadata={CONSTANTS.EPOCH_NUM: epoch}, 
            #             sync=False)
            # mllog_end(key=CONSTANTS.EVAL_STOP, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

            # for callback in callbacks:
            #     callback.on_epoch_end(epoch=epoch, metrics=eval_metrics, model=model, optimizer=optimizer)
            # model.train()
            # if eval_metrics["mean_dice"] >= flags.quality_threshold:
            #     is_successful = True
            
            # Commented out for experiments
            # elif eval_metrics["mean_dice"] < 1e-6:
            #     print("MODEL DIVERGED. ABORTING.")
            #     diverged = True

        mllog_end(key=CONSTANTS.BLOCK_STOP, sync=False,
                  metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1})

        # if is_successful or diverged:
        #     break

    # logfile.write(f"Training done\n")
    # logfile.flush()
    # logfile.close()

    mllog_end(key=CONSTANTS.RUN_STOP, sync=True,
              metadata={CONSTANTS.STATUS: CONSTANTS.SUCCESS if is_successful else CONSTANTS.ABORTED})

    for callback in callbacks:
        callback.on_fit_end()
