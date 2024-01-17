import asyncio
import os
import torch
from  imagespacing import image_spacing_ext
from math import ceil
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
import logging
from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore
import multiprocessing 
import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import concurrent.futures
import pdb
from data_loading.data_loader import get_data_split
from data_loading.pytorch_loader import get_train_transforms,PytTrain, PytVal
from runtime.training import train
import sys
import threading
import queue
from runtime.inference import evaluate
from runtime.arguments import PARSER
from runtime.distributed_utils import init_distributed, get_world_size, get_device, is_main_process, get_rank
from runtime.distributed_utils import seed_everything, setup_seeds
from runtime.logging import get_dllogger, mllog_start, mllog_end, mllog_event, mlperf_submission_log, mlperf_run_param_log
from runtime.callbacks import get_callbacks
import concurrent.futures
DATASET_SIZE = 168
import torch
import torch.multiprocessing as mp 
import subprocess
import time
#def create_PytTrain_instance(x_train, y_train, **train_data_kwargs):
 #   return PytTrain(x_train, y_train, **train_data_kwargs)



def main():
        
        
        
   
        mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet3d.log'))
        mllog.config(filename=os.path.join("/results", 'unet3d.log'))
        mllogger = mllog.get_mllogger()
        mllogger.logger.propagate = False
        mllog_start(key=constants.INIT_START)
      
        flags = PARSER.parse_args()
        image_spacing_instance = image_spacing_ext(flags.raw_dir)
        image_spacing_dic = image_spacing_instance()  # Call the instance to extract image spacings
        flags.image_spacings = image_spacing_dic
        flags.disable_logging=True
        dllogger = get_dllogger(flags)
        local_rank = flags.local_rank
        device = get_device(local_rank)
        is_distributed = init_distributed()
        world_size = get_world_size()
        local_rank = get_rank()

        
    
        mllog_event(key='world_size', value=world_size, sync=False)
        mllog_event(key='local_rank', value=local_rank, sync=True)
        mllog_event(key='Batch size used', value=flags.batch_size, sync=True)
        mllog_event(key='Epochs', value = flags.epochs, sync=True)


        worker_seeds, shuffling_seeds = setup_seeds(flags.seed, flags.epochs, device)
        worker_seed = worker_seeds[local_rank]
        seed_everything(worker_seed)
        mllog_event(key=constants.SEED, value=flags.seed if flags.seed != -1 else worker_seed, sync=False)

        if is_main_process:
            mlperf_submission_log()
            mlperf_run_param_log(flags)


        callbacks = get_callbacks(flags, dllogger, local_rank, world_size)
        flags.seed = worker_seed
        model = Unet3D(1, 3, normalization=flags.normalization, activation=flags.activation)

        mllog_start(key=constants.RUN_START, sync=True)
        mllog_event(key='loader', value=flags.loader, sync=True)
        mllog_end(key=constants.INIT_STOP, sync=True)

      


        
            
        mllog_event(key='Num of workers', value=flags.num_workers, sync = False)
        
           

        x_train, x_val, y_train, y_val = get_data_split(flags.data_dir, world_size, shard_id=local_rank)
            


                
        train_data_kwargs = {"patch_size": flags.input_shape, "oversampling": flags.oversampling, "seed": flags.seed , "raw-dir": flags.raw_dir, "image_spacings":flags.image_spacings}
  


        

        train_dataset = PytTrain(x_train, y_train, **train_data_kwargs)        



                # Similarly, log the validation data loading
       
        val_dataset = PytVal(x_val, y_val,flags.image_spacings)
            
        train_sampler = DistributedSampler(x_train, seed=flags.seed, drop_last=True) if world_size > 1 else None
        val_sampler = None
        

        
        train_dataloader = DataLoader(train_dataset,
                                            batch_size=flags.batch_size,
                                            shuffle=not flags.benchmark and train_sampler is None,
                                            sampler=train_sampler,
                                            num_workers=flags.num_workers,
                                            pin_memory=True,
                                            drop_last=True
                                            )

        val_dataloader = DataLoader(val_dataset,
                                            batch_size=1,
                                            shuffle= False,
                                            sampler=val_sampler,
                                            num_workers=1,
                                            pin_memory=True,
                                            drop_last=False
                                            )


        
            
            
        
        mllog_event(key='len train_dataloader', value=len(train_dataloader), sync=False)
        mllog_event(key='len val_dataloader', value=len(val_dataloader), sync=False)
        

        samples_per_epoch = world_size * len(train_dataloader) * flags.batch_size
        mllog_event(key='samples_per_epoch', value=samples_per_epoch, sync=False)
        flags.evaluate_every = flags.evaluate_every or ceil(20*DATASET_SIZE/samples_per_epoch)
        flags.start_eval_at = flags.start_eval_at or ceil(1000*DATASET_SIZE/samples_per_epoch)

        mllog_event(key=constants.GLOBAL_BATCH_SIZE, value=flags.batch_size * world_size * flags.ga_steps, sync=False)
        mllog_event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=flags.ga_steps)
        loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout,
                            include_background=flags.include_background)
        score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout=flags.layout,
                            include_background=flags.include_background)
        print('loss_fn in main is', loss_fn)
        loss_fn.to(device=device)
        
        if flags.exec_mode == 'train': 
           

            mllog_event(key="Here is where the training function is called")
            #train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn, device = device, callbacks = callbacks, is_distributed= is_distributed)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn, device, callbacks, is_distributed))                 

          

        

            
            

    
        
            

        elif flags.exec_mode == 'evaluate':
            mllog_event(key="Here is where the eval function is called")
        
            eval_metrics = evaluate(flags, model, val_dataloader, loss_fn, score_fn,
                                    device=device, is_distributed=is_distributed)
            if local_rank == 0:
                for key in eval_metrics.keys():
                    print(key, eval_metrics[key])
        else:
            print("Invalid exec_mode.")
            pass
        
   #     data_preprocessing_process.join()





if __name__ == "__main__":
    mp.set_start_method('spawn')
    
 
    main()
    
