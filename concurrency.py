import torch
import threading
import queue
from runtime.training import train



# Define a job queue
job_queue = queue.Queue()

# Define a lock for synchronization
lock = threading.Lock()


# Define the training class
class Training:
    def __init__(self, job_id, train_dataloader):
        self.job_id = job_id
        self.train_dataloader = train_dataloader

    def train(self):

         train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn, 
                device=device, callbacks=callbacks, is_distributed=is_distributed)

# Function for CPU preprocessing
def preprocess_job(job):
    train_dataloader = job.train_dataloader

    # Add train_dataloader to training queue
    with lock:
        training_queue.put(Training(job.job_id, train_dataloader))

# Function for GPU training
def train_job(job):
    job.train()
