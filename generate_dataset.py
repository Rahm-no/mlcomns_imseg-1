import os
import sys
import numpy as np
import random
from os.path import join 
from pathlib import Path

NUM_CASES = 210

DIM1_MEAN = 246
DIM1_STD = 87

DIM2_MEAN = 339
DIM2_STD = 46

def gen_dataset_random(datadir):

    for case in range(NUM_CASES):
        
        name_x = f'case_{case:05d}_x.npy'
        name_y = f'case_{case:05d}_y.npy'

        dim1 = int(random.gauss(DIM1_MEAN, DIM1_STD))
        dim2 = int(random.gauss(DIM2_MEAN, DIM2_STD))

        # dimensions cannot be smaller than 128
        if dim1 < 128:
            dim1 = 128

        if dim2 < 128:
            dim2 = 128

        case_x = np.random.randn(1, dim1, dim2, dim2).astype('float32')
        case_y = np.random.random_integers(low=0, high=1, size=(1, dim1, dim2, dim2)).astype('int8')

        np.save(join(datadir, name_x), case_x)
        np.save(join(datadir, name_y), case_y)


def generate_data_imseg(output_path, desired_size):
    # size range
    # [  1 471 444 444]
    # [  1 128 186 186]

    if not os.path.isdir(output_path):
        print(f'Creating {output_path}')
        Path(output_path).mkdir(parents=True)

    newcase_counter = 0
    total_size = 0
    for newcase_counter in range(197, NUM_CASES):
        size1 = random.randint(128, 471)
        size2 = random.randint(186, 444)
        img = np.random.uniform(low=-2.340702, high=2.639792, size=(1, size1, size2, size2))
        mask = np.random.randint(0, 2, size=(1, size1, size2, size2))
        img = img.astype(np.float32)
        mask = mask.astype(np.uint8)
        fnx = f"{output_path}/case_{newcase_counter:05}_x.npy"
        fny = f"{output_path}/case_{newcase_counter:05}_y.npy"
        print(f'Saved {output_path}/case_{newcase_counter:05}')
        np.save(fnx, img)
        np.save(fny, mask)
        # newcase_counter += 1
        total_size += os.path.getsize(fnx)
        total_size += os.path.getsize(fny)


if __name__=='__main__':
    # gen_dataset_random(sys.argv[1])
    generate_data_imseg(sys.argv[1], int(sys.argv[2]))