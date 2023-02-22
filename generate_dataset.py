import sys
import numpy as np
import random
from numpy.random import default_rng
from os.path import join 

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


if __name__=='__main__':
    gen_dataset_random(sys.argv[1])