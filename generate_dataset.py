import sys
import numpy as np
import random
from numpy.random import default_rng
from os.path import join 
from pathlib import Path

NUM_CASES = 210

def gen_dataset_random(datadir):
    Path(datadir).mkdir(exist_ok=True, parents=True)

    for case in range(NUM_CASES):
        
        name_x = f'case_{case:05d}_x.npy'
        name_y = f'case_{case:05d}_y.npy'

        size1 = random.randint(128, 470)
        size2 = random.randint(186, 434)

        case_x = np.random.uniform(low=-2.340702, high=2.639792, size=(1, size1, size2, size2))
        case_y = np.random.randint(0, 2, size=(1, size1, size2, size2))

        case_x = case_x.astype(np.float32)
        case_y = case_y.astype(np.uint8)

        np.save(join(datadir, name_x), case_x)
        np.save(join(datadir, name_y), case_y)

        print(f'Generated case {case}')


if __name__=='__main__':
    gen_dataset_random(sys.argv[1])