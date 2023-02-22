import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib 

all_cases = os.listdir("/raid/data/imseg/29gb-gen")
all_cases.sort()

# Each have (1, dim1, dim2, dim2)

dim1 = []
dim2 = []

for file in all_cases:
    case = np.load(f'/raid/data/imseg/29gb-npy/{file}')

    dim1.append(case.shape[1])
    dim2.append(case.shape[2])

    print(f'{file}: {case.shape} {case.dtype} {case}')


dim1 = np.asarray(dim1)
dim2 = np.asarray(dim2)


print(f'DIM1 - mean {dim1.mean()} std {dim1.std()} min {dim1.min()} max {dim1.max()}')
print(f'DIM2 - mean {dim2.mean()} std {dim2.std()} min {dim2.min()} max {dim2.max()}')

exit()

N_BINS = 50

fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(15,15))
fig.suptitle("UNET3D Dataset Size Distribution")

ax1.hist(dim1, bins=N_BINS)
median1 = np.median(dim1)

trans = ax1.get_xaxis_transform()
ax1.axvline(median1, color='k', linestyle='dashed', linewidth=1)
plt.text(median1 * 1.5, .85, f'median: {int(median1):,}', transform=trans)

ax2.hist(dim2, bins=N_BINS)
median2 = np.median(dim2)
trans = ax2.get_xaxis_transform()
ax2.axvline(median2, color='k', linestyle='dashed', linewidth=1)
plt.text(median2 * 1.5, .85, f'median: {int(median2):,}', transform=trans)

filename = 'unet3d_dataset_distribution.png'
figure_filename = os.path.join("notes/", filename)

plt.savefig(figure_filename, format="png", dpi=250)
# Clear the current axes.
plt.cla() 
# Closes all the figure windows.
plt.close('all')   
plt.close(fig)
