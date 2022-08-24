import os
import argparse
import hashlib
import json
from tqdm import tqdm
import nibabel
import numpy as np
import torch
from torch.nn.functional import interpolate
from starter_code.utils import load_case

EXCLUDED_CASES = []  # [23, 68, 125, 133, 15, 37]
MAX_ID = 210
MEAN_VAL = 101.0
STDDEV_VAL = 76.9
MIN_CLIP_VAL = -79.0
MAX_CLIP_VAL = 304.0
TARGET_SPACING = [1.6, 1.2, 1.2]
TARGET_SHAPE = [128, 128, 128]


class Stats:
    def __init__(self):
        self.mean = []
        self.std = []
        self.d = []
        self.h = []
        self.w = []

    def append(self, mean, std, d, h, w):
        self.mean.append(mean)
        self.std.append(std)
        self.d.append(d)
        self.h.append(h)
        self.w.append(w)

    def get_string(self):
        self.mean = np.median(np.array(self.mean))
        self.std = np.median(np.array(self.std))
        self.d = np.median(np.array(self.d))
        self.h = np.median(np.array(self.h))
        self.w = np.median(np.array(self.w))
        return f"Mean value: {self.mean}, std: {self.std}, d: {self.d}, h: {self.h}, w: {self.w}"


class Preprocessor:


    def __init__(self, args):
        self.mean = MEAN_VAL
        self.std = STDDEV_VAL
        self.min_val = MIN_CLIP_VAL
        self.max_val = MAX_CLIP_VAL
        self.results_dir = args.results_dir
        self.data_dir = args.data_dir
        self.target_spacing = TARGET_SPACING
        self.stats = Stats()

    def preprocess_dataset(self):
        data = json.load(open("./image_spacings.json"))

        for i in range(2):
            case = str(i)
            while len(case) < 3:
                case = "0" + case
            image = np.load(os.path.join(self.data_dir, f"case_00{case}_raw_x.npy"))
            label = np.load(os.path.join(self.data_dir, f"case_00{case}_raw_y.npy"))
            image_spacings = data[case]

            image = image.reshape((1,) + image.shape)
            label = label.reshape((1,) + label.shape)
            image, label = self.preprocess_case(image, label, image_spacings)
            image, label = self.pad_to_min_shape(image, label)
            self.save(image, label, case)
            # np.save(f"data/case_00{case}_preprocessed_x.npy", image)
            # np.save(f"data/case_00{case}_preprocessed_y.npy", label)

    def preprocess_case(self, image, label, image_spacings):
        image, label = self.resample3d(image, label, image_spacings)
        image = self.normalize_intensity(image.copy())
        return image, label

    @staticmethod
    def pad_to_min_shape(image, label):
        current_shape = image.shape[1:]
        bounds = [max(0, TARGET_SHAPE[i] - current_shape[i]) for i in range(3)]
        paddings = [(0, 0)]
        paddings.extend([(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)])
        return np.pad(image, paddings, mode="edge"), np.pad(label, paddings, mode="edge")


    def resample3d(self, image, label, image_spacings):
        if image_spacings != self.target_spacing:
            spc_arr = np.array(image_spacings)
            targ_arr = np.array(self.target_spacing)
            shp_arr = np.array(image.shape[1:])
            new_shape = (spc_arr / targ_arr * shp_arr).astype(int).tolist()

            image = interpolate(torch.from_numpy(np.expand_dims(image, 0)),
                                size=new_shape, mode='trilinear', align_corners=True)
            label = interpolate(torch.from_numpy(np.expand_dims(label, 0)), size=new_shape, mode='nearest')
            image = np.squeeze(image.numpy(), 0)
            label = np.squeeze(label.numpy(), 0)
        print("resampeld shape:", image.shape)
        return image, label

    def normalize_intensity(self, image: np.array):
        image = np.clip(image, self.min_val, self.max_val)
        image = (image - self.mean) / self.std
        return image

    def save(self, image, label, case: str):
        image = image.astype(np.float32)
        label = label.astype(np.uint8)
        mean, std = np.round(np.mean(image, (1, 2, 3)), 2), np.round(np.std(image, (1, 2, 3)), 2)
        print(f"Saving {case} shape {image.shape} mean {mean} std {std}")
        self.stats.append(mean, std, image.shape[1], image.shape[2], image.shape[3])
        np.save(os.path.join(self.results_dir, f"case_00{case}_x.npy"), image, allow_pickle=False)
        np.save(os.path.join(self.results_dir, f"case_00{case}_y.npy"), label, allow_pickle=False)


def verify_dataset(results_dir):
    with open('checksum.json') as f:
        source = json.load(f)

    assert len(source) == len(os.listdir(results_dir))
    for volume in tqdm(os.listdir(results_dir)):
        with open(os.path.join(results_dir, volume), 'rb') as f:
            data = f.read()
            md5_hash = hashlib.md5(data).hexdigest()
            assert md5_hash == source[volume], f"Invalid hash for {volume}."
    print("Verification completed. All files' checksums are correct.")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data_dir', dest='data_dir', required=True)
    PARSER.add_argument('--results_dir', dest='results_dir', required=True)
    PARSER.add_argument('--mode', dest='mode', choices=["preprocess", "verify"], default="preprocess")
    
    args = PARSER.parse_args()
    if args.mode == "preprocess":
        preprocessor = Preprocessor(args)
        preprocessor.preprocess_dataset()
        # verify_dataset(args.results_dir)


    if args.mode == "verify":
        verify_dataset(args.results_dir)

