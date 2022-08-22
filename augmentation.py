from starter_code.utils import load_case
import numpy as np
import cv2


def HorizontalFlip(input, type):
    return cv2.flip(input, 1)


def VerticalFlip(input, type):
    return cv2.flip(input, 1)


def GaussianBlurring(input, type):
    if type == "img":
        return cv2.GaussianBlur(input, ksize=(5, 5), sigmaX=1, sigmaY=1)
    return input


def Sharpening(input, type):
    if type == "imge":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(input, -1, kernel)
    return input


if __name__ == "__main__":
    # volume, segmentation = load_case(000)
    # images = volume.get_fdata()
    # masks = segmentation.get_fdata()

    # # # transformed = [Sharpening(img, mask) for img in ]

    # imgs = images
    # masks = masks
    # # print("max: ", img.max(), "min:", img.min())
    # imgs_new = np.array([Sharpening(img, "img") for img in imgs])
    # masks_new = np.array([Sharpening(mask, "mask") for mask in masks])
    # print(imgs_new.shape)
    # print(masks_new.shape)

    # np.save("data/case_00000_raw_x.npy", imgs_new)
    # np.save("data/case_00000_raw_y.npy", masks_new)
    case000_x = np.load('/data/kits19/preprocessed_data/case_00000_x.npy')
    print(case000_x.shape)
    # print("max: ", image_new.max(), "min:", image_new.min())



## dependency
## nibabel
## opencv