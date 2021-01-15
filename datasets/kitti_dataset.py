import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms.functional as photometric
import pdb


class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames, self.dx_gt_filenames, self.dy_gt_filenames = \
            self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None, None, None
        else:
            disp_images = [x[2] for x in splits]
            dx_gt = [x[3] for x in splits]
            dy_gt = [x[4] for x in splits]
            return left_images, right_images, disp_images, dx_gt, dy_gt

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def load_dx_dy(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.dx_gt_filenames and self.dy_gt_filenames:  # has disparity slant param ground truth
            dx_gt = self.load_dx_dy(os.path.join(self.datapath, self.dx_gt_filenames[index]))
            dy_gt = self.load_dx_dy(os.path.join(self.datapath, self.dy_gt_filenames[index]))
        else:
            dx_gt = None
            dy_gt = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 1152, 320  # similar to crops of HITNet paper, but multiple of 64

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            dx_gt = dx_gt[y1:y1 + crop_h, x1:x1 + crop_w]
            dy_gt = dy_gt[y1:y1 + crop_h, x1:x1 + crop_w]

            # photometric augmentation: brightness and contrast perturb
            sym_random_brt = np.random.uniform(0.8, 1.2)
            sym_random_cts = np.random.uniform(0.8, 1.2)
            asym_random_brt = np.random.uniform(0.95, 1.05, size=2)
            asym_random_cts = np.random.uniform(0.95, 1.05, size=2)
            # brightness
            left_img = photometric.adjust_brightness(left_img, sym_random_brt)
            right_img = photometric.adjust_brightness(right_img, sym_random_brt)
            left_img = photometric.adjust_brightness(left_img, asym_random_brt[0])
            right_img = photometric.adjust_brightness(right_img, asym_random_brt[1])
            # contrast
            left_img = photometric.adjust_contrast(left_img, sym_random_cts)
            right_img = photometric.adjust_contrast(right_img, sym_random_cts)
            left_img = photometric.adjust_contrast(left_img, asym_random_cts[0])
            right_img = photometric.adjust_contrast(right_img, asym_random_cts[1])

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # random patch exchange of right image
            patch_h = random.randint(50, 180)
            patch_w = random.randint(50, 250)
            patch1_x = random.randint(0, crop_h-patch_h)
            patch1_y = random.randint(0, crop_w-patch_w)
            patch2_x = random.randint(0, crop_h-patch_h)
            patch2_y = random.randint(0, crop_w-patch_w)
            # pdb.set_trace()
            # print(right_img.shape)
            img_patch = right_img[:, patch2_x:patch2_x+patch_h, patch2_y:patch2_y+patch_w]
            right_img[:, patch1_x:patch1_x+patch_h, patch1_y:patch1_y+patch_w] = img_patch

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "dx_gt": dx_gt,
                    "dy_gt": dy_gt}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1280x384
            top_pad = 384 - h
            right_pad = 1280 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            # pad dx and dy gt
            if dx_gt is not None and dy_gt is not None:
                assert len(dx_gt.shape) == 2
                dx_gt = np.lib.pad(dx_gt, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                assert len(dy_gt.shape) == 2
                dy_gt = np.lib.pad(dy_gt, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None and dx_gt is not None and dy_gt is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "dx_gt": dx_gt,
                        "dy_gt": dy_gt}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
