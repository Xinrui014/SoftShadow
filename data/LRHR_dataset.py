from io import BytesIO
import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os
from torchvision import transforms
import yaml
import numpy as np
import torchvision

class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor

class Augment_BBox_torch:
    def __init__(self, image_size):
        self.image_size = image_size  # (height, width)

    def transform0(self, bbox):
        # No transformation
        return bbox

    def transform1(self, bbox):
        # Rotate 90 degrees
        x1, y1, x2, y2 = bbox
        return [self.image_size[1] - y2, x1, self.image_size[1] - y1, x2]

    def transform2(self, bbox):
        # Rotate 180 degrees
        x1, y1, x2, y2 = bbox
        return [self.image_size[1] - x2, self.image_size[0] - y2,
                self.image_size[1] - x1, self.image_size[0] - y1]

    def transform3(self, bbox):
        # Rotate 270 degrees
        x1, y1, x2, y2 = bbox
        return [y1, self.image_size[0] - x2, y2, self.image_size[0] - x1]

    def transform4(self, bbox):
        # Vertical flip
        x1, y1, x2, y2 = bbox
        return [x1, self.image_size[0] - y2, x2, self.image_size[0] - y1]

    def transform5(self, bbox):
        # Rotate 90 degrees + vertical flip
        bbox = self.transform1(bbox)
        return self.transform4(bbox)

    def transform6(self, bbox):
        # Rotate 180 degrees + vertical flip
        bbox = self.transform2(bbox)
        return self.transform4(bbox)

    def transform7(self, bbox):
        # Rotate 270 degrees + vertical flip
        bbox = self.transform3(bbox)
        return self.transform4(bbox)

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]
augment_bbox = Augment_BBox_torch((256,256))
transforms_aug_bbox = [method for method in dir(augment_bbox) if callable(getattr(augment_bbox, method)) if not method.startswith('_')]
class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution

        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        if split == 'train':
            gt_dir = 'train_C'
            input_dir = 'train_A'
            mask_dir = 'train_B'
        else:
            gt_dir = 'test_C'
            input_dir = 'test_A'
            mask_dir = 'test_B'

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
            mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

            self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
            self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
            self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]

            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.split == 'train':
                hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
            else:
                hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')
            hr_name = hr_name.replace('_A', '_C')
            img_HR = Image.open(hr_name).convert("RGB")
            img_mask = Image.open(self.mask_path[index]).convert("1")
            if self.need_LR:
                img_LR = Image.open(self.sr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR, img_mask] = Util.transform_augment(
                [img_LR, img_SR, img_HR, img_mask], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'mask': img_mask, 'Index': index, 'LR_path': self.sr_path[index]}
        else:
            [img_SR, img_HR, img_mask] = Util.transform_augment(
                [img_SR, img_HR, img_mask], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'mask': img_mask, 'Index': index}

class TrainDataset(Dataset):
    def __init__(self, dataroot, sam_inp_size=1024, mask_inp_size=256):
        gt_dir = 'train_C'
        input_dir = 'train_A'
        mask_dir = 'train_B'

        clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

        self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
        self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
        self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]

        self.img_transform = transforms.Compose([
            transforms.Resize((sam_inp_size, sam_inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((mask_inp_size, mask_inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.hr_path)

    def __getitem__(self, index):
        img_LR = None

        img_SR_original = Image.open(self.sr_path[index]).convert("RGB")

        hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
        hr_name = hr_name.replace('_A', '_C')

        img_HR_original = Image.open(hr_name).convert("RGB")
        img_mask_original = Image.open(self.mask_path[index]).convert("1")
        [shadow_img_SR, shadow_img_HR, shadow_img_mask] = Util.transform_augment([img_SR_original, img_HR_original, img_mask_original], split="train", min_max=(-1, 1))

        # sam input and sam mask
        sam_img_SR = self.img_transform(img_SR_original)
        sam_img_mask = self.mask_transform(img_mask_original)




        return {'HR': shadow_img_HR, 'SR': shadow_img_SR, 'mask': shadow_img_mask,
                'Index': index, 'LR_path': self.sr_path[index],
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask}

class TestDataset(Dataset):
    def __init__(self, dataroot, sam_inp_size=1024, mask_inp_size=256, data_len=-1):
        gt_dir = 'test_C'
        input_dir = 'test_A'
        mask_dir = 'test_B'
        self.data_len = data_len

        clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

        self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
        self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
        self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]

        self.img_transform = transforms.Compose([
            transforms.Resize((sam_inp_size, sam_inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((mask_inp_size, mask_inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])


    def __len__(self):
        if self.data_len == -1:
            return len(self.hr_path)
        else:
            return self.data_len

    def __getitem__(self, index):
        img_SR_original = Image.open(self.sr_path[index]).convert("RGB")
        hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')
        # hr_name = self.sr_path[index].replace('shad.jpg', 'noshad.jpg')
        hr_name = hr_name.replace('_A', '_C')
        img_HR_original = Image.open(hr_name).convert("RGB")
        img_mask_original = Image.open(self.mask_path[index]).convert("1")
        [shadow_img_SR, shadow_img_HR, shadow_img_mask] = Util.transform_augment([img_SR_original, img_HR_original, img_mask_original], split="test", min_max=(-1, 1))
        # sam input and sam mask
        sam_img_SR = self.img_transform(img_SR_original)
        sam_img_mask = self.mask_transform(img_mask_original)
        filename = self.sr_path[index].split('/')[-1].split('.')[0]


        return {'HR': shadow_img_HR, 'SR': shadow_img_SR, 'mask': shadow_img_mask,
                'Index': index, 'filename': filename,
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask}

class TuneSAM(Dataset):
    def __init__(self, dataroot, gt_mask_dir, yaml_path, data_len=-1):
        self.data_len = data_len

        self.phase = dataroot.split('/')[-1]
        if self.phase == 'train':
            gt_dir = 'train_C'
            input_dir = 'train_A'
            mask_dir = gt_mask_dir

        elif self.phase == 'test':
            gt_dir = 'test_C'
            input_dir = 'test_A'
            mask_dir = gt_mask_dir

        clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
        mask_files = sorted(os.listdir(mask_dir))
        # mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

        self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
        self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
        self.mask_path = [os.path.join(mask_dir, x) for x in mask_files]

        self.img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        with open(yaml_path, 'r') as file:
            self.bbox_data = yaml.safe_load(file)


    def __len__(self):
        if self.data_len==-1:
            return len(self.hr_path)
        else:
            return self.data_len

    def __getitem__(self, index):
        img_SR_original = Image.open(self.sr_path[index]).convert("RGB")


        if self.phase == 'train':
            hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
        elif self.phase == 'test':
            if "_shadow.png" in self.sr_path[index]:
                hr_name = self.sr_path[index].replace('_shadow.png', '_free.png')
            elif '_shad.jpg' in self.sr_path[index]:
                hr_name = self.sr_path[index].replace('shad.jpg', 'noshad.jpg')
            else:
                # hr_name = self.sr_path[index]
                hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')


        hr_name = hr_name.replace('_A', '_C')

        img_HR_original = Image.open(hr_name).convert("RGB")
        img_mask_original = Image.open(self.mask_path[index]).convert("L")
        [shadow_img_SR, shadow_img_HR, shadow_img_mask] = Util.transform_augment([img_SR_original, img_HR_original, img_mask_original], split="train", min_max=(-1, 1))

        # sam input and sam mask
        sam_img_SR = self.img_transform(img_SR_original)
        sam_img_mask = self.mask_transform(img_mask_original)
        image_name = self.sr_path[index].split('/')[-1].split('.')[0]
        bbox = self.bbox_data[image_name]
        # bbox = bbox[:1]  # Take only the first four bounding boxes if more are present
        # while len(bbox) < 4:
        #     bbox.append([-1, -1, -1, -1])     # Pad with -1 if fewer than four bboxes
        # bbox = np.expand_dims(np.array(bbox), axis=0)
        bbox = np.array(bbox)
        filename = self.sr_path[index].split('/')[-1].split('.')[0]


        return {'HR': shadow_img_HR, 'SR': shadow_img_SR, 'mask': shadow_img_mask,
                'Index': index, 'LR_path': self.sr_path[index],
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask, #sam gt mask, division mask
                'bbox': bbox, 'filename': filename,}

class TuneSAM_patch(Dataset):
    def __init__(self, dataroot, gt_mask_dir, yaml_path, data_len=-1):
        self.data_len = data_len

        self.phase = dataroot.split('/')[-1]
        if self.phase == 'train':
            gt_dir = 'train_C'
            input_dir = 'train_A'
            mask_dir = gt_mask_dir

        elif self.phase == 'test':
            gt_dir = 'test_C'
            input_dir = 'test_A'
            mask_dir = gt_mask_dir

        clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
        mask_files = sorted(os.listdir(mask_dir))
        # mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

        self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
        self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
        self.mask_path = [os.path.join(mask_dir, x) for x in mask_files]

        self.img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.img_transform_SAM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform_SAM = transforms.Compose([
            transforms.ToTensor(),
        ])

        with open(yaml_path, 'r') as file:
            self.bbox_data = yaml.safe_load(file)

    def __len__(self):
        if self.data_len == -1:
            return len(self.hr_path)
        else:
            return self.data_len

    def __getitem__(self, index):
        img_SR_original = Image.open(self.sr_path[index]).convert("RGB")

        if self.phase == 'train':
            hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
        elif self.phase == 'test':
            if "_shadow.png" in self.sr_path[index]:
                hr_name = self.sr_path[index].replace('_shadow.png', '_free.png')
            elif '_shad.jpg' in self.sr_path[index]:
                hr_name = self.sr_path[index].replace('shad.jpg', 'noshad.jpg')
            else:
                # hr_name = self.sr_path[index]
                hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')

        hr_name = hr_name.replace('_A', '_C')

        img_HR_original = Image.open(hr_name).convert("RGB")
        img_mask_original = Image.open(self.mask_path[index]).convert("L")

        [shadow_img_SR, shadow_img_HR, shadow_img_mask] = Util.transform_augment_unresize(
            [img_SR_original, img_HR_original, img_mask_original], min_max=(-1, 1))

        resize = transforms.Resize((shadow_img_SR.shape[1], shadow_img_SR.shape[2]), antialias=True)
        shadow_img_mask = resize(shadow_img_mask.unsqueeze(0)).squeeze(0)

        ps = 256
        H = shadow_img_SR.shape[1]
        W = shadow_img_SR.shape[2]
        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = shadow_img_HR[:, r:r + ps, c:c + ps]
        noisy = shadow_img_SR[:, r:r + ps, c:c + ps]
        SAM_mask = shadow_img_mask[:, r:r + ps, c:c + ps]
        SAM_SR = self.img_transform_SAM(img_SR_original)[:, r:r + ps, c:c + ps]

        random_number = random.getrandbits(3)
        apply_trans = transforms_aug[random_number]
        apply_trans_bbox = transforms_aug_bbox[random_number]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)
        SAM_mask = getattr(augment, apply_trans)(SAM_mask)
        SAM_SR = getattr(augment, apply_trans)(SAM_SR)
        # SAM_mask = torch.unsqueeze(SAM_mask, dim=0)

        # sam input and sam mask
        img_resize = transforms.Resize((1024, 1024))
        sam_img_SR = img_resize(SAM_SR)
        sam_img_mask = SAM_mask
        image_name = self.sr_path[index].split('/')[-1].split('.')[0]
        filename = self.sr_path[index].split('/')[-1].split('.')[0]
        bbox = self.bbox_data[image_name]
        bbox = np.array(bbox)
        H, W = shadow_img_SR.shape[1], shadow_img_SR.shape[2]
        aligned_bbox = self.align_bbox(bbox, original_size=(1024, 1024), new_size=(W, H))
        cropped_box = self.crop_bbox(aligned_bbox, r, c, ps)
        if cropped_box is not None:
            cropped_box = np.array(cropped_box)
            cropped_box = getattr(augment_bbox, apply_trans_bbox)(cropped_box)
        # save shadow_img_SR
        # torchvision.utils.save_image(noisy,"/home/xinrui/projects/ShadowDiffusion_orig/shadow_SR_patch.png")
        # SAM_mask_numpy = SAM_mask.numpy()
        # print(cropped_box)
        cropped_box = np.array(cropped_box)

        return {'HR': clean, 'SR': noisy, 'mask': sam_img_mask,
                'Index': index, 'LR_path': self.sr_path[index],
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask,  # sam gt mask, division mask
                'bbox': cropped_box, 'filename': filename}

    @staticmethod
    def align_bbox(bbox, original_size=(1024, 1024), new_size=None):
        if new_size is None:
            raise ValueError("New size must be provided")

        # Calculate scale factors
        width_scale = new_size[0] / original_size[0]
        height_scale = new_size[1] / original_size[1]
        # print(bbox)

        x_min, y_min, x_max, y_max = bbox

        # Apply scaling
        new_x_min = int(x_min * width_scale)
        new_y_min = int(y_min * height_scale)
        new_x_max = int(x_max * width_scale)
        new_y_max = int(y_max * height_scale)

        # Ensure the coordinates are within the new image bounds
        new_x_min = max(0, min(new_x_min, new_size[0] - 1))
        new_y_min = max(0, min(new_y_min, new_size[1] - 1))
        new_x_max = max(0, min(new_x_max, new_size[0] - 1))
        new_y_max = max(0, min(new_y_max, new_size[1] - 1))

        aligned_bbox=[new_x_min, new_y_min, new_x_max, new_y_max]

        return np.array(aligned_bbox)

    @staticmethod
    def crop_bbox(bbox, r, c, ps):
        x_min, y_min, x_max, y_max = bbox

        # Adjust coordinates
        x_min = max(x_min - c, 0)
        y_min = max(y_min - r, 0)
        x_max = min(x_max - c, ps)
        y_max = min(y_max - r, ps)

        # Check if the bounding box is still within the cropped area
        if x_min >= ps or y_min >= ps or x_max <= 0 or y_max <= 0:
            return [0,0,0,0]  # Bounding box is completely outside the cropped area

        # Ensure the bounding box is within the cropped image
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(ps, x_max)
        y_max = min(ps, y_max)

        return [x_min, y_min, x_max, y_max]


class TuneSAM_patch_test(Dataset):
    def __init__(self, dataroot, gt_mask_dir, yaml_path, data_len=-1):
        self.data_len = data_len

        self.phase = dataroot.split('/')[-1]
        if self.phase == 'train':
            gt_dir = 'train_C'
            input_dir = 'train_A'
            mask_dir = gt_mask_dir

        elif self.phase == 'test':
            gt_dir = 'test_C'
            input_dir = 'test_A'
            mask_dir = gt_mask_dir

        clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
        mask_files = sorted(os.listdir(mask_dir))
        # mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

        self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
        self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
        self.mask_path = [os.path.join(mask_dir, x) for x in mask_files]

        self.img_transform = transforms.Compose([
            # transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.img_transform_SAM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform_SAM = transforms.Compose([
            transforms.ToTensor(),
        ])

        with open(yaml_path, 'r') as file:
            self.bbox_data = yaml.safe_load(file)

    def __len__(self):
        if self.data_len == -1:
            return len(self.hr_path)
        else:
            return self.data_len

    def __getitem__(self, index):
        img_SR_original = Image.open(self.sr_path[index]).convert("RGB")
        if self.phase == 'train':
            hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
        else:
            if "_shadow.png" in self.sr_path[index]:
                hr_name = self.sr_path[index].replace('_shadow.png', '_free.png')
            elif '_shad.jpg' in self.sr_path[index]:
                hr_name = self.sr_path[index].replace('shad.jpg', 'noshad.jpg')
            else:
                # hr_name = self.sr_path[index]
                hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')

        hr_name = hr_name.replace('_A', '_C')

        img_HR_original = Image.open(hr_name).convert("RGB")
        img_mask_original = Image.open(self.mask_path[index]).convert("L")

        [shadow_img_SR, shadow_img_HR, shadow_img_mask] = Util.transform_augment_unresize(
            [img_SR_original, img_HR_original, img_mask_original], min_max=(-1, 1))

        # sam input and sam mask
        sam_img_SR = self.img_transform(img_SR_original)
        sam_img_mask = self.mask_transform(img_mask_original)
        image_name = self.sr_path[index].split('/')[-1].split('.')[0]
        bbox = self.bbox_data[image_name]
        # bbox = bbox[:1]  # Take only the first four bounding boxes if more are present
        # while len(bbox) < 4:
        #     bbox.append([-1, -1, -1, -1])     # Pad with -1 if fewer than four bboxes
        # bbox = np.expand_dims(np.array(bbox), axis=0)
        bbox = np.array(bbox)
        filename = self.sr_path[index].split('/')[-1].split('.')[0]

        return {'HR': shadow_img_HR, 'SR': shadow_img_SR, 'mask': shadow_img_mask,
                'Index': index, 'LR_path': self.sr_path[index],
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask,  # sam gt mask, division mask
                'bbox': bbox, 'filename': filename}