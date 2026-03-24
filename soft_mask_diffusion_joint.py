import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import hydra
import torch
import utils
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.sam_adapter.sam_adapt import SAM
import core.metrics as Metrics
from utils.dataset_utils import splitimage, mergeimage, splitbbox
from data.LRHR_dataset import TuneSAM, TuneSAM_patch, TuneSAM_patch_test
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
# from warmup_scheduler import GradualWarmupScheduler
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from warmup_scheduler import GradualWarmupScheduler
from model.sr3_modules import transformer

# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from omegaconf import DictConfig
import model.networks as networks
from model.sam_adapter.iou_loss import IOU
from lightning.pytorch.loggers import TensorBoardLogger
from model.segment_anything.sam_lora import LoRA_Sam
from model.segment_anything import sam_model_registry


def apply_low_pass_filter(mask: object, kernel_size: object = 3) -> object:
    # Create a Gaussian kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(mask.device)

    # Apply the low-pass filter using convolution
    filtered_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)

    return filtered_mask


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


def total_variation_loss(img, beta=1.0):
    """
    Compute the Total Variation Loss.
    :param img: Tensor of shape (B, C, H, W)
    :param beta: TV loss hyperparameter
    :return: Scalar tensor representing the TV loss
    """
    batch_size, channel, height, width = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return beta * (tv_h + tv_w) / (batch_size * channel * height * width)


def contour_gradient_penalty_loss(img):
    gray_tensor = img
    penumbra_area = ((gray_tensor > 50./255) & (gray_tensor < 220./255)).float()
    # penumbra = penumbra_area.cpu().numpy()

    indices = torch.nonzero(penumbra_area.squeeze())
    if len(indices) > 0:
        c_mean = torch.mean(indices.float(), dim=0)
        cX, cY = int(c_mean[1].item()), int(c_mean[0].item())
    else:
        cX, cY = 0, 0

    sobelx = F.conv2d(gray_tensor,
                      torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).unsqueeze(1).float().to(gray_tensor.device),
                      padding=1)
    sobely = F.conv2d(gray_tensor,
                      torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).unsqueeze(1).float().to(gray_tensor.device),
                      padding=1)
    sobelx = sobelx * penumbra_area
    sobely = sobely * penumbra_area
    batch_size, channel, height, width = gray_tensor.size()
    x_coords = torch.arange(width).unsqueeze(0).repeat(height, 1).to(gray_tensor.device)
    y_coords = torch.arange(height).unsqueeze(1).repeat(1, width).to(gray_tensor.device)
    weight_map_x = torch.ones((height, width)).to(sobelx.device)
    weight_map_y = torch.ones((height, width)).to(sobely.device)
    # Quadrant
    weight_map_x = weight_map_x * (x_coords < cX).float() * -2 + 1  # -1 for left, 1 for right
    weight_map_y = weight_map_y * (y_coords < cY).float() * -2 + 1  # -1 for top, 1 for bottom
    mod_sobelx = sobelx * weight_map_x.unsqueeze(0).unsqueeze(0)
    mod_sobely = sobely * weight_map_y.unsqueeze(0).unsqueeze(0)
    num_pixels = penumbra_area.sum()
    smooth = 1e-7
    # orientation_loss = (torch.sum(torch.clamp(mod_sobelx, 0)) + torch.sum(torch.clamp(mod_sobely, 0)) + smooth) / (
    #         2 * num_pixels + 100 * smooth)

    positive_mod_sobelx = F.relu(mod_sobelx)
    positive_mod_sobely = F.relu(mod_sobely)
    orientation_loss = (positive_mod_sobelx.sum() + positive_mod_sobely.sum() + smooth) / (
                2 * num_pixels + 10 * smooth)

    return orientation_loss


def contour_gradient_noPenumbra_penalty_loss(img):
    gray_tensor = img
    penumbra_area = ((gray_tensor > 50./255) & (gray_tensor < 220./255)).float()
    indices = torch.nonzero(penumbra_area.squeeze())
    if len(indices) > 0:
        c_mean = torch.mean(indices.float(), dim=0)
        cX, cY = int(c_mean[1].item()), int(c_mean[0].item())
    else:
        cX, cY = 0, 0

    sobelx = F.conv2d(gray_tensor,
                      torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).unsqueeze(1).float().to(gray_tensor.device),
                      padding=1)
    sobely = F.conv2d(gray_tensor,
                      torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).unsqueeze(1).float().to(gray_tensor.device),
                      padding=1)
    # sobelx = sobelx * penumbra_area
    # sobely = sobely * penumbra_area
    batch_size, channel, height, width = gray_tensor.size()
    x_coords = torch.arange(width).unsqueeze(0).repeat(height, 1).to(gray_tensor.device)
    y_coords = torch.arange(height).unsqueeze(1).repeat(1, width).to(gray_tensor.device)
    weight_map_x = torch.ones((height, width)).to(sobelx.device)
    weight_map_y = torch.ones((height, width)).to(sobely.device)
    # Quadrant
    weight_map_x = weight_map_x * (x_coords < cX).float() * -2 + 1  # -1 for left, 1 for right
    weight_map_y = weight_map_y * (y_coords < cY).float() * -2 + 1  # -1 for top, 1 for bottom
    mod_sobelx = sobelx * weight_map_x.unsqueeze(0).unsqueeze(0)
    mod_sobely = sobely * weight_map_y.unsqueeze(0).unsqueeze(0)
    smooth = 1e-7
    # orientation_loss = (torch.sum(torch.clamp(mod_sobelx, 0)) + torch.sum(torch.clamp(mod_sobely, 0)) + smooth) / (
    #         2 * num_pixels + 100 * smooth)

    positive_mod_sobelx = F.relu(mod_sobelx)
    positive_mod_sobely = F.relu(mod_sobely)
    orientation_loss = (positive_mod_sobelx.sum() + positive_mod_sobely.sum() + smooth) / (
                2 * (batch_size * channel * height * width))

    return orientation_loss


class SAM_with_bbox(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
    ):
        super(SAM_with_bbox, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, bbox):
        self.features = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=bbox,
            # boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings = self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        return low_res_masks # 256x256

class SamShadowDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = args.datasets.train.num_workers
        self.shuffle = args.datasets.train.use_shuffle

    def setup(self, stage="fit"):
        if stage == "fit":
            if self.args.patch_tricky:
                self.train_data = TuneSAM_patch(self.args.datasets.train.dataroot, self.args.datasets.train.gt_mask_dir,
                                          self.args.bbox_path)
                self.val_data = TuneSAM_patch(self.args.datasets.train.dataroot, self.args.datasets.train.gt_mask_dir,
                                        self.args.bbox_path, data_len=self.args.datasets.train.val_data_len)
            else:
                self.train_data = TuneSAM(self.args.datasets.train.dataroot, self.args.datasets.train.gt_mask_dir, self.args.bbox_path)
                self.val_data = TuneSAM(self.args.datasets.train.dataroot, self.args.datasets.train.gt_mask_dir, self.args.bbox_path, data_len=self.args.datasets.train.val_data_len)
        elif stage == "test":
            if self.args.patch_tricky:
                self.test_data = TuneSAM_patch_test(self.args.datasets.test.dataroot, self.args.datasets.test.gt_mask_dir, self.args.test_bbox_path, data_len=self.args.datasets.test.data_len)
            else:
                self.test_data = TuneSAM(self.args.datasets.test.dataroot, self.args.datasets.test.gt_mask_dir, self.args.test_bbox_path, data_len=self.args.datasets.test.data_len)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_data,
                                batch_size=self.args.datasets.train.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,
                                pin_memory=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_data, batch_size=self.args.datasets.test.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(self.test_data, batch_size=self.args.datasets.test.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        return dataloader


# define a LightningModule, in which I have shadowDiffusion and SAM_adapter two models
def laplacian_magnitude_loss(soft_mask):
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=soft_mask.dtype, device=soft_mask.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
    laplacian = F.conv2d(soft_mask, laplacian_kernel, padding=1)
    magnitude = torch.abs(laplacian)
    mean_magnitude = torch.mean(magnitude)
    return mean_magnitude


def gradient_orientation_map(input_image):
    sobelx = F.conv2d(input_image, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                                                dtype=input_image.dtype,
                                                device=input_image.device), padding=1)
    sobely = F.conv2d(input_image, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                                                dtype=input_image.dtype,
                                                device=input_image.device), padding=1)
    gradient_orientation = torch.atan2(sobely, sobelx)
    return gradient_orientation


def gradient_orientation_loss(soft_mask, shadow_input):
    penumbra_area = soft_mask.detach()
    # set a threshold
    threshold_low = 0.01
    threshold_high = 0.995
    penumbra_area = torch.where((penumbra_area >= threshold_low) & (penumbra_area <= threshold_high), 1.0, 0.0)
    shadow_input = torch.mean(shadow_input, dim=1, keepdim=True)
    shadow_input = F.interpolate(shadow_input, (penumbra_area.shape[2], penumbra_area.shape[3]), mode='bilinear', align_corners=False)
    shadow_input = apply_low_pass_filter(shadow_input)

    gradient_orientation_shadow_input = gradient_orientation_map(shadow_input*penumbra_area)
    gradient_orientation_soft_mask = gradient_orientation_map(soft_mask * penumbra_area)
    # using "+" because the two maps has different orientation
    cosine_similarity = torch.cos(gradient_orientation_shadow_input + gradient_orientation_soft_mask)
    gradient_loss = 1 - cosine_similarity
    gradient_loss = torch.mean(gradient_loss)
    return gradient_loss


def soft_mask_loss_metric(soft_mask, shadow_input):
    laplacian_loss = laplacian_magnitude_loss(soft_mask)
    gradient_loss = gradient_orientation_loss(soft_mask, shadow_input)
    # soft_hard_mask_consistency = soft_hard_mask_consistency(soft_mask, hard_mask)
    return laplacian_loss, gradient_loss


class SamShadow(L.LightningModule):
    def __init__(self, sam_adapter, shadow_diffusion, args, path=None, steps_per_epoch=168):
        super(SamShadow, self).__init__()
        self.args = args
        self.save_model_name = args.name
        # automatic optimization
        self.automatic_optimization = True
        self.steps_per_epoch = int(steps_per_epoch)
        self.diffusion = shadow_diffusion(args)
        self.optimizer_param = args.train.optimizer
        self.diffusion.set_loss()
        # self.sam = sam_adapter(args.sam.input_size, args.sam.loss)
        self.sam_model = sam_model_registry[args.model_type](checkpoint=path.SAM)
        self.sam = SAM_with_bbox(image_encoder=self.sam_model.image_encoder,
                                 prompt_encoder=self.sam_model.prompt_encoder,
                                 mask_decoder=self.sam_model.mask_decoder)
        self.sam_lora = LoRA_Sam(self.sam, args.sam_rank)
        self.lora = self.sam_lora.sam

        self.MSEloss = torch.nn.MSELoss()
        self.loss = []
        self.diffusion_loss = []
        self.mse_loss = []
        self.contour_gradient_loss = []
        self.tv_loss = []

        # if args.phase == 'train':
        self.diffusion.set_new_noise_schedule(args['model']['beta_schedule']['train'])

        # load ShadowDiffusion and LoraSAM finetuned model
        # if path is not None:
        #     self.load_pretrained_models(path)

        self.model_restoration = transformer.Uformer()

        self.save_hyperparameters()
        # for name, param in self.lora.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        # keep lora parameters on training
        # self.freeze_parameters(self.lora.parameters())
        # trian lora mask encoder and mask decoder
        self.freeze_parameters(self.lora.prompt_encoder.parameters())
        a = 1

    @staticmethod
    def freeze_parameters(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_parameters(params):
        for param in params:
            param.requires_grad = True

    def forward(self, train_data):
        # SAM lora
        sam_input = train_data['sam_SR']
        bbox = train_data['bbox']
        pred_mask = self.lora(sam_input, bbox)
        pred_mask = torch.sigmoid(pred_mask)
        division_mask = train_data['sam_mask']
        self.mse_loss = self.MSEloss(pred_mask, division_mask)
        self.log('Loss/MSEloss', self.mse_loss.item(), on_step=True)
        self.contour_gradient_loss = contour_gradient_noPenumbra_penalty_loss(pred_mask)
        self.log('Loss/contour_gradient_loss', self.contour_gradient_loss.item(), on_step=True)
        # self.tv_loss = total_variation_loss(pred_mask, beta=self.args.tv_loss_beta)
        # self.log('Loss/TVloss', self.tv_loss.item(), on_step=True)

        # pred_mask = train_data['mask']
        # ShadowDiffusion
        if self.args.detach:
            train_data['mask'] = pred_mask.detach()
        else:
            train_data['mask'] = pred_mask
        l_pix = self.diffusion(train_data)
        b, c, h, w = train_data['HR'].shape
        self.diffusion_loss = l_pix.sum() / int(b * c * h * w)
        self.log('l_pix_loss', self.diffusion_loss.item(), on_step=True)

        if self.args.loss_type == 'mse':
            self.loss = self.mse_loss + self.diffusion_loss
        # if self.args.loss_type == 'mse+tv':
        #     self.loss = self.mse_loss + self.tv_loss
        if self.args.loss_type == 'mse+constrain':
            self.loss = self.mse_loss + self.args.grad_loss_weight * self.contour_gradient_loss + self.args.diffusion_loss_weight * self.diffusion_loss
        # if self.args.loss_type == 'mse+tv+constrain':
        #     self.loss = self.mse_loss + self.tv_loss+ self.args.grad_loss_weight * self.contour_gradient_loss

        # # optimize SAM and diffusion saperately
        # sam_optimizer, diffusion_optimizer = self.optimizers()
        #
        # self.manual_backward(self.loss, retain_graph=True, create_graph=True)
        # sam_optimizer.step()
        # sam_optimizer.zero_grad()
        #
        # self.manual_backward(self.diffusion_loss)
        # diffusion_optimizer.step()
        # diffusion_optimizer.zero_grad()


        return {'loss': self.loss,
                'pred_mask': pred_mask,
                'residual_mask': division_mask}


    def sample(self, data):
        if self.args.patch_tricky:
            restored_list = []
            predict_mask_list = []
            diffusion_mask_list = []
            tile = 256
            repeat = 1
            base_repeat = 1
            tile_overlap = 30
            for _ in range(repeat // base_repeat):
                # rgb_gt = data['HR'].numpy().squeeze().transpose((1, 2, 0))
                input = data['SR'].repeat(base_repeat, 1, 1, 1).cuda()
                bbox = data['bbox']
                sam_input = data['sam_SR'].repeat(base_repeat, 1, 1, 1).cuda()
                B, C, H, W = input.shape

                sam_data, _, starts = splitimage(sam_input, crop_size=tile, overlap_size=tile_overlap)
                split_data, small_data, starts = splitimage(input, crop_size=tile, overlap_size=tile_overlap)
                split_bbox, starts = splitbbox(input, bbox, crop_size=tile, overlap_size=tile_overlap)

                merge_predict_mask = [None] * len(split_data)
                merge_diffusion_mask = [None] * len(split_data)
                for i, (sam, rm, box)in enumerate(zip(sam_data, small_data, split_bbox)):
                    mask = torch.sigmoid(self.lora(sam, box))
                    merge_predict_mask[i] = mask
                    split_data[i], merge_diffusion_mask[i] = self.diffusion.super_resolution(rm, mask, continous=False)
                restored_list.append(mergeimage(split_data, starts, crop_size=tile, resolution=(B, C, H, W)))
                predict_mask_list.append(mergeimage(merge_predict_mask, starts, crop_size=tile, resolution=(B, 1, H, W)))
                diffusion_mask_list.append(mergeimage(merge_diffusion_mask, starts, crop_size=tile, resolution=(B, 1, H, W)))
            shadow_removal_sr = torch.mean(torch.cat(restored_list, dim=0), dim=0, keepdim=True)
            pred_mask = torch.mean(torch.cat(predict_mask_list, dim=0), dim=0, keepdim=True)
            diffusion_mask_pred = torch.mean(torch.cat(diffusion_mask_list, dim=0), dim=0, keepdim=True)
        #     # shadow_removal_sr = torch.clamp(restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))


        elif self.args.h_hat:
            sam_input = data['sam_SR']
            bbox = data['bbox']
            pred_mask = self.lora(sam_input, bbox)
            pred_mask = torch.sigmoid(pred_mask)

            data['mask'] = pred_mask

            x_hat = self.model_restoration((data['SR'] + 1) / 2, data['mask'])
            x_hat = torch.clamp(x_hat, 0, 1)
            # x_hat = x_hat * 2 - 1
            h_hat = (data['SR'] + 1) / (2 * (x_hat) + 1e-4)
            # h_hat = torch.clamp(h_hat, 0, 1)
            h_hat = torch.where(h_hat == 0, h_hat + 1e-4, h_hat)
            shadow_removal_sr, diffusion_mask_pred = self.diffusion.super_resolution_d(data['SR'], data['mask'], h_hat, continous=True)
        else:
            sam_input = data['sam_SR']
            bbox = data['bbox']
            pred_mask = self.lora(sam_input, bbox)
            pred_mask = torch.sigmoid(pred_mask)

            data['mask'] = pred_mask
            # pred_mask = data['mask']
            shadow_removal_sr, diffusion_mask_pred = self.diffusion.super_resolution(data['SR'], data['mask'],
                                                                                     continous=False)
        return shadow_removal_sr, diffusion_mask_pred, pred_mask

    def print_param_values(self):
        print("SAM parameters:")
        for name, param in self.sam.mask_decoder.soft_mask_decoder_adapter.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
            print(f"{name}: {param.data.mean().item()}")
            if param.grad is not None:
                print(f"{name}: {param.grad.mean().item()}")
            else:
                print(f"{name}: NO gradient")
            break

    def training_step(self, train_data, batch_idx):
        result = self(train_data)
        loss = result['loss']
        pred_mask = result['pred_mask']
        residual_mask = result['residual_mask']
        filename = train_data['filename']
        log_img = train_data['SR'][0].clamp_(-1, 1)
        lr_img = (log_img + 1) / 2

        # print("After training step:")
        # self.print_param_values()
        if self.global_step % 100 == 0:
            self.logger.experiment.add_image('Train/sam_pred_mask', pred_mask[0], self.global_step)
            self.logger.experiment.add_image('Train/residual_mask', residual_mask[0], self.global_step)
            self.logger.experiment.add_image('Train/lr_image', lr_img, self.global_step)
            self.logger.experiment.add_text('Train/filename_sam_pred_mask', filename[0], self.global_step)

        return loss



    def validation_step(self, val_data, batch_idx):
        shadow_removal_sr, diffusion_mask_pred, _ = self.sample(val_data)
        res = Metrics.tensor2img(shadow_removal_sr)
        hr_img = Metrics.tensor2img(val_data['HR'])
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        log_img = shadow_removal_sr[0].clamp_(-1, 1)
        log_img = (log_img + 1) / 2
        self.logger.experiment.add_image('Val/Shadow_Removal', log_img, self.global_step)
        self.logger.experiment.add_image('Val/gt_image', (val_data['HR'][0]+1)/2,self.global_step)
        self.log('Val/psnr', eval_psnr, sync_dist=True)
        self.log('Val/ssim', eval_ssim, sync_dist=True)


    def predict_step(self, test_data):
        self.diffusion.eval()
        shadow_removal_sr, diffusion_mask_pred, sam_pred_soft_mask = self.sample(test_data)
        test_normalize = False

        if test_normalize:
            hr_img = Metrics.tensor2img(test_data['HR'])
            res = Metrics.tensor2img(shadow_removal_sr)
            avg_channel = np.mean(res, axis=(0, 1))
            avg_channel_gt = np.mean(hr_img, axis=(0, 1))
            res = res * avg_channel_gt / avg_channel
        else:
            res = Metrics.tensor2img(shadow_removal_sr)
            hr_img = Metrics.tensor2img(test_data['HR'])

            # res = cv2.resize(res, [256, 256], interpolation=cv2.INTER_AREA)
            # hr_img = cv2.resize(hr_img, [256, 256], interpolation=cv2.INTER_AREA)
            # res = cv2.cvtColor(res.astype(np.float32), cv2.COLOR_RGB2GRAY)
            # hr_img = cv2.cvtColor(hr_img.astype(np.float32), cv2.COLOR_RGB2GRAY)


        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        filename = test_data['filename']

        return eval_psnr, eval_ssim, filename, sam_pred_soft_mask, res, diffusion_mask_pred, test_data['HR']


    def load_pretrained_models(self, path):
        if path.ddpm is not None:
            gen_path = '{}_gen.pth'.format(path.ddpm)
            self.diffusion.load_state_dict(torch.load(gen_path), strict=False)
            # checkpoint = torch.load(gen_path, map_location=lambda storage, loc: storage)
            # self.load_state_dict(checkpoint['state_dict'], strict=False)
        if path.SAM_Diffusion is not None:
            gen_path = path.SAM_Diffusion
            checkpoint = torch.load(gen_path, map_location=lambda storage, loc: storage)
        # filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'diffusion' in k}
        # self.load_state_dict(filtered_state_dict, strict=False)
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        if path.lora_sam is not None:
            sam_path = path.lora_sam
            checkpoint_sam = torch.load(sam_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint_sam['state_dict'], strict=False)
        print("Checkpoint loaded successfully.")

        # load tuned SAM
        # sam_state_dict = torch.load(path.lora_sam, map_location=self.device)
        # lora_state_dict = {}
        #
        # for key, value in sam_state_dict['state_dict'].items():
        #     if key.startswith('lora.'):
        #         lora_key = key.replace('lora.', '')
        #         lora_state_dict[lora_key] = value
        #
        # self.lora.load_state_dict(lora_state_dict, strict=False)


    def configure_optimizers(self):

        diffusion_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=self.args.train.optimizer.lr)
        warmup_epochs = 10
        scheduler_cosine = CosineAnnealingLR(diffusion_optimizer, self.args.train.max_epochs - warmup_epochs,
                                                                                  eta_min=1e-6)
        scheduler = GradualWarmupScheduler(diffusion_optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)



        return  {
            "optimizer": diffusion_optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # step or epoch
                "frequency": 1
                }
            }



# using SAM mask train diffusion
def train(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}",
                               name=args.version + "_logger")
    data_module = SamShadowDataModule(args)
    data_module.setup("fit")
    ckpt_path = args.ckpt_path
    # model = diffusion(networks.define_G, args, ckpt_path)
    model = SamShadow(SAM, networks.define_G, args, ckpt_path)
    model.load_pretrained_models(ckpt_path)

    # load training parameters

    save_model_name = args.name
    max_epochs = args.train.max_epochs
    save_every_n_epochs = args.train.every_n_epochs
    log_every_n_steps = 1
    log_every_n_epochs = 1
    accumulate_grad_batches = args.train.accumulate_grad_batches

    save_path = f"./experiments_lightning/{save_model_name}/{args.version}"
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename='{epoch}',
        # save_on_train_epoch_end=True
        save_top_k=-1,
        every_n_epochs=save_every_n_epochs,  # Save every 20 epochs
        save_last=True,  # Save the last model as well
    )
    trainer = L.Trainer(
        accelerator='gpu',
        precision=32,
        devices=args.train.gpu_ids,
        max_epochs=max_epochs,
        # gradient_clip_val=0.5,
        default_root_dir=save_path,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=20,
        log_every_n_steps=log_every_n_steps,
        accumulate_grad_batches=accumulate_grad_batches,  # mini batch size is 4 so the overall batchsize is 16
        strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True),
        # strategy=DDPStrategy(find_unused_parameters=True)
        # strategy=DDPStrategy(gradient_as_bucket_view=True)
    )
    trainer.fit(model, data_module)


def sample(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}", name=f"{args.version}")
    data_module = SamShadowDataModule(args)
    data_module.setup("test")
    # model = SamShadow.load_from_checkpoint(args.samshadow_ckpt_path)
    ckpt_path = args.ckpt_path
    model = SamShadow(SAM, networks.define_G, args, ckpt_path)
    # load restoration model
    # path_chk_rest_student = '/home/xinrui/projects/ShadowDiffusion/ShadowDiffusion_2/pretrained/ISTD_Plus/degradation_model.pth'
    # utils.load_checkpoint(model.model_restoration, path_chk_rest_student)

    # load checkpoints
    # checkpoint_sam = torch.load(args.samshadow_ckpt_path, map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint_sam['state_dict'], strict=False)
    model.load_pretrained_models(ckpt_path)
    predictor = L.Trainer(
        accelerator='gpu',
        devices=args.test.gpu_ids,
        max_epochs=-1,
        benchmark=True,
        logger=logger
    )
    predictions = predictor.predict(model, data_module)
    PSNR_SSIM_list_with_name = []
    PSNR = []
    SSIM = []
    save_path = args.save_result_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(len(predictions)):
        eval_psnr, eval_ssim, filename, sam_pred_soft_mask, res, diffusion_mask_pred, gt_image = predictions[i]
        filename = filename[0]
        PSNR_SSIM_list_with_name.append((f'{filename}_PSNR', eval_psnr))
        PSNR_SSIM_list_with_name.append((f'{filename}_SSIM', eval_ssim))
        PSNR.append(eval_psnr)
        SSIM.append(eval_ssim)
        # Save SR image
        sr_path = os.path.join(save_path, f'{filename}_sr.png')
        res = res.astype(np.uint8)
        sr_img = Image.fromarray(res)
        sr_img.save(sr_path)
        # Save HR image
        hr_path = os.path.join(save_path, f'{filename}_hr.png')
        hr_img = Metrics.tensor2img(gt_image)
        hr_img = Image.fromarray(hr_img)
        hr_img.save(hr_path)
        # Save mask
        soft_mask = sam_pred_soft_mask.squeeze().cpu().numpy()
        soft_mask = cv2.resize(soft_mask * 255.0, [256, 256], interpolation=cv2.INTER_AREA)
        soft_mask_path = os.path.join(save_path, f'{filename}.png')
        soft_mask_img = Image.fromarray(soft_mask.astype(np.uint8))
        soft_mask_img.save(soft_mask_path)

        mask = diffusion_mask_pred.squeeze().cpu().numpy() * 255
        mask_path = os.path.join(save_path, f'{filename}_diffusion_mask.png')
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.save(mask_path)

    psnr_mean = np.mean(PSNR)
    ssim_mean = np.mean(SSIM)
    print(f'PSNR: {psnr_mean:}')
    print(f'SSIM: {ssim_mean:}')

    with open(os.path.join(save_path, 'PSNR_SSIM_list.log'), 'w') as file:
        file.write(f"PSNR_mean: {psnr_mean}\n")
        file.write(f"SSIM_mean: {ssim_mean}\n")
        for key, value in PSNR_SSIM_list_with_name:
            file.write(f"{key}: {value}\n")


@hydra.main(config_path="./config", config_name='soft_mask_diffusion_SRD_joint', version_base=None)
# @hydra.main(config_path="./config", config_name='joint_LRSS', version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    L.seed_everything(1234)
    args.version = args.get('version', 'none_version')

    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        sample(args)


if __name__ == "__main__":
    main()
