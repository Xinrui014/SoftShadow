import os
import cv2
import numpy as np
import torch
from PIL import Image
import hydra
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.sam_adapter.sam_adapt import SAM
import core.metrics as Metrics
from data.LRHR_dataset import LRHRDataset, TrainDataset, TestDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from omegaconf import DictConfig
import model.networks as networks
from model.sam_adapter.iou_loss import IOU
from lightning.pytorch.loggers import TensorBoardLogger


def apply_low_pass_filter(mask: object, kernel_size: object = 3) -> object:
    # Create a Gaussian kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(mask.device)

    # Apply the low-pass filter using convolution
    filtered_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)

    return filtered_mask


class SamShadowDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = args.datasets.train.num_workers
        self.shuffle = args.datasets.train.use_shuffle

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_data = TrainDataset(self.args.datasets.train.dataroot, 1024)
            self.val_data = TestDataset(self.args.datasets.test.dataroot, 1024, data_len=3)
        elif stage == "test":
            self.test_data = TestDataset(self.args.datasets.test.dataroot, 1024)

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


class SamShadow(L.LightningModule):
    def __init__(self, shadow_diffusion, args, path=None, steps_per_epoch=168):
        super(SamShadow, self).__init__()
        self.args = args
        self.save_model_name = args.name
        self.automatic_optimization = False
        self.steps_per_epoch = int(steps_per_epoch)
        self.diffusion = shadow_diffusion(args)
        self.optimizer_param = args.train.optimizer
        self.diffusion.set_loss()
        self.diffusion_loss = []

        if args.phase == 'train':
            self.diffusion.set_new_noise_schedule(args['model']['beta_schedule']['train'])
        if path is not None:
            self.load_pretrained_models(path)
            if args.phase == 'train':
                # optimizer
                opt_path = '{}_opt.pth'.format(path.ddpm)
                opt = torch.load(opt_path, map_location=self.device)
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

        self.save_hyperparameters()


    def forward(self, train_data):

        l_pix = self.diffusion(train_data)
        b, c, h, w = train_data['HR'].shape
        self.diffusion_loss = l_pix.sum() / int(b * c * h * w)

        # optimize SAM and diffusion saperately
        diffusion_optimizer = self.optimizers()
        self.manual_backward(self.diffusion_loss)
        diffusion_optimizer.step()
        diffusion_optimizer.zero_grad()


        self.log('l_pix_loss', self.diffusion_loss.item(), on_step=True)

        return {'loss': self.diffusion_loss}


    def sample(self, data):
        shadow_removal_sr, diffusion_mask_pred = self.diffusion.super_resolution(data['SR'], data['mask'], continous=False)
        return shadow_removal_sr, diffusion_mask_pred

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


    def validation_step(self, val_data, batch_idx):
        shadow_removal_sr, diffusion_mask_pred = self.sample(val_data)
        res = Metrics.tensor2img(shadow_removal_sr)
        hr_img = Metrics.tensor2img(val_data['HR'])
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        log_img = shadow_removal_sr[0].clamp_(-1, 1)
        log_img = (log_img + 1) / 2
        self.logger.experiment.add_image('Val/Shadow_Removal', log_img, self.global_step)
        self.logger.experiment.add_image('Val/Diffusion_Mask', diffusion_mask_pred[0],self.global_step)
        self.log('Val/psnr', eval_psnr, sync_dist=True)
        self.log('Val/ssim', eval_ssim, sync_dist=True)


    def predict_step(self, test_data):
        shadow_removal_sr, diffusion_mask_pred = self.sample(test_data)
        res = Metrics.tensor2img(shadow_removal_sr)
        hr_img = Metrics.tensor2img(test_data['HR'])
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        filename = test_data['filename']
        return eval_psnr, eval_ssim, filename, shadow_removal_sr, diffusion_mask_pred, test_data['HR']


    def load_pretrained_models(self, path):
        gen_path = '{}_gen.pth'.format(path.ddpm)
        self.diffusion.load_state_dict(torch.load(gen_path), strict=False)

    def configure_optimizers(self):
        # param_groups = [
        #     {'params': self.sam.parameters(), 'lr': 0.0002},
        #     {'params': self.diffusion.parameters(), 'lr': 1e-05}
        # ]
        # optimizer = torch.optim.Adam(param_groups)

        diffusion_optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=1e-05)

        return [diffusion_optimizer]


# using SAM mask train diffusion
def train(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}",
                               name=args.version + "_logger")
    data_module = SamShadowDataModule(args)
    data_module.setup("fit")
    ckpt_path = args.ckpt_path
    # model = diffusion(networks.define_G, args, ckpt_path)
    model = SamShadow(networks.define_G, args, ckpt_path)

    # load training parameters
    save_model_name = args.name
    max_epochs = args.train.max_epochs
    save_every_n_epochs = args.train.every_n_epochs
    log_every_n_steps = 1
    log_every_n_epochs = 1

    save_path = f"./experiments_lightning/{save_model_name}/{args.version}"
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename='{epoch}',
        save_top_k=-1,
        every_n_epochs=save_every_n_epochs,  # Save every 20 epochs
        save_last=True,  # Save the last model as well
    )
    trainer = L.Trainer(
        accelerator='gpu',
        precision=16,
        devices=args.train.gpu_ids,
        max_epochs=max_epochs,
        # gradient_clip_val=0.5,
        default_root_dir=save_path,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=10,
        log_every_n_steps=log_every_n_steps,
        strategy=DDPStrategy(gradient_as_bucket_view=True)
    )
    trainer.fit(model, data_module)


def sample(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}", name="sample")
    data_module = SamShadowDataModule(args)
    data_module.setup("test")
    model = SamShadow.load_from_checkpoint(args.samshadow_ckpt_path)
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
        eval_psnr, eval_ssim, filename, shadow_removal_sr, diffusion_mask_pred, gt_image = predictions[i]
        filename = filename[0]
        PSNR_SSIM_list_with_name.append((f'{filename}_PSNR', eval_psnr))
        PSNR_SSIM_list_with_name.append((f'{filename}_SSIM', eval_ssim))
        PSNR.append(eval_psnr)
        SSIM.append(eval_ssim)
        # Save SR image
        sr_path = os.path.join(save_path, f'{filename}_sr.png')
        res = Metrics.tensor2img(shadow_removal_sr)
        sr_img = Image.fromarray(res)
        sr_img.save(sr_path)
        # Save HR image
        hr_path = os.path.join(save_path, f'{filename}_hr.png')
        hr_img = Metrics.tensor2img(gt_image)
        hr_img = Image.fromarray(hr_img)
        hr_img.save(hr_path)
        # Save mask
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


@hydra.main(config_path="./config", config_name='shadow_SRD', version_base=None)
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
