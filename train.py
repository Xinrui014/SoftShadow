import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
import random
import numpy as np
from tqdm import tqdm
from model.sam_adapter.sam_adapt import SAM
from model.sam_adapter.datasets import TrainDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import torch.nn as nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/shadow.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')


    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=False)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # Set Seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # init wandb
    wandb.init(project='samshadow')

    # init dataset for both diffusion and SAM-adapter
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # init and load diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Training parameter
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    # Init and load SAM_adapter

    # SAM-adapter epoch and min_learning rate
    max_epoch = 200
    # image_path = "/home/xinrui/projects/SAM-Adapter/data/ISTD_Dataset/train/train_A"
    # mask_path = "/home/xinrui/projects/SAM-Adapter/data/ISTD_Dataset/train/train_B"
    # datasets = TrainDataset(image_folder=image_path, mask_folder=mask_path)
    # loader = DataLoader(datasets, batch_size=1)
    # num_cuda_devices= torch.cuda.device_count()

    device = torch.device('cuda')
    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()

    adp_model = SAM(inp_size=1024, loss='iou').to(device)
    sam_checkpoint = torch.load("./experiments/official_test/SAM_adapter_ckpt/model_epoch_best.pth")
    adp_model.load_state_dict(sam_checkpoint, strict=False)

    # the prompt generator is not trained
    # for name, para in adp_model.named_parameters():
    #     if "image_encoder" in name and "prompt_generator" not in name:
    #         para.requires_grad_(False)

    # freeze all the parameter of adp_model
    for param in adp_model.parameters():
        param.requires_grad = False

    # Use DataParallel if multiple GPUs are available
    if num_gpus > 1:
        adp_model = nn.DataParallel(adp_model)
        adp_model = adp_model.module
    adp_model.train()

    # Initialize the AdamW optimizer for adp_model
    lr = 0.0002
    weight_decay = 0.0  # Set the weight decay value as needed
    # adp_model_optimizer = optim.AdamW(adp_model.parameters(), lr=lr, weight_decay=weight_decay)

    adp_params = [{'params': adp_model.parameters(), 'lr': 0.0002}]
    diffusion_params = [{'params': diffusion.netG.parameters(), 'lr': 3e-05}]
    params = adp_params + diffusion_params
    optimizer = optim.Adam(params)
    # lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=1e-7)

    # print name paramter in diffusion and the sam model
    # for name, para in adp_model.named_parameters():
    #     # print layer name and if it requires grad and if it is a leaf tensor
    #     # check if the tensor is not leaf but requires gradient
    #     if para.requires_grad and not para.is_leaf:
    #         print(name, para.requires_grad, para.is_leaf)
    # for name, para in diffusion.netG.named_parameters():
    #     if para.requires_grad and not para.is_leaf:
    #         print(name, para.requires_grad, para.is_leaf)

    # Set number of iterations to accumulate gradients
    gradient_accumulation_steps = 1

    # training
    if opt['phase'] == 'train':
        # n_epoch = current_epoch+max_epoch
        epoch = 0
        with tqdm(total=max_epoch, unit='epoch', position=0) as pbar:
            # while current_epoch < n_epoch:
            #     current_epoch += 1
            while epoch < max_epoch:
                epoch += 1
                current_epoch += 1
                for index, train_data in tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', position=1):
                    # current_step += 1
                    if current_step > n_iter:
                        break

                    # add SAM-adapter training mode


                    # inp = loader['inp']
                    # gt = loader['gt']
                    inp = train_data['SR']
                    gt = train_data['mask']
                    inp = F.interpolate(inp, size=(1024, 1024), mode='bilinear', align_corners=False)
                    gt = F.interpolate(gt, size=(1024, 1024), mode='bilinear', align_corners=False)

                    # diffusion.netG.zero_grad()
                    # adp_model.zero_grad()

                    with torch.autocast(device_type="cuda"):
                        adp_model.set_input(inp, gt)
                        # use the low_res output 256x256 and then resize to 160x160

                        pred_mask = adp_model.forward(inp)
                        train_data['mask'] = F.interpolate(pred_mask, size=(160,160), mode='bilinear', align_corners=False)
                        diffusion.feed_data(train_data)

                        l_pix = diffusion.netG(diffusion.data)
                        b, c, h, w = diffusion.data['HR'].shape
                        l_pix = l_pix.sum() / int(b * c * h * w)
                        l_pix = l_pix/gradient_accumulation_steps

                        wandb.log({'step_loss': l_pix.item()})

                        # Perform a single backward pass
                        l_pix.backward()

                        # print first layer gradient
                        # first_layer_grad = list(diffusion.netG.parameters())[0].grad
                        # logger.info("the first parameter's gradient is:", first_layer_grad[0][0])

                        if (index + 1) % gradient_accumulation_steps == 0:
                            current_step += 1
                            optimizer.step()
                            optimizer.zero_grad()
                            diffusion.ema_helper.update(diffusion.netG)
                        # lr_scheduler.step()
                        # print the current learning rate
                        # Get the current learning rate for each parameter group
                        # for i, param_group in enumerate(optimizer.param_groups):
                        #     current_lr = param_group['lr']
                        #     print(f"Current learning rate for parameter group {i}: {current_lr}")
                # lr_scheduler.step()
                # print the learning rate
                # current_lr_adp = lr_scheduler.get_last_lr()[0]
                # current_lr_diffusion = lr_scheduler.get_last_lr()[1]
                # print(
                #     f"Epoch [{epoch + 1}/{max_epoch}], Learning Rate (ADP): {current_lr_adp:.8f}, Learning Rate (Diffusion): {current_lr_diffusion:.8f}")
                # Set log
                wandb.log({'loss': l_pix.item(), 'epoch': epoch})
                diffusion.log_dict['l_pix'] = l_pix.item()



                # log
                if epoch % 10 == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)

                if epoch % 20 == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)
                    # save SAM_adapter
                    sam_adapter_ckpt = {
                        'epoch': epoch,
                        'model_state_dict': adp_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(sam_adapter_ckpt, os.path.join(opt['path']['checkpoint'], f"adp_model_epoch_{epoch}.pth"))


                pbar.update(1)
            logger.info('End of training.')
            wandb.finish()