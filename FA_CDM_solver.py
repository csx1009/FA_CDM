import argparse
import os
import nibabel as nib
import sys
import random

import torch
from tensorboard import summary

from SUNLoader import SUNVolumes
from guided_diffusion.BTCLoader import BTCVolumes
from guided_diffusion.CaffineLoader import CaffineVolumes
from guided_diffusion.GBMLoader import GBMVolumes
from guided_diffusion.HCPloader import HCPVolumes
from dpm_solver import NoiseScheduleVP, DPM_Solver, model_wrapper

# sys.path.append("..")
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
import nibabel as nib
import pathlib
import warnings
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import math
import torch


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta values in diffusion process
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    # result0=th.load('./Bratssliced/validation/000246/result0')
    #  print('loadedresult0', result0.shape)
    dist_util.setup_dist(devices=args.devices)
    print("dist_util:", dist_util.dev())
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.dataset == 'HCP':
        ds = HCPVolumes(args.data_dir, mode='test',
                        half_resolution=(args.image_size == 64) and not args.half_res_crop,
                        random_half_crop=(args.image_size == 64) and args.half_res_crop,
                        gen_type=None,
                        concat_coords=args.concat_coords,
                        LR=False)
    elif args.dataset == 'BTC':
        ds = BTCVolumes(args.data_dir, mode='test',
                        half_resolution=(args.image_size == 64) and not args.half_res_crop,
                        random_half_crop=(args.image_size == 64) and args.half_res_crop,
                        gen_type=None,
                        concat_coords=args.concat_coords,
                        LR=False)
    elif args.dataset == 'SUN':
        ds = SUNVolumes(args.data_dir, mode='test',
                        half_resolution=(args.image_size == 64) and not args.half_res_crop,
                        random_half_crop=(args.image_size == 64) and args.half_res_crop,
                        gen_type=None,
                        concat_coords=args.concat_coords,
                        LR=True)
    elif args.dataset == "Caffine":
        ds = CaffineVolumes(args.data_dir, mode='test',
                LR=args.LR
            )
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False)
    all_images = []
    all_labels = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量 (in Millions): {params / 1e6:.2f} M")
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")
        # model.convert_to_fp16()
    model.eval()
    for raw_img, out_dict, weak_label, label, number in iter(datal):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"reseeded (in for loop) to {seed}")
        # raw_img, out_dict, weak_label, label, number = datal_data
        img = raw_img.to(dist_util.dev())
        target = label.to(dist_util.dev())
        model_kwargs = {}
        noise = th.randn(args.batch_size, 8, 128, 128, 64).to(dist_util.dev())
        # sample_fn = diffusion.p_sample_loop
        # sample = sample_fn(model=model,
        #                    shape=noise.shape,
        #                    noise=noise,
        #                    cond=img,
        #                    clip_denoised=args.clip_denoised,
        #                    model_kwargs=model_kwargs,
        #                    )
        sample = diffusion.dpm_solver_sample(
            model,
            shape=(args.batch_size, 8, 128, 128, 64),
            segmentation_img=img,  # 你的 img
            steps=20,  # 极速采样
            order=2,
            device=dist_util.dev()
        )
        # sample_fn = diffusion.ddim_sample_loop_known
        # sample, x_noisy, org = sample_fn(
        #     model,
        #     noise.shape,
        #     img,
        #     mode='segmentation',
        #     clip_denoised=args.clip_denoised,
        #     model_kwargs=model_kwargs,
        #     sampling_steps=args.sampling_steps,
        # )
        print('done sampling')
        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=0)  # don't squeeze batch dimension for bs 1

        # Pad/Crop to original resolution
        sample = sample[:, :, :, :155]

        if len(target.shape) == 5:
            target = target.squeeze(dim=0)

        target = target[:, :, :, :155]
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(args.output_dir, args.dataset)).mkdir(parents=True, exist_ok=True)
        print("sample.shape", sample.shape)
        if args.dataset == 'HCP' or args.dataset == 'Caffine':
            name = ['fit_FWF', 'fit_NDI', 'fit_ODI', 'DI', 'F', 'MSD', 'MSK', 'uFA']

        elif args.dataset == 'GBM':
            name = ['DTI_AD', 'DTI_FA', 'DTI_RD', 'DTI_TR']

        print(number)
        for i in range(sample.shape[0]):
            path_save = os.path.join(args.output_dir, args.dataset, number[0])
            os.makedirs(path_save, exist_ok=True)
            output_name = os.path.join(path_save, f'sample_{name[i]}.nii.gz')
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(path_save, f'target_{name[i]}.nii.gz')
            # print(target_list[i].shape)
            img = nib.Nifti1Image(target.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
        # if sample.shape[0] != 1:
        #     warnings.warn('we are discarding batch>1 (should be implemented)')
        # pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # if sample.shape[1] == 1:
        #     output_name = os.path.join(args.output_dir, f'segmentation_{number[0]}.nii.gz')
        #     img = nib.Nifti1Image(sample.detach().cpu().numpy()[0, 0, ...], np.eye(4))
        #     nib.save(img=img, filename=output_name)
        # else:
        #     sample_save = sample.detach().cpu().numpy().squeeze()
        #     for k in range(sample_save.shape[0]):
        #         img = nib.Nifti1Image(sample_save[k, ...], np.eye(4))
        #         output_name = os.path.join(args.output_dir, f'segmentation_{number[0]}_{k}.nii.gz')
        #         nib.save(img=img, filename=output_name)
        # print(f'saved to {output_name}')
        #
        # dice = lambda x, y: 2*(x * y).sum()/(x**2 + y**2).sum()
        # binary_sample = sample > 0.5
        # print(f'DSC = {dice(binary_sample, label.to(sample.device)>0).item()}')


def create_argparser():
    defaults = dict(
        dataset="Caffine",
        num_channels=32,
        class_cond=False,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=False,
        use_scale_shift_norm=False,
        attention_resolutions=[],
        channel_mult=(1,1,2,3,4),
        diffusion_steps=1000,
        noise_schedule="linear",
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        dims=3,
        batch_size=1,
        num_groups=32,
        in_channels=12,
        concat_coords=False,
        out_channels=8,
        bottleneck_attention=False,
        resample_2d=False,
        renormalize=True,
        additive_skips=True,
        decoder_device_thresh=15,
        predict_xstart=True,
        FBC="U-Net",
        LR=False,
        data_dir="/path",
        data_mode="test",
        seed=0,
        image_size=128,
        half_res_crop=False,
        use_fp16=False,
        model_path='/path',
        devices=[3],
        output_dir="/path",
        num_samples=1000,
        use_ddim=False,
        num_workers=12,
        sampling_steps=0,
        clip_denoised=True
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()
