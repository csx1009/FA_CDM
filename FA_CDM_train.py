"""
Train a diffusion model on images.
"""
import os
import sys
import argparse
import torch as th
import torch.utils.tensorboard
import random

from torch.utils.data import ConcatDataset

from guided_diffusion.CaffineLoader import CaffineVolumes
from guided_diffusion.GBMLoader import GBMVolumes
from guided_diffusion.HCPloader import HCPVolumes

sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop, AutomaticWeightedLoss, FusionLossModule, get_blob_logdir
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(devices=args.devices)
    print(f"Visible Devices Count = {torch.cuda.device_count()}")
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        print(f'Using Tensorboard with logdir = {summary_writer.get_logdir()}')
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()



    logger.log("creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    print(args)
    model, diffusion = create_model_and_diffusion(
        **arguments
    )
    print("number of parameters: {:_}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
    print(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    logger.log("creating data loader...")

    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir, test_flag=False)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True)

    elif args.dataset == 'brats3d':
        assert args.image_size in [128, 256]
        ds = BRATSDataset(args.data_dir, test_flag=False,
                          normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
                          mode='train',
                          half_resolution=(args.image_size == 128) and not args.half_res_crop,
                          random_half_crop=(args.image_size == 128) and args.half_res_crop,
                          concat_coords=args.concat_coords,
                          num_classes=args.out_channels,
                          )
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
    elif args.dataset == 'brats3d':
        assert args.image_size in [128, 256]
        ds = BRATSDataset(args.data_dir, test_flag=False,
                          normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
                          mode='train',
                          half_resolution=(args.image_size == 128) and not args.half_res_crop,
                          random_half_crop=(args.image_size == 128) and args.half_res_crop,
                          concat_coords=args.concat_coords,
                          num_classes=args.out_channels,
                          )
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
    elif args.dataset == 'HCP':
        assert args.image_size in [64, 128]
        ds = HCPVolumes(args.data_dir, mode='train',
                        half_resolution=(args.image_size == 64) and not args.half_res_crop,
                        random_half_crop=(args.image_size == 64) and args.half_res_crop,
                        gen_type=None,
                        concat_coords=args.concat_coords,
                        LR=args.LR)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
    elif args.dataset == "GBM":
        ds_C = CaffineVolumes(args.data_dir, mode='train',
                LR=args.LR
            )
        print(args.data_dir)
        ds_H = HCPVolumes(args.data_dir, mode='train',
                        half_resolution=(args.image_size == 64) and not args.half_res_crop,
                        random_half_crop=(args.image_size == 64) and args.half_res_crop,
                        gen_type=None,
                        concat_coords=args.concat_coords,
                        LR=args.LR)
        ds = ConcatDataset([ds_H, ds_C])
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
    logger.log("training...")
    device = 'cuda:0'
    print(device)

    awl = AutomaticWeightedLoss(num_losses=2)
    awl.to(device)
    # if args.pretrain == True:
    #     fusion_criterion = FusionLossModule(device=device)
    #     unet1_params = list(model.input_blocks.parameters()) + \
    #                    list(model.middle_block.parameters()) + \
    #                    list(model.output_blocks.parameters()) + \
    #                    list(model.out.parameters()) + \
    #                    list(model.time_embed.parameters())  # 如果用了 time embedding
    #
    #     # 同时优化网络参数和 Loss 权重参数
    #     optimizer_stage1 = torch.optim.Adam(
    #         unet1_params + list(fusion_criterion.awl.parameters()),
    #         lr=1e-4
    #     )
    #     print(">>> Starting Stage 1: U-Net 1 Fusion Pre-training...")
    #
    #     for epoch in range(args.pretrain_epochs):
    #         model.train()
    #         epoch_loss = 0
    #
    #         for batch_idx, (raw_img, out_dict, weak_label, label, number) in enumerate(datal):
    #             # 准备数据
    #             # img1: 源图像 Stack [B, N, H, W, D]
    #             # 注意维度：你的 Loss 期望 D 在最后，请确保数据加载器输出一致
    #             # 如果 dataloader 输出是 [B, N, D, H, W]，需要在这里 permute
    #             sources = raw_img.to(device)
    #
    #             # 前向传播 (只跑 U-Net 1)
    #             # 你需要在 WNetModel 里加一个只跑 U-Net 1 的方法，或者直接调用子模块
    #             # 这里假设 WNetModel.forward_unet1 存在
    #             # t 设为 0，因为融合任务是确定性的
    #
    #
    #             # fused_img: [B, 1, H, W, D]
    #             fused_img,_ = model.forward_unet1(sources)
    #
    #             # 计算损失 (使用改进后的模块)
    #             loss, l_grad, l_ssim = fusion_criterion(fused_img, sources)
    #
    #             # 反向传播
    #             optimizer_stage1.zero_grad()
    #             loss.backward()
    #             optimizer_stage1.step()
    #
    #             epoch_loss += loss.item()
    #
    #         if epoch == 0 or (epoch + 1) % 10 == 0:
    #                 print(
    #                     f"Epoch: {epoch} : Loss={loss.item():.4f} (Grad={l_grad:.4f}, SSIM={l_ssim:.4f}) | AWL vars={fusion_criterion.awl.log_vars.data}")
    #
    #         # 保存预训练权重
    #         if epoch == args.pretrain_epochs-1:
    #             torch.save(model.state_dict(),os.path.join(get_blob_logdir(), f"wnet_stage1_ep{epoch}.pt"))
    TrainLoop(
        model=model,
        diffusion=diffusion,
        awl=awl,
        data=datal,
        device=device,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='default',
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats3d',
        use_tensorboard=True,
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=[0],
        dims=3,  # 2 for 2d images, 3 for 3d volumes
        learn_sigma=False,
        num_groups=29,
        channel_mult="1,3,4,4,4,4,4",
        in_channels=5,  # 4 MRI sequences + out_channels  (+ #dimensions if concat_coords==True)
        out_channels=1,  # out_channels = number  of classes
        bottleneck_attention=False,
        num_workers=0,
        resample_2d=False,
        mode='segmentation',
        renormalize=True,
        additive_skips=True,
        decoder_device_thresh=15,
        half_res_crop=False,
        # only relevant for image_size=128: crop to half resolution instead of subsampling, if image_size=128
        concat_coords=False,  # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        LR=True,  # 是否进行LR训练
        FBC='U-Net',  # 是否使用FBC训练
        pretrain = True, # 是否进行预训练
        pretrain_epochs=500,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
