import argparse
import datetime
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")), "deit"))
from deit.main import (
    DistillationLoss,
    RASampler,
    build_dataset,
    evaluate,
    get_args_parser,
    new_data_aug_generator,
    train_one_epoch,
    utils,
)

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")), "catalyst"))
from catalyst.data.sampler import BalanceClassSampler, DistributedSamplerWrapper

from dinov3.checkpointer import init_model_from_checkpoint_for_evals
from dinov3.configs import DinoV3SetupArgs, get_cfg_from_args
from dinov3.models import vision_transformer as vits

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict


def get_cls_num_list(labels):
    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1
    labels = list(counter.keys())
    labels.sort()
    cls_num_list = [counter[label] for label in labels]
    return cls_num_list


class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        loss = torch.sum(-target * F.log_softmax(logit_adjusted, dim=-1), dim=-1)
        return loss.mean()


def build_model(args, img_size=224):
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,
        pos_embed_rope_base=args.pos_embed_rope_base,
        pos_embed_rope_min_period=args.pos_embed_rope_min_period,
        pos_embed_rope_max_period=args.pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
        qkv_bias=args.qkv_bias,
        layerscale_init=args.layerscale,
        norm_layer=args.norm_layer,
        ffn_layer=args.ffn_layer,
        ffn_bias=args.ffn_bias,
        proj_bias=args.proj_bias,
        n_storage_tokens=args.n_storage_tokens,
        mask_k_bias=args.mask_k_bias,
        untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
        untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
    )
    model = vits.__dict__[args.arch](**vit_kwargs)
    return model, model.embed_dim


def build_model_from_cfg(cfg):
    return build_model(cfg.student, img_size=cfg.crops.global_crops_size)


def create_model(
    config_file,
    num_classes,
    img_size=224,
    drop_path_rate=0.0,
    **kwargs,
):
    if args.finetune:
        backbone_weights_path, classifier_weights_path = args.finetune.split(";")
    else:
        backbone_weights_path = None
        classifier_weights_path = None

    setup_args = DinoV3SetupArgs(
        config_file=config_file,
        pretrained_weights=backbone_weights_path,
    )
    config = get_cfg_from_args(setup_args, strict=False)
    config.crops.global_crops_size = img_size
    config.student.drop_path_rate = drop_path_rate
    model, embed_dim = build_model_from_cfg(config)
    if backbone_weights_path is not None:
        init_model_from_checkpoint_for_evals(model, setup_args.pretrained_weights, "teacher")

    model.head = nn.Linear(embed_dim, num_classes)
    if classifier_weights_path is not None:
        print(f"Loading classifier weights from {classifier_weights_path}")
        classifier_weights = torch.load(classifier_weights_path, map_location="cpu", weights_only=True)
        model.head.load_state_dict(classifier_weights, strict=True)

    return model


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    #random.seed(seed)

    cudnn.benchmark = True

    dataset_train, _ = build_dataset(is_train=True, args=args)
    args.nb_classes = len(dataset_train.classes)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        elif args.balanced_sampler:
            sampler_train = DistributedSamplerWrapper(
                BalanceClassSampler(labels=dataset_train.targets, mode=args.balanced_sampler_mode),
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        if args.balanced_sampler:
            sampler_train = BalanceClassSampler(labels=dataset_train.targets, mode=args.balanced_sampler_mode)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.config_file,
        #args.model,
        #pretrained=False,
        num_classes=args.nb_classes,
        #drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        #drop_block_rate=None,
        img_size=args.input_size,
    )

    #if args.finetune:
    #    if args.finetune.startswith('https'):
    #        checkpoint = torch.hub.load_state_dict_from_url(
    #            args.finetune, map_location='cpu', check_hash=True)
    #    else:
    #        checkpoint = torch.load(args.finetune, map_location='cpu')

    #    checkpoint_model = checkpoint['model']
    #    state_dict = model.state_dict()
    #    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
    #        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #            print(f"Removing key {k} from pretrained checkpoint")
    #            del checkpoint_model[k]

    #    # interpolate position embedding
    #    pos_embed_checkpoint = checkpoint_model['pos_embed']
    #    embedding_size = pos_embed_checkpoint.shape[-1]
    #    num_patches = model.patch_embed.num_patches
    #    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    #    # height (== width) for the checkpoint position embedding
    #    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    #    # height (== width) for the new position embedding
    #    new_size = int(num_patches ** 0.5)
    #    # class_token and dist_token are kept unchanged
    #    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    #    # only the position tokens are interpolated
    #    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    #    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    #    pos_tokens = torch.nn.functional.interpolate(
    #        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    #    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    #    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    #    checkpoint_model['pos_embed'] = new_pos_embed

    #    model.load_state_dict(checkpoint_model, strict=False)

    if args.attn_only:
        for name_p, p in model.named_parameters():
            if ".attn." in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print("no position encoding")
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print("no patch embed")

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        if args.logit_adjusted_loss:
            cls_num_list = get_cls_num_list(dataset_train.targets)
            cls_num_list = torch.Tensor(cls_num_list).to(device)
            criterion = LogitAdjustedLoss(cls_num_list) 
        else: 
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.logit_adjusted_loss:
            raise ValueError("Logit adjusted loss is not supported with label smoothing")
        else:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    teacher_model = None
    #if args.distillation_type != "none":
    #    assert args.teacher_path, "need to specify teacher-path when using distillation"
    #    print(f"Creating teacher model: {args.teacher_model}")
    #    teacher_model = create_model(
    #        args.teacher_model,
    #        pretrained=False,
    #        num_classes=args.nb_classes,
    #        global_pool="avg",
    #    )
    #    if args.teacher_path.startswith("https"):
    #        checkpoint = torch.hub.load_state_dict_from_url(
    #            args.teacher_path, map_location="cpu", check_hash=True
    #        )
    #    else:
    #        checkpoint = torch.load(args.teacher_path, map_location="cpu")
    #    teacher_model.load_state_dict(checkpoint["model"])
    #    teacher_model.to(device)
    #    teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion,
        teacher_model,
        args.distillation_type,
        args.distillation_alpha,
        args.distillation_tau,
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args=args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": get_state_dict(model_ema),
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(data_loader_val, model, device, chunk_size=args.grad_accum_steps)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / "best_checkpoint.pth"]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "model_ema": get_state_dict(model_ema),
                            "scaler": loss_scaler.state_dict(),
                            "args": args,
                        },
                        checkpoint_path,
                    )

        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", parents=[get_args_parser()]
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use a balanced sampler for training data",
    )
    parser.add_argument(
        "--balanced-sampler-mode",
        type=lambda x: int(x) if x.isdigit() else x,
        default="downsampling",
        help="Balanced sampler mode. Can be 'downsampling', 'upsampling' or an integer value",
    )
    parser.add_argument(
        "--logit-adjusted-loss",
        action="store_true",
        help="Use logit adjusted loss",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        metavar="N",
        help="The number of steps to accumulate gradients (default: 1)",
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
