import os
import os.path as osp
import torch
import numpy as np
import random
import datetime
import logging
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from ranker.ranker_dataset import RankerDataset
from ranker.pointnet import PointNetRanker
from opencood.utils import multi_gpu_utils


def parse_config():
    parser = argparse.ArgumentParser(description="Train the PointNet-based box ranker")

    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the V2V4Real dataset")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Directory name (relative to root_dir) containing ranker training data")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Directory name (relative to root_dir) containing ranker val data")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_annotate_frames", type=int, default=2,
                        help="Number of annotated frames used per scenario (must match data generation)")

    parser.add_argument("--use_ground", action="store_true",
                        help="Include ground point mask as ranker input feature")
    parser.add_argument("--use_offset", action="store_true",
                        help="Train ranker to also predict box offset correction")
    parser.add_argument("--random_drop_points", action="store_true",
                        help="Augment by randomly dropping points during training")

    parser.add_argument("--save_dir", type=str, default=None,
                        help="Experiment output directory (default: timestamped)")

    parser.add_argument("--no_dist", action="store_true",
                        help="Disable distributed training")
    parser.add_argument("--dist_url", default="env://",
                        help="URL used to set up distributed training")

    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name. If not set, only local logging is used.")
    parser.add_argument("--wandb_note", type=str, default="")

    args = parser.parse_args()

    random.seed(310)
    np.random.seed(310)
    torch.manual_seed(310)

    return args


def inplace_relu(m):
    if m.__class__.__name__.find("ReLU") != -1:
        m.inplace = True


def main():
    args = parse_config()

    if args.save_dir is not None:
        save_dir = osp.join("./results", args.save_dir)
    else:
        save_dir = osp.join("./results", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(osp.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(osp.join(save_dir, "logs"), exist_ok=True)

    logger = logging.getLogger("ranker")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(osp.join(save_dir, "logs", "train.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(s):
        logger.info(s)
        print(s)

    log_string(f"Args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu_utils.init_distributed_mode(args)

    if args.rank == 0 and args.wandb_project is not None:
        import wandb
        wandb.login()

    log_string("Loading dataset...")
    train_set = RankerDataset(
        root_dir=args.root_dir,
        data_dir=args.train_data_dir,
        train_split="subset2",
        data_split="train",
        car_id="0",
        num_points=1000,
        num_annotate_frames=args.num_annotate_frames,
        use_ground=args.use_ground,
        random_drop_points=args.random_drop_points,
    )

    if args.distributed:
        train_sampler = DistributedSampler(train_set)
        train_loader = DataLoader(
            train_set, sampler=train_sampler,
            num_workers=16, batch_size=args.batch_size,
            pin_memory=False, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size,
            shuffle=True, num_workers=16, drop_last=True,
        )

    num_steps = len(train_loader)

    val_loader = None
    if args.val_data_dir is not None:
        val_set = RankerDataset(
            root_dir=args.root_dir,
            data_dir=args.val_data_dir,
            data_split="train",
        )
        if args.distributed:
            val_sampler = DistributedSampler(val_set, shuffle=False)
            val_loader = DataLoader(
                val_set, sampler=val_sampler,
                num_workers=16, batch_size=args.batch_size,
                pin_memory=False, drop_last=True,
            )
        else:
            val_loader = DataLoader(
                val_set, batch_size=1,
                shuffle=False, num_workers=16, drop_last=True,
            )

    ranker = PointNetRanker(
        use_lwh=True,
        use_angle=True,
        use_depth=True,
        use_ground=args.use_ground,
        use_dropout=False,
        use_bn=False,
        feature_transform=False,
        use_offset=args.use_offset,
    )
    ranker.apply(inplace_relu)
    ranker.to(device)
    ranker_without_ddp = ranker

    if args.distributed:
        ranker = torch.nn.parallel.DistributedDataParallel(
            ranker, device_ids=[args.gpu], find_unused_parameters=True)
        ranker_without_ddp = ranker.module

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        ranker_without_ddp.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    if args.rank == 0 and args.wandb_project is not None:
        import wandb
        wandb.init(
            project=args.wandb_project,
            notes=args.wandb_note,
            config=vars(args),
        )
        args = wandb.config

    log_string("Starting training...")
    for epoch in range(args.epoch):
        log_string(f"Epoch {epoch+1}/{args.epoch}")
        mean_err_total = []
        mean_err_score = []
        mean_err_offset = []

        ranker.train()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for step, batch_dict in tqdm(enumerate(train_loader), total=num_steps, smoothing=0.9):
            optimizer.zero_grad()

            ground = batch_dict['ground'].to(device) if args.use_ground else None
            bbox_center = batch_dict['bbox_center'].to(device)
            ptc = batch_dict['ptc'].to(device)
            distance = batch_dict['distance'].to(device)
            iou = batch_dict['iou'].to(device)
            offset = batch_dict['offset'].to(device)

            if args.use_offset:
                pred_score, pred_offset, _, _ = ranker(bbox_center, ptc, ground, distance)
                loss_offset = criterion(pred_offset, offset)
            else:
                pred_score, _, _ = ranker(bbox_center, ptc, ground, distance)
                loss_offset = torch.tensor(0.0)

            pred_score = pred_score.squeeze(-1)
            loss_score = criterion(pred_score, iou)
            loss_total = 5 * loss_score + loss_offset

            mean_err_score.append(loss_score.item())
            mean_err_offset.append(loss_offset.item())
            mean_err_total.append(loss_total.item())

            loss_total.backward()
            optimizer.step()

            if (step + 1 < num_steps) and args.rank == 0 and args.wandb_project is not None:
                import wandb
                wandb.log({
                    "train/loss_total": loss_total,
                    "train/loss_score": loss_score,
                    "train/loss_offset": loss_offset,
                    "train/epoch": (step + 1 + num_steps * epoch) / num_steps,
                })

        scheduler.step()

        torch.save(
            ranker_without_ddp.state_dict(),
            osp.join(save_dir, "checkpoints", f"epoch_{epoch+1:04d}.pth"),
        )

        log_string(
            f"Loss: {np.mean(mean_err_total):.6f} | "
            f"Score: {np.mean(mean_err_score):.6f} | "
            f"Offset: {np.mean(mean_err_offset):.6f}"
        )

        if val_loader is not None and (epoch + 1) % 5 == 0:
            mean_val_err = []
            ranker.eval()
            for _, batch_dict in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                ground = batch_dict['ground'].to(device) if args.use_ground else None
                bbox_center = batch_dict['bbox_center'].to(device)
                ptc = batch_dict['ptc'].to(device)
                distance = batch_dict['distance'].to(device)
                iou = batch_dict['iou'].to(device)

                with torch.no_grad():
                    pred, _, _ = ranker(bbox_center, ptc, ground, distance)
                pred = pred.squeeze(-1)
                loss = criterion(pred, iou)
                mean_val_err.append(loss.item())

            val_loss = np.mean(mean_val_err)
            log_string(f"Val loss: {val_loss:.6f}")
            if args.rank == 0 and args.wandb_project is not None:
                import wandb
                wandb.log({"val/val_loss": val_loss})
            ranker.train()

    log_string("Training complete.")
    if args.rank == 0 and args.wandb_project is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
