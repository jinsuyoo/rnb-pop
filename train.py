import argparse
import os
import statistics
import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler

from opencood.utils import train_utils, multi_gpu_utils, yaml_utils
from opencood.data_utils.datasets.v2v4real_dataset import V2V4RealDataset


def train_parser():
    parser = argparse.ArgumentParser(description="Train 3D object detector for ego car")

    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help="Path to the config yaml file")
    parser.add_argument("--model_dir", type=str, default=None)

    parser.add_argument("--no_dist", action="store_true",
                        help="Disable distributed training")
    parser.add_argument("--dist_url", default="env://",
                        help="URL used to set up distributed training")

    parser.add_argument("--save_path", type=str, default=None,
                        help="Directory to save model checkpoints")

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_train_scenario", type=int, default=None)
    parser.add_argument("--stage", type=int, default=1,
                        help="Training stage (1=first round, 2=second round self-training)")

    parser.add_argument("--ego_car_id", type=str, default='0')

    # Pseudo-label loading
    parser.add_argument("--load_npy_label", action="store_true",
                        help="Load pseudo-labels from .npy files instead of GT")
    parser.add_argument("--npy_label_path", type=str, default=None,
                        help="Directory containing .npy pseudo-label files")
    parser.add_argument("--npy_label_idx", type=str, default=None,
                        help="Filename suffix for the label array in .npy files")
    parser.add_argument("--npy_label_order", type=str, default="lwh")

    # Training params
    parser.add_argument("--n_epoches", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)

    parser.add_argument("--min_timestamp", type=int, default=0)
    parser.add_argument("--max_timestamp", type=int, default=9999)
    parser.add_argument("--num_train_data", type=int, default=9999)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosineannealwarm")
    parser.add_argument("--warmup_lr", type=float, default=2e-4)
    parser.add_argument("--warmup_epoches", type=int, default=6)
    parser.add_argument("--lr_min", type=float, default=2e-5)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name. If not set, only TensorBoard is used.")
    parser.add_argument("--wandb_note", type=str, default="")

    parser.add_argument("--pretrained_model_path", type=str, default="", nargs='?',
                        help="Path to a pretrained model to initialize from")

    parser.add_argument("--num_lidar_beams", type=int, default=32, choices=[8, 16, 32])

    # Distance-based curriculum (Sec. 3.4)
    parser.add_argument("--distance_filtering", action="store_true",
                        help="Enable distance-based curriculum filtering")
    parser.add_argument("--min_distance", type=float, default=0.)
    parser.add_argument("--max_distance", type=float, default=90.)

    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    hypes["load_npy_label"] = opt.load_npy_label
    hypes["npy_label_path"] = opt.npy_label_path
    hypes["npy_label_idx"] = opt.npy_label_idx
    hypes["num_train_scenario"] = opt.num_train_scenario
    hypes["cur_stage"] = opt.stage

    hypes["optimizer"] = {}
    hypes["optimizer"]["core_method"] = opt.optimizer
    hypes["optimizer"]["lr"] = opt.lr
    hypes["optimizer"]["args"] = {}
    hypes["optimizer"]["args"]["eps"] = opt.eps
    hypes["optimizer"]["args"]["weight_decay"] = opt.weight_decay

    hypes["lr_scheduler"] = {}
    hypes["lr_scheduler"]["epoches"] = opt.n_epoches
    hypes["lr_scheduler"]["core_method"] = opt.lr_scheduler
    hypes["lr_scheduler"]["warmup_lr"] = opt.warmup_lr
    hypes["lr_scheduler"]["warmup_epoches"] = opt.warmup_epoches
    hypes["lr_scheduler"]["lr_min"] = opt.lr_min

    if hypes["model"]["core_method"] == "second":
        hypes["model"]["args"]["batch_size"] = opt.batch_size
    if hypes["model"]["core_method"] == "voxel_net":
        hypes["model"]["args"]["batch_size"] = opt.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpu_utils.init_distributed_mode(opt)

    if opt.rank == 0 and opt.wandb_project is not None:
        import wandb
        wandb.login()

    # Build dataloaders
    print('Building dataloader..')
    train_set = V2V4RealDataset(
        params=hypes, data_split='train',
        ego_car_id=opt.ego_car_id,
        distance_filtering=opt.distance_filtering,
        num_lidar_beams=opt.num_lidar_beams,
        min_distance=opt.min_distance,
        max_distance=opt.max_distance,
        min_timestamp=opt.min_timestamp,
        max_timestamp=opt.max_timestamp,
    )

    val_set = V2V4RealDataset(
        params=hypes, data_split='val',
        ego_car_id=opt.ego_car_id,
        num_lidar_beams=opt.num_lidar_beams,
        no_augment=True,
    )

    if opt.distributed:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        train_batch_sampler = BatchSampler(train_sampler, opt.batch_size, drop_last=False)
        val_batch_sampler = BatchSampler(val_sampler, opt.batch_size, drop_last=True)

        train_loader = DataLoader(
            train_set,
            batch_sampler=train_batch_sampler,
            num_workers=opt.num_workers,
            collate_fn=train_set.collate_batch_train,
        )
        val_loader = DataLoader(
            val_set,
            batch_sampler=val_batch_sampler,
            num_workers=opt.num_workers,
            collate_fn=val_set.collate_batch_train,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            collate_fn=train_set.collate_batch_train,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            collate_fn=val_set.collate_batch_train,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
        )

    # Create model
    print('Creating model..')
    model = train_utils.create_model(hypes)

    if opt.pretrained_model_path:
        print(f'Loading pretrained model from: {opt.pretrained_model_path}')
        model.load_state_dict(torch.load(opt.pretrained_model_path), strict=False)

    init_epoch = 0
    saved_path = train_utils.create_log_folder(opt.save_path, hypes, use_time=False)
    saved_path = os.path.join(saved_path, f"stage_{opt.stage}")
    os.makedirs(saved_path, exist_ok=True)
    print(f"Experiment will be saved to: {saved_path}")

    model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    print("Creating loss function..")
    criterion = train_utils.create_loss(hypes)

    print("Setting up optimizer..")
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    writer = SummaryWriter(saved_path)

    if opt.rank == 0 and opt.wandb_project is not None:
        import wandb
        wandb.init(
            project=opt.wandb_project,
            notes=opt.wandb_note,
            config=hypes,
        )
        hypes = wandb.config

    n_epoches = opt.n_epoches
    print(f"Total epochs: {n_epoches:d}, steps per epoch: {num_steps:d}")

    for epoch in range(init_epoch, max(n_epoches, init_epoch)):
        if opt.debug and epoch == 1:
            break

        if hypes["lr_scheduler"]["core_method"] != "cosineannealwarm":
            scheduler.step(epoch)
        if hypes["lr_scheduler"]["core_method"] == "cosineannealwarm":
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print("Current learning rate: %.7f" % param_group["lr"])

        if opt.distributed:
            train_sampler.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for step, batch_data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)
            label_dict = batch_data["ego"]["label_dict"]
            label_dict = train_utils.to_device(label_dict, device)

            ouput_dict = model(batch_data["ego"])
            final_loss = criterion(ouput_dict, label_dict, is_truck=False)
            final_loss.backward()
            optimizer.step()

            print(final_loss.item())
            if (step + 1 < num_steps) and opt.rank == 0:
                if opt.wandb_project is not None:
                    import wandb
                    wandb.log({
                        "train_loss": final_loss,
                        "epoch": (step + 1 + (num_steps * epoch)) / num_steps,
                    })
                criterion.logging(epoch, step, len(train_loader), writer, pbar=pbar2)
                pbar2.update(1)

            if hypes["lr_scheduler"]["core_method"] == "cosineannealwarm":
                scheduler.step_update(epoch * num_steps + step)

        if (epoch + 1) % opt.save_freq == 0:
            torch.save(
                model_without_ddp.state_dict(),
                os.path.join(saved_path, f"net_epoch{epoch+1:03d}.pth"),
            )

        if (epoch + 1) % opt.eval_freq == 0:
            model.eval()
            valid_ave_loss = []
            with torch.no_grad():
                for _, batch_data in enumerate(val_loader):
                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data["ego"])
                    final_loss = criterion(ouput_dict, batch_data["ego"]["label_dict"], is_truck=False)
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print(f"At epoch {epoch:d}, the validation loss is {valid_ave_loss:.7f}")
            writer.add_scalar("Validate_Loss", valid_ave_loss, epoch)
            if opt.rank == 0 and opt.wandb_project is not None:
                import wandb
                wandb.log({"val_loss": valid_ave_loss})

    print(f"Training finished. Checkpoints saved to {saved_path}")

    if opt.rank == 0 and opt.wandb_project is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
