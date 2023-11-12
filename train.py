import os
import argparse
import sys
import logging

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.crat_pred import CratPred

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

log_dir = os.path.dirname(os.path.abspath(__file__))
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser = CratPred.init_args(parser)

parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a checkpoint file to resume training')


def main():
    args = parser.parse_args()

    if args.checkpoint_path:
        full_checkpoint_path = os.path.join(root_path, 'lightning_logs/version_4/checkpoints', args.checkpoint_path)
    else:
        full_checkpoint_path = None

    # Validation dataset
    dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args)
    val_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        collate_fn=collate_fn_dict,
        pin_memory=True
    )

    # Training dataset
    dataset = ArgoCSVDataset(args.train_split, args.train_split_pre, args)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=False,
        shuffle=True
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{loss_train:.2f}-{loss_val:.2f}-{ade1_val:.2f}-{fde1_val:.2f}-{ade_val:.2f}-{fde_val:.2f}",
        monitor="loss_val",
        save_top_k=-1,
    )

    # model = CratPred(args)

    if full_checkpoint_path and os.path.isfile(full_checkpoint_path):
        model = CratPred.load_from_checkpoint(full_checkpoint_path, args=args)
    else:
        model = CratPred(args)

    trainer = pl.Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
        gpus=args.gpus,
        weights_save_path=None,
        max_epochs=args.num_epochs,
        accelerator="cpu",
        resume_from_checkpoint=full_checkpoint_path if full_checkpoint_path and os.path.isfile(
            full_checkpoint_path) else None

        # device=1
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

    # Type the following command in the terminal to run training:
    # Remember to change full_checkpoint_path if you want to resume training
    # python train.py --checkpoint_path epoch=1-loss_train=51.10-loss_val=54.29-ade1_val=2.02-fde1_val=4.38-ade_val=2.02-fde_val=4.24.ckpt
