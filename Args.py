import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

class Args:
    def __init__(self):
        # Dataset arguments
        self.train_split = os.path.join(root_path, "dataset", "argoverse", "train", "data")
        self.val_split = os.path.join(root_path, "dataset", "argoverse", "val", "data")
        self.test_split = os.path.join(root_path, "dataset", "argoverse", "test_obs", "data")
        self.train_split_pre = os.path.join(root_path, "dataset", "argoverse", "train_pre.pkl")
        self.val_split_pre = os.path.join(root_path, "dataset", "argoverse", "val_pre.pkl")
        self.test_split_pre = os.path.join(root_path, "dataset", "argoverse", "test_pre.pkl")
        self.reduce_dataset_size = 0
        self.use_preprocessed = False
        self.align_image_with_target_x = True

        # Training arguments
        self.num_epochs = 72
        self.lr_values = [1e-3, 1e-4, 1e-3, 1e-4]
        self.lr_step_epochs = [32, 36, 68]
        self.wd = 0.01
        self.batch_size = 32
        self.val_batch_size = 32
        self.workers = 8
        self.val_workers = 8
        self.gpus = 0

        # Model arguments
        self.latent_size = 128
        self.num_preds = 30
        self.mod_steps = [1, 5]
        self.mod_freeze_epoch = 36

        # Other arguments
        self.split = "val"
        self.ckpt_path = "lightning_logs/version_7/checkpoints/epoch=71-loss_train=40.19-loss_val=14.66-ade1_val=1.42-fde1_val=3.12-ade_val=0.85-fde_val=1.45.ckpt"