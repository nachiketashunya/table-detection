from mmengine.config import Config
from mmengine.runner import Runner
import wandb

#
MAX_EPOCHS = 100
work_dir="/csehome/m23csa016/MTP/CascadeTabNet/work_dirs"
config="/csehome/m23csa016/MTP/CascadeTabNet/Config"

# Generate new config file path
percents = [100]

# wandb.login(key="82fadbf5b2810c5fdaee488a728eabb8f084b7a3")

for p in percents:
    config_path=f"{config}/config_{p}.py"
    wdir_path=f"{work_dir}/train_{p}/"
    run_name=f"{p}_training"

    # Initialize wandb
    # wandb.init()

    """Load configuration, set paths, and train the model."""
    cfg = Config.fromfile(config_path)
    cfg.train_cfg.max_epochs = MAX_EPOCHS
    cfg.train_cfg.val_interval = MAX_EPOCHS // 10
    cfg.work_dir = wdir_path

    runner = Runner.from_cfg(cfg)
    runner.train()

    wandb.finish()