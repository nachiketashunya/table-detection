import random
import os
import sys
sys.path.append("/csehome/m23csa016/MTP")
from DataPreparation.generateVOC2JSON import generateVOC2Json
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from testing import inference

xmlfiles = os.listdir("/scratch/m23csa016/tabdet_data/Orig_Annotations")


# Shuffle the list to randomize the order
random.shuffle(xmlfiles)

# Calculate the split point for 30-70 ratio
split_index = int(0.7 * len(xmlfiles))

# Split into train and test sets
test_files = xmlfiles[:split_index]  # 70% for testing
train_files = xmlfiles[split_index:]   # 30% for training

save_path = "/scratch/m23csa016/tabdet_data/Annotations/automate"

generateVOC2Json(train_files, save_path, "train")
generateVOC2Json(test_files, save_path, "test")

train_annot = os.path.join(save_path, "train.json")
test_annot = os.path.join(save_path, "test.json")
data_per = [100]

# Train the model
config_path = "Config/new_config.py"
work_dir_path = "Config/new_config.py"

# load config
cfg = Config.fromfile(config_path)

# use config filename as default work_dir if cfg.work_dir is None
cfg.work_dir = work_dir_path

# build the default runner
runner = Runner.from_cfg(cfg)
runner.train()

inference(test_annot, data_per)

undetected_path = ""
undetected_xmls = os.listdir(undetected_path)

train_files.append(undetected_xmls)
