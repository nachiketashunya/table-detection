import random
import os
import sys
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
import wandb
import pandas as pd

sys.path.append("/csehome/m23csa016/MTP")
from DataPreparation.generateVOC2JSON import generateVOC2Json
from testing import inference

# Constants
XML_DIR = "/scratch/m23csa016/tabdet_data/Orig_Annotations"
SAVE_PATH = "/scratch/m23csa016/tabdet_data/Annotations/automate"
CHECKPOINT_PATH = "/scratch/m23csa016/tabdet_data/Checkpoints/Data_AL_PD_avg"
TEST_RESULTS = "/scratch/m23csa016/tabdet_data/testres/Data_AL_PD_avg"
MAX_EPOCHS = 60

"""
    Process
    1. Split the dataset into train and test
        -- Based on fraction
    2. Further Split the train dataset into 10-90 Ratio
    3. Use 10% Training data to train the model
    4. Inference on 90% Test data
    5. List down undetected and detected xmls
    6. Merge undetected xmls with train data and retrain the model
    7. Now inference on initial test data and report the metrics
    8. Repeat 1-7 for different fractions
"""

def prepare_data(xmlfiles, fraction, save_path):
    """Split the dataset and prepare training and testing files."""
    tr_split = int(fraction * len(xmlfiles))
    random.shuffle(xmlfiles)
    
    train_train = xmlfiles[:tr_split]
    train_test = xmlfiles[tr_split:]
    
    generateVOC2Json(train_test, save_path, "train_test")
    generateVOC2Json(train_train, save_path, "train_train")

    return xmlfiles, train_test, train_train

def configure_and_train(config_path, checkpoint_path, work_dir_path, run_name):
    # Initialize wandb
    wandb.init(
        project='MTP-Cluster-Avg',
        name=run_name
    )

    """Load configuration, set paths, and train the model."""
    os.makedirs(checkpoint_path, exist_ok=True)
    cfg = Config.fromfile(config_path)
    cfg.train_cfg.max_epochs = MAX_EPOCHS
    cfg.train_cfg.val_interval = MAX_EPOCHS // 10
    cfg.work_dir = work_dir_path
    cfg.default_hooks.checkpoint.out_dir = checkpoint_path

    runner = Runner.from_cfg(cfg)
    runner.train()

    wandb.finish()

def inference_and_update(train_test_annot, config_path, checkpoint_path, out_dir, train_train):
    """Run inference and update the training set with undetected XMLs."""
    os.makedirs(out_dir, exist_ok=True)
    inference(train_test_annot, config_file=config_path, 
              checkpoint_file=os.path.join(checkpoint_path, f"train_auto_pd_avg/epoch_{MAX_EPOCHS}.pth"), 
              out_dir=out_dir)

    undetected_path = f"{out_dir}/undet_xmls.txt"
    with open(undetected_path, 'r') as file:
        undetected_xmls = [line.strip() for line in file]

    train_train.extend(undetected_xmls)
    return undetected_xmls

def retrain_with_undetected(fraction, train_train, save_path):
    """Retrain the model with updated training data."""
    generateVOC2Json(train_train, save_path, "train")

    work_dir_path = "work_dirs/train_auto_pd_avg"
    config_path = "Config/config_auto.py"
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f"{fraction * 100}/CLUSTER_AVG")
    out_dir = os.path.join(TEST_RESULTS, f'{fraction * 100}/CLUSTER_AVG')

    # Original XML files
    xmlfiles = os.listdir(XML_DIR)

    # Store information about dataset size
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'datainfo.txt'), 'w') as f:
        f.write(f"Total XML Files: {len(xmlfiles)}\n")
        f.write(f"Total Training Files: {len(train_train)}\n")
        f.write(f"Fraction: {len(train_train)/len(xmlfiles) * 100:.2f}%\n")

    run_name = f"{fraction*100}_CLUSTER_AVG"
    configure_and_train(config_path, checkpoint_path, work_dir_path, run_name)

# Create cluster based Undetected XML files
def create_clustered_xml(dir):
    # Define ranges for confidence scores and percentages
    ranges = {
        (0, 40): 1.00,
        (40, 50): 0.90,
        (50, 60): 0.80,
        (60, 70): 0.70,
        (70, 80): 0.60,
        (80, 90): 0.50,
        (90, 100): 0.40
    }

    # Step 1: Read the text file
    data = []
    input_file = os.path.join(dir, "undetected.txt")
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            confidence_score = float(parts[1])
            data.append([confidence_score, parts[2]])

    # Convert data to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=['Confidence', 'File'])

    # Step 2: Group and sample data
    sampled_data = []
    representation = {}

    for (lower, upper), percentage in ranges.items():
        # Filter rows within the current range
        df_range = df[(df['Confidence'] >= lower/100) & (df['Confidence'] < upper/100)]
        
        # Sample the required percentage of rows from this range
        sample_size = int(np.floor(len(df_range) * percentage))
        sampled = df_range.sample(n=sample_size, random_state=42)  # Random state for reproducibility
        
        # Append the sampled data
        sampled_data.append(sampled)

        # Note the no of files
        representation[f"{lower}-{upper}"] = len(sampled)

    # Combine all sampled data into one DataFrame
    final_sampled_df = pd.concat(sampled_data)

    # Step 3: Write the sampled data into a new text file
    output_file = os.path.join(dir, 'clus_undet_xmls.txt')
    ret_xmls = []
    with open(output_file, 'w') as file:
        for _, row in final_sampled_df.iterrows():
            xml_file = row['File'].replace('.jpg', '.xml')
            file.write(f"{xml_file}\n")

            ret_xmls.append(xml_file)

    print("Clustering and sampling complete.\n")
    print("Number of files in each range: \n")
    
    for key, values in representation.items():
        print(f"{key}: {values} Files")
    
    return ret_xmls
    
def main():
    # Log into wandb
    wandb.login(key="82fadbf5b2810c5fdaee488a728eabb8f084b7a3")

    # List original XML files
    xmlfiles = os.listdir(XML_DIR)
    random.shuffle(xmlfiles)

    split_index = int(0.1 * len(xmlfiles))
    test_files = xmlfiles[:split_index]
    train_files = xmlfiles[split_index:]

    generateVOC2Json(test_files, SAVE_PATH, "test")

    for fraction in np.round(np.arange(0.10, 0.45, 0.05), 2):
        # Prepare dataset
        train_files, train_test, train_train = prepare_data(train_files, fraction, SAVE_PATH)
        
        # Create Clusters for undetected XMLs
        dir = os.path.join(TEST_RESULTS, f'{fraction*100}/TrainSet')
        undetected_xmls = create_clustered_xml(dir)

        train_train.extend(undetected_xmls)

        # Retrain with updated dataset
        retrain_with_undetected(fraction, train_train, SAVE_PATH)

if __name__ == "__main__":
    main()
