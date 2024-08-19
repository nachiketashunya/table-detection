#!/bin/bash
#SBATCH --job-name=ctab_run
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 ##Define number of GPUs
#SBATCH --output=ctab_%j.log

echo "JOb Submitted"

module load conda/conda
source /opt/ohpc/apps/conda/bin/activate
conda activate pyten

#
work_dir="/csehome/m23csa016/MTP/CascadeTabNet/work_dirs"
config="/csehome/m23csa016/MTP/CascadeTabNet/Config"
# Loop through different percentages
for percentage in 10 15 20 25 30 40
do
    # Generate new config file path
    config_path="${config}/config_${percentage}.py"
    wdir_path="${work_dir}/train_${percentage}/"
    
    python /csehome/m23csa016/MTP/mmdetection/tools/train.py "${config_path}" --work-dir "${wdir_path}"

    echo "Training completed for ${percentage}% data."

done


