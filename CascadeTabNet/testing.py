import json
import os
from mmdet.apis import DetInferencer

# Constants
DATA_ROOT = "/scratch/m23csa016/tabdet_data"
CONFIG_PATH = 'CascadeTabNet/Config'
CHECKPOINT_PATH = 'CascadeTabNet/Checkpoints'
IMAGE_FOLDER = os.path.join(DATA_ROOT, "Dilated")
OUTPUT_DIR = 'testres'

def create_or_clear_files(out_dir):
    files = [
        f"{out_dir}/undetected.txt",
        f"{out_dir}/undet_xmls.txt",
        f"{out_dir}/detected.txt",
        f"{out_dir}/det_xmls.txt"
    ]
    for file in files:
        with open(file, 'w'):
            pass
    return files

def process_result(result):
    output = []
    result_str = ""
    total_score = 0
    predictions = result['predictions'][0]
    
    for label, score, bbox in zip(predictions['labels'], predictions['scores'], predictions['bboxes']):
        x_min, y_min, x_max, y_max = bbox
        result_str += f"{label} {score:.4f} {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f} "
        total_score += score

    avg_score = total_score / len(predictions['labels']) if predictions['labels'] else 0
    output.append(result_str)
    
    return output, avg_score

def write_to_file(file_path, content):
    with open(file_path, 'a') as f:
        f.write(content)

def inference(annot_file, by_percentage=False, data_per=[100], config_file=None, checkpoint_file=None, out_dir=None):
    for p in data_per:
        if by_percentage:
            config_file = os.path.join(CONFIG_PATH, f"config_{p}.py")
            checkpoint_file = os.path.join(CHECKPOINT_PATH, f"Data_{p}/train_{p}/epoch_200.pth")
            out_dir = os.path.join(OUTPUT_DIR, f'data_{p}/end2end')
        
        os.makedirs(out_dir, exist_ok=True)
        pred_dir = os.path.join(out_dir, "preds")
        os.makedirs(pred_dir, exist_ok=True)

        undetected_file, undet_xmls_file, detected_file, det_xmls_file = create_or_clear_files(out_dir)

        inferencer = DetInferencer(model=config_file, weights=checkpoint_file)
        
        with open(annot_file) as f:
            test_annotations = json.load(f)

        for img in test_annotations['images']:
            img_path = os.path.join(IMAGE_FOLDER, img['file_name'])
            result = inferencer(img_path, out_dir=out_dir)

            output, avg_score = process_result(result)

            if avg_score < 0.95:
                write_to_file(undetected_file, f"-1 {avg_score:.4f} {img['file_name']}\n")
                write_to_file(undet_xmls_file, f"{os.path.splitext(img['file_name'])[0]}.xml\n")
            else:
                write_to_file(detected_file, f"0 {avg_score:.4f} {img['file_name']}\n")
                write_to_file(det_xmls_file, f"{os.path.splitext(img['file_name'])[0]}.xml\n")

            with open(f"{pred_dir}/{img['file_name']}.txt", 'w') as f:
                f.write("\n".join(output))
