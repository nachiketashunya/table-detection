import json
import os
from mmdet.apis import DetInferencer

# Paths
# Specify the path to model config and checkpoint file
data_root = "/scratch/m23csa016/tabdet_data"
config_path = 'CascadeTabNet/Config'
checkpoint_path = 'CascadeTabNet/Checkpoints'

image_folder = os.path.join(data_root, "Orig_Image")  # Single folder for all images
output_dir = 'testres'

def inference(annot_file, by_percentage=False, data_per=[100], config_file=None, checkpoint_file=None, out_dir=None):
    # Set the paths for the files once
    undetected_file = f"{out_dir}/undetected.txt"
    undet_xmls_file = f"{out_dir}/undet_xmls.txt"
    detected_file = f"{out_dir}/detected.txt"
    det_xmls_file = f"{out_dir}/det_xmls.txt"

    # Ensure that files are empty before the loop starts
    files_to_clear = [undetected_file, undet_xmls_file, detected_file, det_xmls_file]

    # Check each file, create if it doesn't exist, and clear its content
    for file in files_to_clear:
        with open(file, 'w') as f:
            pass  # Opening in 'w' mode clears the file content

    for p in data_per:
        if by_percentage:
            config_file = os.path.join(config_path, f"config_{p}.py")
            checkpoint_file = os.path.join(checkpoint_path, f"Data_{p}/train_{p}/epoch_200.pth")

            out_dir = os.path.join(output_dir, f'data_{p}/end2end')
            os.makedirs(out_dir, exist_ok=True)
        

        pred_dir = os.path.join(out_dir, "preds")
        os.makedirs(pred_dir, exist_ok=True)

        # Initialize the DetInferencer
        inferencer = DetInferencer(model=config_file, weights=checkpoint_file)
        # Load test annotations
        with open(annot_file) as f:
            test_annotations = json.load(f)

        # Get test image filenames
        test_image_filenames = [img['file_name'] for img in test_annotations['images']]

        # Run inference and save results
        for img_filename in test_image_filenames:
            img_path = os.path.join(image_folder, img_filename)
            
            # Perform inference
            result = inferencer(img_path, out_dir=out_dir)

            # Process the result to the desired format
            output = []
            # Extracting labels, scores, and bboxes
            if len(result['predictions'][0]['labels']) != 0:
                label = result['predictions'][0]['labels'][0]
            else:
                label = -1
            
            if len(result['predictions'][0]['scores']) != 0:
                score = result['predictions'][0]['scores'][0]           
            else:
                score = 0

            if len(result['predictions'][0]['bboxes']) != 0:
                bbox = result['predictions'][0]['bboxes'][0]
            else:
                bbox = [-1, -1, -1, -1]

            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]

            output.append(f"{label} {score:.4f} {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f}")

            # Keep track of correctly detected tables
            if score < 0.95:
                with open(undetected_file, 'a+') as f:
                    f.write(f"-1 {score:.4f} {img_filename}\n")

                with open(undet_xmls_file, 'a+') as f:
                    f.write(f"{os.path.splitext(img_filename)[0]}.xml\n")
            
            else:
                with open(detected_file, 'a+') as f:
                    f.write(f"0 {score:.4f} {img_filename}\n")
                
                with open(det_xmls_file, 'a+') as f:
                    f.write(f"{os.path.splitext(img_filename)[0]}.xml\n")
            
            # Save the results to a file
            with open(f"{pred_dir}/{img_filename}.txt", 'w') as f:
                f.write("\n".join(output))
