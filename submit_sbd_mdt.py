from ultralytics import YOLO
import os

model_sbd_mdt = YOLO('best_sbd_mdt.pt')
output_file = os.path.join('submission', 'results_sbd_mdt.txt')

def submit_sbd_mdt(testset_image_files):
    for image_file in testset_image_files:
        results = model_sbd_mdt(image_file)
        print(results)
    