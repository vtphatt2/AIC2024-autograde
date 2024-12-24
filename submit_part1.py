from ultralytics import YOLO
import os

model_sbd_mdt = YOLO('best_part1.pt')
output_file = os.path.join('submission', 'results_part1.txt')

def submit_part1(testset_image_files):
    for image_file in testset_image_files:
        results = model_sbd_mdt(image_file)
        print(results)
    