from glob import glob
from ultralytics import YOLO
import os


model_for_sbd_mdt = YOLO('best_sbd_mdt.pt')
# model_for_part_1 = YOLO('best_part_2.pt')
# model_for_part_2 = YOLO('best_part_3.pt')
# model_for_part_3 = YOLO('best_part_4.pt')


test_image_files = glob(os.path.join('testset1', 'images', '*.jpg'))

result_path = os.path.join('submission', 'result.txt')


def process_and_write_to_file(results):



# for image_path in test_image_files:
#     results = model_for_sbd_mdt (image_path)
#     process_and_write_to_file(results)