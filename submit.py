from submit_sbd_mdt import submit_sbd_mdt
from submit_part1 import submit_part1
from submit_part2 import submit_part2
from submit_part3 import submit_part3
from glob import glob 
import os


testset_image_files = glob(os.path.join('testset1', 'images', '*.jpg'))
testset_image_files.sort()


os.makedirs('submission', exist_ok=True)
# submit_sbd_mdt(testset_image_files)
submit_part1(testset_image_files)
# submit_part2(testset_image_files)
# submit_part3(testset_image_files)


input_files = [
    os.path.join("submission", "results_sbd_mdt.txt"),
    os.path.join("submission", "results_part1.txt"),
    os.path.join("submission", "results_part2.txt"),
    os.path.join("submission", "results_part3.txt"),
]
output_file = os.path.join("submission", "results.txt")

data = {}

for file in input_files:
    with open(file, "r") as f:
        for line in f:
            img_name, values = line.strip().split(" ", 1)

            if img_name not in data:
                data[img_name] = [values] 
            else:
                data[img_name].append(values) 

with open(output_file, "w") as f:
    for img_name, values in data.items():
        merged_line = f"{img_name} {' '.join(values)}\n"
        f.write(merged_line)