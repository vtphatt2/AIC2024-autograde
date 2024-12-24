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
# submit_part1(testset_image_files)
submit_part2(testset_image_files)
# submit_part3(testset_image_files)

