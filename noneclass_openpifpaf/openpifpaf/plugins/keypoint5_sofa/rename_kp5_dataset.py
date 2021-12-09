import numpy as np
import os

 
# count image number
def get_num_image(path):
    num_image = 0
    for image in os.listdir(path):
        if os.path.splitext(image)[1] == '.jpg':
            num_image += 1
    return num_image
 

input_path_bed="./data-keypoint-5/keypoint-5/bed/images/"
input_path_chair="./data-keypoint-5/keypoint-5/chair/images/"
input_path_sofa="./data-keypoint-5/keypoint-5/sofa/images/"
input_path_swivelchair="./data-keypoint-5/keypoint-5/swivelchair/images/"
input_path_table="./data-keypoint-5/keypoint-5/table/images/"
input_path = [input_path_bed, input_path_chair, input_path_sofa, input_path_swivelchair, input_path_table]
 
output_path_bed="./data-keypoint-5/renamed_dataset/bed/"
output_path_chair="./data-keypoint-5/renamed_dataset/chair/"
output_path_sofa="./data-keypoint-5/renamed_dataset/sofa/"
output_path_swivelchair="./data-keypoint-5/renamed_dataset/swivelchair/"
output_path_table="./data-keypoint-5/renamed_dataset/table/"
output_path = [output_path_bed, output_path_chair, output_path_sofa, output_path_swivelchair, output_path_table]

name_increment = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

for index in range(5):
    file_name = os.listdir(input_path[index])

    file_name.sort()
    
    for item in file_name:
        #print(item)
        current_image = input_path[index]+item.split('.')[0]+".jpg"

        i = get_num_image(output_path[index])  
        #print(i)

        if os.path.exists(current_image):
            i = i+1
            new_name = 'kp5_' + name_increment[index] + '_0' + format(str(i), '0>7s') + '.jpg'  # 00000001.jpg
            dst_new_name = os.path.join(os.path.abspath(output_path[index]), new_name) 
            os.rename(current_image, dst_new_name)
            
 