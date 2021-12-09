import numpy as np
import os

 
# count image number
def get_num_image(path):
    num_image = 0
    for image in os.listdir(path):
        if os.path.splitext(image)[1] == '.jpg':
            num_image += 1
    return num_image
 

#rename keypoint5
input_path_chair="./data-furniture/keypoint-5/chair/images/"
input_path_sofa="./data-furniture/keypoint-5/sofa/images/"
input_path_swivelchair="./data-furniture/keypoint-5/swivelchair/images/"
input_path_table="./data-furniture/keypoint-5/table/images/"
input_path = [input_path_chair, input_path_sofa, input_path_swivelchair, input_path_table]
 
output_path_chair="./data-furniture/renamed_keypoint5_dataset/chair/"
output_path_sofa="./data-furniture/renamed_keypoint5_dataset/sofa/"
output_path_swivelchair="./data-furniture/renamed_keypoint5_dataset/swivelchair/"
output_path_table="./data-furniture/renamed_keypoint5_dataset/table/"
output_path = [output_path_chair, output_path_sofa, output_path_swivelchair, output_path_table]

name_increment = ['chair', 'sofa', 'swivelchair', 'table']

for index in range(4):
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
            
#rename pascal3d+
input_pascal3d_images = "./data-furniture/combined_pascal3d/images/"
input_pascal3d_annotations = "./data-furniture/combined_pascal3d/annotations/"

output_pascal3d_images = "./data-furniture/combined_pascal3d/renamed_pascal3d_images/"
output_pascal3d_annotations = "./data-furniture/combined_pascal3d/renamed_pascal3d_annotations/"

file_pascal_images = os.listdir(input_pascal3d_images)
file_pascal_images.sort()
file_pascal_annotations = os.listdir(input_pascal3d_annotations)
file_pascal_annotations.sort()

for pas_image, pas_ann in zip(file_pascal_images, file_pascal_annotations):
    path_curr_im = os.path.join(input_pascal3d_images, pas_image)
    name_curr_im = os.path.splitext(pas_image)[0]

    path_curr_ann = os.path.join(input_pascal3d_annotations, pas_ann)
    name_curr_ann = os.path.splitext(pas_ann)[0]

    if os.path.exists(path_curr_im):
        tailname_curr_im = name_curr_im.split(sep='_')[1]
        newname_curr_im = name_curr_im.split(sep='_')[0] + '_0' + format(str(tailname_curr_im), '0>7s') + '.jpg'
        dist_newname_curr_im = os.path.join(output_pascal3d_images, newname_curr_im)
        os.rename(path_curr_im, dist_newname_curr_im)

    if os.path.exists(path_curr_ann):
        tailname_curr_ann = name_curr_ann.split(sep='_')[1]
        newname_curr_ann = name_curr_ann.split(sep='_')[0] + '_0' + format(str(tailname_curr_ann), '0>7s') + '.mat'
        dist_newname_curr_ann = os.path.join(output_pascal3d_annotations, newname_curr_ann)
        os.rename(path_curr_ann, dist_newname_curr_ann)