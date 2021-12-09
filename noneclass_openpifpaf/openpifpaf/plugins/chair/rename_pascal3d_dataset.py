import numpy as np
import os

#rename pascal3d+
input_pascal3d_images_chair = "./data-pascal3d/images/chair_imagenet/"
input_pascal3d_images_diningtable = "./data-pascal3d/images/diningtable_imagenet/"
input_pascal3d_images_sofa = "./data-pascal3d/images/sofa_imagenet/"
input_pascal3d_images = [input_pascal3d_images_chair, input_pascal3d_images_diningtable, input_pascal3d_images_sofa]

input_pascal3d_annotations_chair = "./data-pascal3d/annotations/chair_imagenet/"
input_pascal3d_annotations_diningtable = "./data-pascal3d/annotations/diningtable_imagenet/"
input_pascal3d_annotations_sofa = "./data-pascal3d/annotations/sofa_imagenet/"
input_pascal3d_annotations = [input_pascal3d_annotations_chair, input_pascal3d_annotations_diningtable, input_pascal3d_annotations_sofa]

output_pascal3d_images_chair = "./data-pascal3d/renamed_images/chair_imagenet/"
output_pascal3d_images_diningtable = "./data-pascal3d/renamed_images/diningtable_imagenet/"
output_pascal3d_images_sofa = "./data-pascal3d/renamed_images/sofa_imagenet/"
output_pascal3d_images = [output_pascal3d_images_chair, output_pascal3d_images_diningtable, output_pascal3d_images_sofa]

output_pascal3d_annotations_chair = "./data-pascal3d/renamed_annotations/chair_imagenet/"
output_pascal3d_annotations_diningtable = "./data-pascal3d/renamed_annotations/diningtable_imagenet/"
output_pascal3d_annotations_sofa = "./data-pascal3d/renamed_annotations/sofa_imagenet/"
output_pascal3d_annotations = [output_pascal3d_annotations_chair, output_pascal3d_annotations_diningtable, output_pascal3d_annotations_sofa]

for iter in range(3):
    file_pascal_images = os.listdir(input_pascal3d_images[iter])
    file_pascal_images.sort()
    file_pascal_annotations = os.listdir(input_pascal3d_annotations[iter])
    file_pascal_annotations.sort()

    for pas_image, pas_ann in zip(file_pascal_images, file_pascal_annotations):
        path_curr_im = os.path.join(input_pascal3d_images[iter], pas_image)
        name_curr_im = os.path.splitext(pas_image)[0]

        path_curr_ann = os.path.join(input_pascal3d_annotations[iter], pas_ann)
        name_curr_ann = os.path.splitext(pas_ann)[0]

        if os.path.exists(path_curr_im):
            tailname_curr_im = name_curr_im.split(sep='_')[1]
            newname_curr_im = name_curr_im.split(sep='_')[0] + '_0' + format(str(tailname_curr_im), '0>7s') + '.jpg'
            dist_newname_curr_im = os.path.join(output_pascal3d_images[iter], newname_curr_im)
            os.rename(path_curr_im, dist_newname_curr_im)

        if os.path.exists(path_curr_ann):
            tailname_curr_ann = name_curr_ann.split(sep='_')[1]
            newname_curr_ann = name_curr_ann.split(sep='_')[0] + '_0' + format(str(tailname_curr_ann), '0>7s') + '.mat'
            dist_newname_curr_ann = os.path.join(output_pascal3d_annotations[iter], newname_curr_ann)
            os.rename(path_curr_ann, dist_newname_curr_ann)