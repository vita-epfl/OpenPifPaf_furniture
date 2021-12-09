from scipy.io import loadmat
import numpy as np
import os
import time
from shutil import copyfile
import json
from PIL import Image

'''
annotation_chair = "./data-pascal3d/renamed_annotations/chair/"
annotation_sofa = "./data-pascal3d/renamed_annotations/sofa/"
annotation_diningtable = "./data-pascal3d/renamed_annotations/diningtable/"
annotation_path= [annotation_chair, annotation_sofa, annotation_diningtable]
cnt = 0
list_to_dict = ['chair', 'sofa', 'diningtable']

for iter in range(3):

    ann_listdir = os.listdir(annotation_path[iter])
    ann_listdir.sort()

    for name_ann in ann_listdir:
        tmp_mat = loadmat(os.path.join(annotation_path[iter], name_ann))
        curr_ann = tmp_mat['record']['objects'][0][0][0]

        flag = 0
        for item in curr_ann:
            if item['class'][0] == list_to_dict[iter] and item['truncated'][0][0] == 0 and item['occluded'][0][0] == 0 and item['difficult'][0][0] == 0:
                flag = 1
                break
        if flag == 1:
            cnt = cnt + 1

print(cnt)


#n04331277_00007359.jpg

#n04331277_00008089.jpg
'''
tmp_mat = loadmat('./data-pascal3d/renamed_annotations/chair/2008_1_000043.mat')
curr_ann = tmp_mat['record']['objects'][0][0][0][0]['anchors'][0]
print(len(curr_ann))

a, b = 1
print(a)
print(b)