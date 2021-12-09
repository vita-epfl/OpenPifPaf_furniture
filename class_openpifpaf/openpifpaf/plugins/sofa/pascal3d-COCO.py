from scipy.io import loadmat
import numpy as np
import copy
import os
import time
from shutil import copyfile
import json
from PIL import Image

from constants import FURNITURE_KEYPOINTS_10, FURNITURE_SKELETON_10

#m = loadmat('/home/tang/pytorch_ws/furniture_dataset/mit_csail_3dinterpreter/keypoint-5/chair/coords.mat')
#print(m['coords'][:,:,0,2169])

class Pascal3dToCoco:

    def __init__(self, input_path, annotation_path, output_path_image, output_path_annotation):
        self.input_path = input_path
        self.annotation_path = annotation_path
        self.output_path_image = output_path_image
        self.output_path_annotation = output_path_annotation
        self.json_file_furniture = {}
        self.json_file_furniture['chair'] = {}
        self.json_file_furniture['chair']['train'] = {}
        self.json_file_furniture['chair']['val'] = {}
        self.json_file_furniture['sofa'] = {}
        self.json_file_furniture['sofa']['train'] = {}
        self.json_file_furniture['sofa']['val'] = {}
        self.json_file_furniture['all'] = {}
        self.json_file_furniture['all']['train'] = {}
        self.json_file_furniture['all']['val'] = {}
        self.splits = {}
        self.splits['chair'] = {}
        self.splits['chair']['train'] = []
        self.splits['chair']['val'] = []
        self.splits['sofa'] = {}
        self.splits['sofa']['train'] = []
        self.splits['sofa']['val'] = []
        self.splits['all'] = {}
        self.splits['all']['train'] = []
        self.splits['all']['val'] = []
        self.annotations = {}
        self.annotations['chair'] = {}
        self.annotations['chair']['train'] = []
        self.annotations['chair']['val'] = []
        self.annotations['sofa'] = {}
        self.annotations['sofa']['train'] = []
        self.annotations['sofa']['val'] = []
        self.annotations['all'] = {}
        self.annotations['all']['train'] = []
        self.annotations['all']['val'] = []
        self.list_to_dict = ['chair','sofa','all']

        for iter in range(3):
            os.makedirs(self.output_path_image[iter], exist_ok=True)
            os.makedirs(self.output_path_annotation[iter], exist_ok=True)
            
            #pascal3d dataset
            image_listdir = os.listdir(self.input_path[iter])
            image_listdir.sort()
            ann_listdir = os.listdir(self.annotation_path[iter])
            ann_listdir.sort()

            print(len(image_listdir))
            print(len(ann_listdir))
            for name_image, name_ann in zip(image_listdir, ann_listdir):
                if name_image == 'n03001627_00000007.jpg':
                    print(name_ann)
                curr_im = os.path.join(self.input_path[iter], name_image)
                tmp_mat = loadmat(os.path.join(self.annotation_path[iter], name_ann))
                curr_ann = copy.copy(tmp_mat['record']['objects'][0][0][0])
                self.splits[self.list_to_dict[iter]]['train'].append(curr_im)
                self.annotations[self.list_to_dict[iter]]['train'].append(curr_ann)

                flag = 0
                for idx in range(len(curr_ann)):
                    if len(curr_ann[idx]['anchors'][0]) != 0 and curr_ann[idx]['truncated'][0][0] == 0 and curr_ann[idx]['occluded'][0][0] == 0 and curr_ann[idx]['difficult'][0][0] == 0:
                        flag = 1
                        break
                if flag == 1:
                    self.splits[self.list_to_dict[iter]]['val'].append(curr_im)
                    self.annotations[self.list_to_dict[iter]]['val'].append(curr_ann)
        #print(self.splits['chair']['val'][332])
        #print(self.annotations['chair']['val'][332][0]['anchors'])

                    
    def process(self):
        """Parse and process the txt dataset into a single json file compatible with coco format"""

        # process pascal3d+ dataset
        for iter in range(3):
            for phase, im_paths in self.splits[self.list_to_dict[iter]].items():  
                cnt_images = 0
                cnt_ann = 0
                self.initiate_json(iter, phase)  # Initiate json file at each phase

                for im_path in im_paths:
                    im_size, im_id = self._process_image(iter, phase, im_path)
                    #print(im_id)
                    cnt_images += 1

                    self._process_annotation(iter, phase, cnt_ann, im_id)
                    cnt_ann += 1

                    dst = os.path.join(self.output_path_image[iter], phase, os.path.split(im_path)[-1])
                    copyfile(im_path, dst)
            
            for phase in ['train', 'val']:
                self.save_json_files(iter, phase)
                print(f'\nType:{self.list_to_dict[iter]}')
                print(f'\nPhase:{phase}')
                print(f'JSON files directory:  {self.output_path_annotation[iter]}')

    def save_json_files(self, iter, phase):
        path_json = os.path.join(self.output_path_annotation[iter], 'pascal3d_'+phase+'.json')
        with open(path_json,'w') as outfile:
            json.dump(self.json_file_furniture[self.list_to_dict[iter]][phase], outfile)


    def _process_image(self, iter, phase, im_path):
        """Update image field in json file"""
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0]

        if im_name.split(sep='_')[0][0] == 'n':
            tmp_prefix = im_name.split(sep='_')[0][1:]  
        else:
            tmp_prefix = im_name.split(sep='_')[0][:]
        tmp_mid = im_name.split(sep='_')[1]  
        tmp_tail = im_name.split(sep='_')[2]  
        im_id = int(tmp_prefix + tmp_mid + tmp_tail)

        im = Image.open(im_path)
        width, height = im.size
        dict_ann = {
            'coco_url': "unknown",
            'file_name': file_name,
            'id': im_id,
            'license': 1,
            'date_captured': "unknown",
            'width': width,
            'height': height}
        self.json_file_furniture[self.list_to_dict[iter]][phase]["images"].append(dict_ann)
        return (width, height), im_id

    def _process_annotation(self, iter, phase, cnt_ann, im_id):
        
        curr_ann = self.annotations[self.list_to_dict[iter]][phase][cnt_ann]
        num_instance = len(curr_ann)
        for idx in range(num_instance):
            # Enlarge box
            kps, num = self._transform_10(iter, phase, cnt_ann, idx)
            kps = list(kps.reshape((-1,)))
            tmp_box = curr_ann[idx]['bbox'][0]
            box = [int(tmp_box[0]), int(tmp_box[1]), int(tmp_box[2]) - int(tmp_box[0]), int(tmp_box[3]) - int(tmp_box[1])]

            furniture_id = str(im_id) + '_' + str(idx)  # include the specific annotation id
            
            self.json_file_furniture[self.list_to_dict[iter]][phase]["annotations"].append({
                'image_id': im_id,
                'category_id': 1,
                'iscrowd': 0,
                'id': furniture_id,
                'area': box[2] * box[3],
                'bbox': box,
                'num_keypoints': num,
                'keypoints': kps,
                'segmentation': []})

    def _transform_10(self, iter, phase, cnt_ann, idx):
        '''Pascal3D+ to coco format transform'''
        kps_out = np.zeros((len(FURNITURE_KEYPOINTS_10), 3))
        cnt = 0
        current_ann = self.annotations[self.list_to_dict[iter]][phase][cnt_ann][idx]
        type_furniture = current_ann['class'][0]

        if len(current_ann['anchors'][0]) == 0:
            return kps_out, cnt 
        elif phase == 'val' and (current_ann['truncated'][0][0] == 1 or current_ann['occluded'][0][0] == 1 or current_ann['difficult'][0][0] == 1):
            return kps_out, cnt 
        else:
            if type_furniture == 'chair':
                if current_ann['anchors'][0][0][8][0][0][1][0][0] == 1:
                    kps_out[0,0] = current_ann['anchors'][0][0][8][0][0][0][0][0]
                    kps_out[0,1] = current_ann['anchors'][0][0][8][0][0][0][0][1]
                    kps_out[0,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[0,0] = 0
                    kps_out[0,1] = 0
                    kps_out[0,2] = 0

                if current_ann['anchors'][0][0][9][0][0][1][0][0] == 1:
                    kps_out[1,0] = current_ann['anchors'][0][0][9][0][0][0][0][0]
                    kps_out[1,1] = current_ann['anchors'][0][0][9][0][0][0][0][1]
                    kps_out[1,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[1,0] = 0
                    kps_out[1,1] = 0
                    kps_out[1,2] = 0


                if current_ann['anchors'][0][0][6][0][0][1][0][0] == 1:
                    kps_out[2,0] = current_ann['anchors'][0][0][6][0][0][0][0][0]
                    kps_out[2,1] = current_ann['anchors'][0][0][6][0][0][0][0][1]
                    kps_out[2,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[2,0] = 0
                    kps_out[2,1] = 0
                    kps_out[2,2] = 0

                if current_ann['anchors'][0][0][7][0][0][1][0][0] == 1:
                    kps_out[3,0] = current_ann['anchors'][0][0][7][0][0][0][0][0]
                    kps_out[3,1] = current_ann['anchors'][0][0][7][0][0][0][0][1]
                    kps_out[3,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[3,0] = 0
                    kps_out[3,1] = 0
                    kps_out[3,2] = 0

                if current_ann['anchors'][0][0][4][0][0][1][0][0] == 1:
                    kps_out[4,0] = current_ann['anchors'][0][0][4][0][0][0][0][0]
                    kps_out[4,1] = current_ann['anchors'][0][0][4][0][0][0][0][1]
                    kps_out[4,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[4,0] = 0
                    kps_out[4,1] = 0
                    kps_out[4,2] = 0

                if current_ann['anchors'][0][0][5][0][0][1][0][0] == 1:
                    kps_out[5,0] = current_ann['anchors'][0][0][5][0][0][0][0][0]
                    kps_out[5,1] = current_ann['anchors'][0][0][5][0][0][0][0][1]
                    kps_out[5,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[5,0] = 0
                    kps_out[5,1] = 0
                    kps_out[5,2] = 0

                if current_ann['anchors'][0][0][2][0][0][1][0][0] == 1:
                    kps_out[6,0] = current_ann['anchors'][0][0][2][0][0][0][0][0]
                    kps_out[6,1] = current_ann['anchors'][0][0][2][0][0][0][0][1]
                    kps_out[6,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[6,0] = 0
                    kps_out[6,1] = 0
                    kps_out[6,2] = 0

                if current_ann['anchors'][0][0][3][0][0][1][0][0] == 1:
                    kps_out[7,0] = current_ann['anchors'][0][0][3][0][0][0][0][0]
                    kps_out[7,1] = current_ann['anchors'][0][0][3][0][0][0][0][1]
                    kps_out[7,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[7,0] = 0
                    kps_out[7,1] = 0
                    kps_out[7,2] = 0

                if current_ann['anchors'][0][0][0][0][0][1][0][0] == 1:
                    kps_out[8,0] = current_ann['anchors'][0][0][0][0][0][0][0][0]
                    kps_out[8,1] = current_ann['anchors'][0][0][0][0][0][0][0][1]
                    kps_out[8,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[8,0] = 0
                    kps_out[8,1] = 0
                    kps_out[8,2] = 0

                if current_ann['anchors'][0][0][1][0][0][1][0][0] == 1:
                    kps_out[9,0] = current_ann['anchors'][0][0][1][0][0][0][0][0]
                    kps_out[9,1] = current_ann['anchors'][0][0][1][0][0][0][0][1]
                    kps_out[9,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[9,0] = 0
                    kps_out[9,1] = 0
                    kps_out[9,2] = 0


            elif type_furniture == 'sofa':
                if current_ann['anchors'][0][0][0][0][0][1][0][0] == 1:
                    kps_out[0,0] = current_ann['anchors'][0][0][0][0][0][0][0][0]
                    kps_out[0,1] = current_ann['anchors'][0][0][0][0][0][0][0][1]
                    kps_out[0,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[0,0] = 0
                    kps_out[0,1] = 0
                    kps_out[0,2] = 0

                if current_ann['anchors'][0][0][1][0][0][1][0][0] == 1:
                    kps_out[1,0] = current_ann['anchors'][0][0][1][0][0][0][0][0]
                    kps_out[1,1] = current_ann['anchors'][0][0][1][0][0][0][0][1]
                    kps_out[1,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[1,0] = 0
                    kps_out[1,1] = 0
                    kps_out[1,2] = 0


                if current_ann['anchors'][0][0][4][0][0][1][0][0] == 1:
                    kps_out[2,0] = current_ann['anchors'][0][0][4][0][0][0][0][0]
                    kps_out[2,1] = current_ann['anchors'][0][0][4][0][0][0][0][1]
                    kps_out[2,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[2,0] = 0
                    kps_out[2,1] = 0
                    kps_out[2,2] = 0

                if current_ann['anchors'][0][0][5][0][0][1][0][0] == 1:
                    kps_out[3,0] = current_ann['anchors'][0][0][5][0][0][0][0][0]
                    kps_out[3,1] = current_ann['anchors'][0][0][5][0][0][0][0][1]
                    kps_out[3,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[3,0] = 0
                    kps_out[3,1] = 0
                    kps_out[3,2] = 0

                if current_ann['anchors'][0][0][2][0][0][1][0][0] == 1:
                    kps_out[4,0] = current_ann['anchors'][0][0][2][0][0][0][0][0]
                    kps_out[4,1] = current_ann['anchors'][0][0][2][0][0][0][0][1]
                    kps_out[4,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[4,0] = 0
                    kps_out[4,1] = 0
                    kps_out[4,2] = 0

                if current_ann['anchors'][0][0][3][0][0][1][0][0] == 1:
                    kps_out[5,0] = current_ann['anchors'][0][0][3][0][0][0][0][0]
                    kps_out[5,1] = current_ann['anchors'][0][0][3][0][0][0][0][1]
                    kps_out[5,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[5,0] = 0
                    kps_out[5,1] = 0
                    kps_out[5,2] = 0

                if current_ann['anchors'][0][0][8][0][0][1][0][0] == 1:
                    kps_out[6,0] = current_ann['anchors'][0][0][8][0][0][0][0][0]
                    kps_out[6,1] = current_ann['anchors'][0][0][8][0][0][0][0][1]
                    kps_out[6,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[6,0] = 0
                    kps_out[6,1] = 0
                    kps_out[6,2] = 0

                if current_ann['anchors'][0][0][9][0][0][1][0][0] == 1:
                    kps_out[7,0] = current_ann['anchors'][0][0][9][0][0][0][0][0]
                    kps_out[7,1] = current_ann['anchors'][0][0][9][0][0][0][0][1]
                    kps_out[7,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[7,0] = 0
                    kps_out[7,1] = 0
                    kps_out[7,2] = 0

                if current_ann['anchors'][0][0][6][0][0][1][0][0] == 1:
                    kps_out[8,0] = current_ann['anchors'][0][0][6][0][0][0][0][0]
                    kps_out[8,1] = current_ann['anchors'][0][0][6][0][0][0][0][1]
                    kps_out[8,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[8,0] = 0
                    kps_out[8,1] = 0
                    kps_out[8,2] = 0

                if current_ann['anchors'][0][0][7][0][0][1][0][0] == 1:
                    kps_out[9,0] = current_ann['anchors'][0][0][7][0][0][0][0][0]
                    kps_out[9,1] = current_ann['anchors'][0][0][7][0][0][0][0][1]
                    kps_out[9,2] = 2
                    cnt = cnt+1
                else:
                    kps_out[9,0] = 0
                    kps_out[9,1] = 0
                    kps_out[9,2] = 0

            for index in range(10):
                kps_out[index,0] = round(kps_out[index,0], 2)
                kps_out[index,1] = round(kps_out[index,1], 2)
            return kps_out, cnt

    def initiate_json(self, iter, phase):
        """
        Initiate Json for training and val phase for the 10 kp furniture skeleton
        """
        self.json_file_furniture[self.list_to_dict[iter]][phase]["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
                                                    date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                                                 time.localtime()),
                                                    description=("Conversion of Pascal3D+ dataset into MS-COCO"
                                                                 " format with 10 keypoints"))
        self.json_file_furniture[self.list_to_dict[iter]][phase]["categories"] = [dict(name='furniture',
                                                           id=1,
                                                           skeleton=FURNITURE_SKELETON_10,
                                                           supercategory='furniture',
                                                           keypoints=FURNITURE_KEYPOINTS_10)]
        self.json_file_furniture[self.list_to_dict[iter]][phase]["images"] = []
        self.json_file_furniture[self.list_to_dict[iter]][phase]["annotations"] = []


def main():
    
    input_path_chair="./data-pascal3d/renamed_images/chair/"
    input_path_sofa="./data-pascal3d/renamed_images/sofa/"
    input_path_all="./data-pascal3d/renamed_images/all/"
    input_path= [input_path_chair, input_path_sofa, input_path_all]
    
    annotation_chair = "./data-pascal3d/renamed_annotations/chair/"
    annotation_sofa = "./data-pascal3d/renamed_annotations/sofa/"
    annotation_all = "./data-pascal3d/renamed_annotations/all/"
    annotation_path= [annotation_chair, annotation_sofa, annotation_all]

    output_path_image_chair = "./data-pascal3d/images_chair/"
    output_path_image_sofa = "./data-pascal3d/images_sofa/"
    output_path_image_all = "./data-pascal3d/images_all/"
    output_path_image = [output_path_image_chair, output_path_image_sofa, output_path_image_all]

    output_path_annotation_chair = "./data-pascal3d/annotations_chair/"
    output_path_annotation_sofa = "./data-pascal3d/annotations_sofa/"
    output_path_annotation_all = "./data-pascal3d/annotations_all/"
    output_path_annotation = [output_path_annotation_chair, output_path_annotation_sofa, output_path_annotation_all]

    pascal3d_to_coco = Pascal3dToCoco(input_path, annotation_path, output_path_image, output_path_annotation)
    
    pascal3d_to_coco.process()


if __name__ == "__main__":
    main()