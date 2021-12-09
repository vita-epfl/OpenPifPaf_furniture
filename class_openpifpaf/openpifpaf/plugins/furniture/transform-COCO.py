from scipy.io import loadmat
import numpy as np
import os
import time
from shutil import copyfile
import json
from PIL import Image

from constants import FURNITURE_KEYPOINTS_10, FURNITURE_SKELETON_10

#m = loadmat('/home/tang/pytorch_ws/furniture_dataset/mit_csail_3dinterpreter/keypoint-5/chair/coords.mat')
#print(m['coords'][:,:,0,2169])

class FurnitureToCoco:

    def __init__(self, input_path_keypoint5, input_path_pascal3d, annotation_path_keypoint5, annotation_path_pascal3d, output_path_image, output_path_annotation):
        self.input_path_keypoint5 = input_path_keypoint5
        self.input_path_pascal3d = input_path_pascal3d
        self.annotation_path_keypoint5 = annotation_path_keypoint5
        self.annotation_path_pascal3d = annotation_path_pascal3d
        self.output_path_image = output_path_image
        self.output_path_annotation = output_path_annotation
        self.json_file_furniture = {}
        self.json_file_furniture['train'] = {}
        self.json_file_furniture['val'] = {}
        self.splits_keypoint5 = {}
        self.splits_keypoint5['train'] = []
        self.splits_keypoint5['val'] = []
        self.annotations_keypoint5 = {}
        self.annotations_keypoint5['train'] = []
        self.annotations_keypoint5['val'] = []
        self.splits_pascal3d = {}
        self.splits_pascal3d['train'] = []
        self.splits_pascal3d['val'] = []
        self.annotations_pascal3d = {}
        self.annotations_pascal3d['train'] = []
        self.annotations_pascal3d['val'] = []
        kp5_init_kp_num = [10, 14, 13, 8]

        os.makedirs(self.output_path_image, exist_ok=True)
        os.makedirs(self.output_path_annotation, exist_ok=True)
  
        #keypoint-5 dataset
        for iter in range(4):
            
            '''Prepare annotations'''
            tmp_mat = loadmat(self.annotation_path_keypoint5[iter])
            
            if(iter==0):
                tmp_mat['coords'][:,:,0,560]=tmp_mat['coords'][:,:,1,560]

            cnt_ann_kpt = 0

            image_listdir = os.listdir(self.input_path_keypoint5[iter])
            image_listdir.sort()
            for name_image in image_listdir:
                tmp_main_name = os.path.splitext(name_image)[0]
                tmp_tail_name = int(tmp_main_name.split(sep='_')[2])
                if tmp_tail_name%10 == 3 or tmp_tail_name%10 == 6 or tmp_tail_name%10 == 9:
                    self.splits_keypoint5['val'].append(os.path.join(self.input_path_keypoint5[iter], name_image))
                    self.annotations_keypoint5['val'].append(tmp_mat['coords'][:,0:kp5_init_kp_num[iter],0,cnt_ann_kpt])
                else:
                    self.splits_keypoint5['train'].append(os.path.join(self.input_path_keypoint5[iter], name_image))
                    self.annotations_keypoint5['train'].append(tmp_mat['coords'][:,0:kp5_init_kp_num[iter],0,cnt_ann_kpt])
                cnt_ann_kpt += 1
        
        #pascal3d dataset
        image_listdir = os.listdir(self.input_path_pascal3d)
        image_listdir.sort()
        ann_listdir = os.listdir(self.annotation_path_pascal3d)
        ann_listdir.sort()
        for name_image, name_ann in zip(image_listdir, ann_listdir):
            tmp_main_name = os.path.splitext(name_image)[0]
            tmp_tail_name = int(tmp_main_name.split(sep='_')[1])
            tmp_mat = loadmat(os.path.join(annotation_path_pascal3d, name_ann))
            if tmp_tail_name%10 == 3 or tmp_tail_name%10 == 6 or tmp_tail_name%10 == 9:
                self.splits_pascal3d['val'].append(os.path.join(self.input_path_pascal3d, name_image))
                self.annotations_pascal3d['val'].append(tmp_mat['record']['objects'][0][0][0])
            else:
                self.splits_pascal3d['train'].append(os.path.join(self.input_path_pascal3d, name_image))
                self.annotations_pascal3d['train'].append(tmp_mat['record']['objects'][0][0][0])

    def process(self):
        """Parse and process the txt dataset into a single json file compatible with coco format"""

        
        # process keypoint-5 dataset
        for phase, im_paths in self.splits_keypoint5.items():  
            cnt_images = 0
            cnt_ann = 0
            self.initiate_json(phase)  # Initiate json file at each phase

            path_dir = (os.path.join(self.output_path_image, phase))
            os.makedirs(path_dir, exist_ok=True)

            for im_path in im_paths:
                im_size, im_id = self._process_image_keypoint5(phase, im_path)
                #print(im_id)
                cnt_images += 1

                self._process_annotation_keypoint5(phase, im_path, cnt_ann, im_size, im_id)
                cnt_ann += 1

                dst = os.path.join(self.output_path_image, phase, os.path.split(im_path)[-1])
                copyfile(im_path, dst)


        # process pascal3d+ dataset
        for phase, im_paths in self.splits_pascal3d.items():  
            cnt_images = 0
            cnt_ann = 0

            for im_path in im_paths:
                im_size, im_id = self._process_image_pascal3d(phase, im_path)
                #print(im_id)
                cnt_images += 1

                self._process_annotation_pascal3d(phase, im_path, cnt_ann, im_size, im_id)
                cnt_ann += 1

                dst = os.path.join(self.output_path_image, phase, os.path.split(im_path)[-1])
                copyfile(im_path, dst)
        
        for phase in ['train', 'val']:
            self.save_json_files(phase)
            print(f'\nPhase:{phase}')
            print(f'JSON files directory:  {self.output_path_annotation}')

    def save_json_files(self, phase):
        path_json = os.path.join(self.output_path_annotation, 'furniture_'+phase+'.json')
        with open(path_json,'w') as outfile:
            json.dump(self.json_file_furniture[phase], outfile)

    def _process_image_keypoint5(self, phase, im_path):
        """Update image field in json file"""
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0]
        tmp_prefix = im_name.split(sep='_')[1]  # Type of the image
        tmp_im_id = int(im_name.split(sep='_')[2])  # Numeric code in the image
        im_id = 0
        if tmp_prefix == 'chair':
            im_id = 10000000 + tmp_im_id
        if tmp_prefix == 'sofa':
            im_id = 20000000 + tmp_im_id
        if tmp_prefix == 'swivelchair':
            im_id = 30000000 + tmp_im_id
        if tmp_prefix == 'table':
            im_id = 40000000 + tmp_im_id
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
        self.json_file_furniture[phase]["images"].append(dict_ann)
        return (width, height), im_id

    def _process_image_pascal3d(self, phase, im_path):
        """Update image field in json file"""
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0]
        tmp_prefix = im_name.split(sep='_')[0][1:]  
        tmp_tail = im_name.split(sep='_')[1]  
        im_id = int(tmp_prefix + tmp_tail)

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
        self.json_file_furniture[phase]["images"].append(dict_ann)
        return (width, height), im_id

    def _process_annotation_keypoint5(self, phase, im_path, cnt_ann, im_size, im_id):
        '''Process single instance'''
        
        # Enlarge box
        kps, num = self._transform_keypoint5_10(phase, im_path, cnt_ann)
        tmp_kps = kps
        box_tight = [np.min(tmp_kps[:, 0]), np.min(tmp_kps[:, 1]),np.max(tmp_kps[:, 0]), np.max(tmp_kps[:, 1])]

        kps = list(kps.reshape((-1,)))

        w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
        x_o = max(box_tight[0] - 0.1 * w, 0)
        y_o = max(box_tight[1] - 0.1 * h, 0)
        x_i = min(box_tight[0] + 1.1 * w, im_size[0])
        y_i = min(box_tight[1] + 1.1 * h, im_size[1])
        box = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)

        furniture_id = str(im_id) + str(0)  # include the specific annotation id
        
        self.json_file_furniture[phase]["annotations"].append({
            'image_id': im_id,
            'category_id': 1,
            'iscrowd': 0,
            'id': furniture_id,
            'area': box[2] * box[3],
            'bbox': box,
            'num_keypoints': num,
            'keypoints': kps,
            'segmentation': []})

    def _process_annotation_pascal3d(self, phase, im_path, cnt_ann, im_size, im_id):
        
        num_instance = len(self.annotations_pascal3d[phase][cnt_ann])

        for iter in range(num_instance):
            # Enlarge box
            kps, num = self._transform_pascal3d_10(phase, cnt_ann, iter)
            tmp_kps = kps
            box_tight = [np.min(tmp_kps[:, 0]), np.min(tmp_kps[:, 1]),np.max(tmp_kps[:, 0]), np.max(tmp_kps[:, 1])]

            kps = list(kps.reshape((-1,)))

            w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
            x_o = max(box_tight[0] - 0.1 * w, 0)
            y_o = max(box_tight[1] - 0.1 * h, 0)
            x_i = min(box_tight[0] + 1.1 * w, im_size[0])
            y_i = min(box_tight[1] + 1.1 * h, im_size[1])
            box = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)

            furniture_id = str(im_id) + '_' + str(iter)  # include the specific annotation id
            
            self.json_file_furniture[phase]["annotations"].append({
                'image_id': im_id,
                'category_id': 1,
                'iscrowd': 0,
                'id': furniture_id,
                'area': box[2] * box[3],
                'bbox': box,
                'num_keypoints': num,
                'keypoints': kps,
                'segmentation': []})

    def _transform_keypoint5_10(self, phase, im_path, cnt_ann):
        '''keypoint5 to coco format transform'''
        kps_out = np.zeros((len(FURNITURE_KEYPOINTS_10), 3))
        cnt = 0
        
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0]
        im_type = im_name.split(sep='_')[1]  # type of furniture

        if(im_type == 'chair'):
            kps_out[0,0] = self.annotations_keypoint5[phase][cnt_ann][0,0]
            kps_out[0,1] = self.annotations_keypoint5[phase][cnt_ann][1,0]
            kps_out[0,2] = 2

            kps_out[1,0] = self.annotations_keypoint5[phase][cnt_ann][0,1]
            kps_out[1,1] = self.annotations_keypoint5[phase][cnt_ann][1,1]
            kps_out[1,2] = 2

            kps_out[2,0] = self.annotations_keypoint5[phase][cnt_ann][0,3]
            kps_out[2,1] = self.annotations_keypoint5[phase][cnt_ann][1,3]
            kps_out[2,2] = 2

            kps_out[3,0] = self.annotations_keypoint5[phase][cnt_ann][0,2]
            kps_out[3,1] = self.annotations_keypoint5[phase][cnt_ann][1,2]
            kps_out[3,2] = 2

            kps_out[4,0] = self.annotations_keypoint5[phase][cnt_ann][0,4]
            kps_out[4,1] = self.annotations_keypoint5[phase][cnt_ann][1,4]
            kps_out[4,2] = 2

            kps_out[5,0] = self.annotations_keypoint5[phase][cnt_ann][0,5]
            kps_out[5,1] = self.annotations_keypoint5[phase][cnt_ann][1,5]
            kps_out[5,2] = 2

            kps_out[6,0] = self.annotations_keypoint5[phase][cnt_ann][0,7]
            kps_out[6,1] = self.annotations_keypoint5[phase][cnt_ann][1,7]
            kps_out[6,2] = 2

            kps_out[7,0] = self.annotations_keypoint5[phase][cnt_ann][0,6]
            kps_out[7,1] = self.annotations_keypoint5[phase][cnt_ann][1,6]
            kps_out[7,2] = 2

            kps_out[8,0] = self.annotations_keypoint5[phase][cnt_ann][0,8]
            kps_out[8,1] = self.annotations_keypoint5[phase][cnt_ann][1,8]
            kps_out[8,2] = 2

            kps_out[9,0] = self.annotations_keypoint5[phase][cnt_ann][0,9]
            kps_out[9,1] = self.annotations_keypoint5[phase][cnt_ann][1,9]
            kps_out[9,2] = 2

            cnt = 10

        elif(im_type == 'sofa'):
            kps_out[0,0] = self.annotations_keypoint5[phase][cnt_ann][0,1]
            kps_out[0,1] = self.annotations_keypoint5[phase][cnt_ann][1,1]
            kps_out[0,2] = 2

            kps_out[1,0] = self.annotations_keypoint5[phase][cnt_ann][0,8]
            kps_out[1,1] = self.annotations_keypoint5[phase][cnt_ann][1,8]
            kps_out[1,2] = 2

            kps_out[2,0] = self.annotations_keypoint5[phase][cnt_ann][0,0]
            kps_out[2,1] = self.annotations_keypoint5[phase][cnt_ann][1,0]
            kps_out[2,2] = 2

            kps_out[3,0] = self.annotations_keypoint5[phase][cnt_ann][0,7]
            kps_out[3,1] = self.annotations_keypoint5[phase][cnt_ann][1,7]
            kps_out[3,2] = 2

            kps_out[4,0] = self.annotations_keypoint5[phase][cnt_ann][0,3]
            kps_out[4,1] = self.annotations_keypoint5[phase][cnt_ann][1,3]
            kps_out[4,2] = 2

            kps_out[5,0] = self.annotations_keypoint5[phase][cnt_ann][0,10]
            kps_out[5,1] = self.annotations_keypoint5[phase][cnt_ann][1,10]
            kps_out[5,2] = 2

            kps_out[6,0] = self.annotations_keypoint5[phase][cnt_ann][0,2]
            kps_out[6,1] = self.annotations_keypoint5[phase][cnt_ann][1,2]
            kps_out[6,2] = 2

            kps_out[7,0] = self.annotations_keypoint5[phase][cnt_ann][0,9]
            kps_out[7,1] = self.annotations_keypoint5[phase][cnt_ann][1,9]
            kps_out[7,2] = 2

            kps_out[8,0] = self.annotations_keypoint5[phase][cnt_ann][0,6]
            kps_out[8,1] = self.annotations_keypoint5[phase][cnt_ann][1,6]
            kps_out[8,2] = 2

            kps_out[9,0] = self.annotations_keypoint5[phase][cnt_ann][0,13]
            kps_out[9,1] = self.annotations_keypoint5[phase][cnt_ann][1,13]
            kps_out[9,2] = 2

            cnt = 10

        elif(im_type == 'swivelchair'):
            kps_out[0,0] = 0
            kps_out[0,1] = 0
            kps_out[0,2] = 0

            kps_out[1,0] = 0
            kps_out[1,1] = 0
            kps_out[1,2] = 0

            kps_out[2,0] = 0
            kps_out[2,1] = 0
            kps_out[2,2] = 0

            kps_out[3,0] = 0
            kps_out[3,1] = 0
            kps_out[3,2] = 0

            kps_out[4,0] = self.annotations_keypoint5[phase][cnt_ann][0,7]
            kps_out[4,1] = self.annotations_keypoint5[phase][cnt_ann][1,7]
            kps_out[4,2] = 2

            kps_out[5,0] = self.annotations_keypoint5[phase][cnt_ann][0,8]
            kps_out[5,1] = self.annotations_keypoint5[phase][cnt_ann][1,8]
            kps_out[5,2] = 2

            kps_out[6,0] = self.annotations_keypoint5[phase][cnt_ann][0,10]
            kps_out[6,1] = self.annotations_keypoint5[phase][cnt_ann][1,10]
            kps_out[6,2] = 2

            kps_out[7,0] = self.annotations_keypoint5[phase][cnt_ann][0,9]
            kps_out[7,1] = self.annotations_keypoint5[phase][cnt_ann][1,9]
            kps_out[7,2] = 2

            kps_out[8,0] = self.annotations_keypoint5[phase][cnt_ann][0,11]
            kps_out[8,1] = self.annotations_keypoint5[phase][cnt_ann][1,11]
            kps_out[8,2] = 2

            kps_out[9,0] = self.annotations_keypoint5[phase][cnt_ann][0,12]
            kps_out[9,1] = self.annotations_keypoint5[phase][cnt_ann][1,12]
            kps_out[9,2] = 2

            cnt = 6

        elif(im_type == 'table'):
            kps_out[0,0] = self.annotations_keypoint5[phase][cnt_ann][0,4]
            kps_out[0,1] = self.annotations_keypoint5[phase][cnt_ann][1,4]
            kps_out[0,2] = 2

            kps_out[1,0] = self.annotations_keypoint5[phase][cnt_ann][0,5]
            kps_out[1,1] = self.annotations_keypoint5[phase][cnt_ann][1,5]
            kps_out[1,2] = 2

            kps_out[2,0] = self.annotations_keypoint5[phase][cnt_ann][0,6]
            kps_out[2,1] = self.annotations_keypoint5[phase][cnt_ann][1,6]
            kps_out[2,2] = 2

            kps_out[3,0] = self.annotations_keypoint5[phase][cnt_ann][0,7]
            kps_out[3,1] = self.annotations_keypoint5[phase][cnt_ann][1,7]
            kps_out[3,2] = 2

            kps_out[4,0] = self.annotations_keypoint5[phase][cnt_ann][0,0]
            kps_out[4,1] = self.annotations_keypoint5[phase][cnt_ann][1,0]
            kps_out[4,2] = 2

            kps_out[5,0] = self.annotations_keypoint5[phase][cnt_ann][0,1]
            kps_out[5,1] = self.annotations_keypoint5[phase][cnt_ann][1,1]
            kps_out[5,2] = 2

            kps_out[6,0] = self.annotations_keypoint5[phase][cnt_ann][0,2]
            kps_out[6,1] = self.annotations_keypoint5[phase][cnt_ann][1,2]
            kps_out[6,2] = 2

            kps_out[7,0] = self.annotations_keypoint5[phase][cnt_ann][0,3]
            kps_out[7,1] = self.annotations_keypoint5[phase][cnt_ann][1,3]
            kps_out[7,2] = 2

            kps_out[8,0] = 0
            kps_out[8,1] = 0
            kps_out[8,2] = 0

            kps_out[9,0] = 0
            kps_out[9,1] = 0
            kps_out[9,2] = 0

            cnt = 8

        for index in range(10):
            kps_out[index,0] = round(kps_out[index,0], 2)
            kps_out[index,1] = round(kps_out[index,1], 2)
        return kps_out, cnt

    def _transform_pascal3d_10(self, phase, cnt_ann, iter):
        '''keypoint5 to coco format transform'''
        kps_out = np.zeros((len(FURNITURE_KEYPOINTS_10), 3))
        cnt = 0
        current_ann = self.annotations_pascal3d[phase][cnt_ann][iter]
        type_furniture = current_ann['class']

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

    def initiate_json(self, phase):
        """
        Initiate Json for training and val phase for the 10 kp furniture skeleton
        """
        self.json_file_furniture[phase]["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
                                                    date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                                                 time.localtime()),
                                                    description=("Conversion of furniture dataset into MS-COCO"
                                                                 " format with 10 keypoints"))
        self.json_file_furniture[phase]["categories"] = [dict(name='furniture',
                                                           id=1,
                                                           skeleton=FURNITURE_SKELETON_10,
                                                           supercategory='furniture',
                                                           keypoints=FURNITURE_KEYPOINTS_10)]
        self.json_file_furniture[phase]["images"] = []
        self.json_file_furniture[phase]["annotations"] = []


def main():
    
    input_path_chair="./data-furniture/renamed_keypoint5_dataset/chair/"
    input_path_sofa="./data-furniture/renamed_keypoint5_dataset/sofa/"
    input_path_swivelchair="./data-furniture/renamed_keypoint5_dataset/swivelchair/"
    input_path_table="./data-furniture/renamed_keypoint5_dataset/table/"
    input_path_keypoint5 = [input_path_chair, input_path_sofa, input_path_swivelchair, input_path_table]
    input_path_pascal3d = "./data-furniture/combined_pascal3d/renamed_pascal3d_images/"
    
    annotation_chair = "./data-furniture/keypoint-5/chair/coords.mat"
    annotation_sofa = "./data-furniture/keypoint-5/sofa/coords.mat"
    annotation_swivelchair = "./data-furniture/keypoint-5/swivelchair/coords.mat"
    annotation_table = "./data-furniture/keypoint-5/table/coords.mat"
    annotation_path_keypoint5 = [annotation_chair, annotation_sofa, annotation_swivelchair, annotation_table]
    annotation_path_pascal3d = "./data-furniture/combined_pascal3d/renamed_pascal3d_annotations/"

    output_path_image = "./data-furniture/images/"
    output_path_annotation = "./data-furniture/annotations/"

    furniture_to_coco = FurnitureToCoco(input_path_keypoint5, input_path_pascal3d, annotation_path_keypoint5, annotation_path_pascal3d, output_path_image, output_path_annotation)
    
    furniture_to_coco.process()


if __name__ == "__main__":
    main()