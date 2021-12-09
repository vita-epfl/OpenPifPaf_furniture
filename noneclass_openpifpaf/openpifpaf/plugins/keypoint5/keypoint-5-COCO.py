from scipy.io import loadmat
from scipy.io import savemat
import copy
import numpy as np
import os
import time
from shutil import copyfile
import json
from PIL import Image

from openpifpaf.plugins.keypoint5.constants import FURNITURE_KEYPOINTS_10, FURNITURE_SKELETON_10

class Keypoint5ToCoco:

    def __init__(self, input_path, annotation_path, output_path_image, output_path_annotation):
        self.input_path = input_path
        self.annotation_path = annotation_path
        self.output_path_image = output_path_image
        self.output_path_annotation = output_path_annotation
        self.json_file_kp5_furniture = {}
        self.json_file_kp5_furniture['bed'] = {}
        self.json_file_kp5_furniture['chair'] = {}
        self.json_file_kp5_furniture['sofa'] = {}
        self.json_file_kp5_furniture['swivelchair'] = {}
        self.json_file_kp5_furniture['table'] = {}
        self.json_file_kp5_furniture['all'] = {}

        self.splits = {}
        self.splits['bed'] = {}
        self.splits['chair'] = {}
        self.splits['sofa'] = {}
        self.splits['swivelchair'] = {}
        self.splits['table'] = {}
        self.splits['all'] = {}
        self.annotations = {}
        self.annotations['bed'] = {}
        self.annotations['chair'] = {}
        self.annotations['sofa'] = {}
        self.annotations['swivelchair'] = {}
        self.annotations['table'] = {}
        self.annotations['all'] = {}
        self.deviation = {}
        init_kp_num = [10, 10, 14, 13, 8]

        os.makedirs(self.output_path_image['bed'], exist_ok=True)
        os.makedirs(self.output_path_image['chair'], exist_ok=True)
        os.makedirs(self.output_path_image['sofa'], exist_ok=True)
        os.makedirs(self.output_path_image['swivelchair'], exist_ok=True)
        os.makedirs(self.output_path_image['table'], exist_ok=True)
        os.makedirs(self.output_path_image['all'], exist_ok=True)
        os.makedirs(self.output_path_annotation['bed'], exist_ok=True)
        os.makedirs(self.output_path_annotation['chair'], exist_ok=True)
        os.makedirs(self.output_path_annotation['sofa'], exist_ok=True)
        os.makedirs(self.output_path_annotation['swivelchair'], exist_ok=True)
        os.makedirs(self.output_path_annotation['table'], exist_ok=True)
        os.makedirs(self.output_path_annotation['all'], exist_ok=True)
  
        self.splits['all']['test'] = []
        self.annotations['all']['test'] = []
        self.splits['all']['val'] = []
        self.annotations['all']['val'] = []
        self.splits['all']['train'] = []
        self.annotations['all']['train'] = []

        for iter in range(5):
            
            '''Prepare annotations'''
            tmp_mat = loadmat(self.annotation_path[iter])
            num_image = len(tmp_mat['coords'][0,0,0,:])
            num_kp = init_kp_num[iter]
            num_people = len(tmp_mat['coords'][0,0,:,0])
            curr_deviation = np.zeros((num_image, num_kp))

            for idx_image in range(num_image):
                for idx_kp in range(num_kp):
                    mean_point = np.array([0, 0])
                    num_real_people = 0
                    for idx_people in range(num_people):
                        curr_point = tmp_mat['coords'][:,idx_kp,idx_people,idx_image]
                        if np.isnan(curr_point[0]) or np.isnan(curr_point[1]) or curr_point[0] == 0 or curr_point[1] == 0:
                            continue
                        mean_point = mean_point + curr_point
                        num_real_people = num_real_people+1
                    mean_point = mean_point / float(num_real_people)
                    
                    tmp_deviation = 0
                    for idx_people in range(num_people):
                        curr_point = tmp_mat['coords'][:,idx_kp,idx_people,idx_image]
                        if np.isnan(curr_point[0]) or np.isnan(curr_point[1]) or curr_point[0] == 0 or curr_point[1] == 0:
                            continue
                        dst = pow(float(np.linalg.norm(curr_point-mean_point)),2)
                        tmp_deviation = tmp_deviation + dst
                    
                    if num_real_people == 1:
                        tmp_deviation = 0
                    else:    
                        tmp_deviation = float(tmp_deviation) / (float(num_real_people) - 1)

                    tmp_mat['coords'][:,idx_kp,0,idx_image] = mean_point
                    curr_deviation[idx_image, idx_kp] = pow(tmp_deviation,0.5)
            
            if iter==0:
                print(curr_deviation)
                self.deviation['bed'] = curr_deviation
                self.splits['bed']['test'] = []
                self.annotations['bed']['test'] = []
                self.splits['bed']['val'] = []
                self.annotations['bed']['val'] = []
                self.splits['bed']['train'] = []
                self.annotations['bed']['train'] = []
            elif iter==1:
                print(curr_deviation)
                self.deviation['chair'] = curr_deviation
                self.splits['chair']['test'] = []
                self.annotations['chair']['test'] = []
                self.splits['chair']['val'] = []
                self.annotations['chair']['val'] = []
                self.splits['chair']['train'] = []
                self.annotations['chair']['train'] = []
            elif iter==2:
                print(curr_deviation)
                self.deviation['sofa'] = curr_deviation
                self.splits['sofa']['test'] = []
                self.annotations['sofa']['test'] = []
                self.splits['sofa']['val'] = []
                self.annotations['sofa']['val'] = []
                self.splits['sofa']['train'] = []
                self.annotations['sofa']['train'] = []
            elif iter==3:
                print(curr_deviation)
                self.deviation['swivelchair'] = curr_deviation
                self.splits['swivelchair']['test'] = []
                self.annotations['swivelchair']['test'] = []
                self.splits['swivelchair']['val'] = []
                self.annotations['swivelchair']['val'] = []
                self.splits['swivelchair']['train'] = []
                self.annotations['swivelchair']['train'] = []
            elif iter==4:
                print(curr_deviation)
                self.deviation['table'] = curr_deviation
                self.splits['table']['test'] = []
                self.annotations['table']['test'] = []
                self.splits['table']['val'] = []
                self.annotations['table']['val'] = []
                self.splits['table']['train'] = []
                self.annotations['table']['train'] = []

            cnt_ann = 0

            image_listdir = os.listdir(self.input_path[iter])
            image_listdir.sort()
            for name_image in image_listdir:
                tmp_main_name = os.path.splitext(name_image)[0]
                tmp_tail_name = int(tmp_main_name.split(sep='_')[2])
                if tmp_tail_name%10 == 1 or tmp_tail_name%10 == 2:
                    if iter==0:
                        self.splits['bed']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['bed']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==1:
                        self.splits['chair']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['chair']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==2:        
                        self.splits['sofa']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['sofa']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==3:
                        self.splits['swivelchair']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['swivelchair']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==4:
                        self.splits['table']['test'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['table']['test'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])

                elif tmp_tail_name%10 == 3 or tmp_tail_name%10 == 4:
                    if iter==0:
                        self.splits['bed']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['bed']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==1:
                        self.splits['chair']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['chair']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==2:
                        self.splits['sofa']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['sofa']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==3:
                        self.splits['swivelchair']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['swivelchair']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==4:
                        self.splits['table']['val'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['table']['val'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])

                    

                else:
                    if iter==0:
                        self.splits['bed']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['bed']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==1:
                        self.splits['chair']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['chair']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==2:
                        self.splits['sofa']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['sofa']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==3:
                        self.splits['swivelchair']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['swivelchair']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                        self.splits['all']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['all']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])
                    elif iter==4:
                        self.splits['table']['train'].append(os.path.join(self.input_path[iter], name_image))
                        self.annotations['table']['train'].append(tmp_mat['coords'][:,0:init_kp_num[iter],0,cnt_ann])

                cnt_ann += 1
            
    def process(self, fur_type):
        """Parse and process the txt dataset into a single json file compatible with coco format"""
        for phase, im_paths in self.splits[fur_type].items():  # Train and Val
            cnt_images = 0
            cnt_ann = 0
            self.initiate_json(fur_type)  # Initiate json file at each phase

            path_dir = (os.path.join(self.output_path_image[fur_type], phase))
            os.makedirs(path_dir, exist_ok=True)

            for im_path in im_paths:
                im_size, im_name, im_id = self._process_image(fur_type, im_path)
                #print(im_id)
                cnt_images += 1

                self._process_annotation(fur_type, phase, im_path, cnt_ann, im_size, im_id)
                cnt_ann += 1

                dst = os.path.join(self.output_path_image[fur_type], phase, os.path.split(im_path)[-1])
                copyfile(im_path, dst)

                if (cnt_images % 1000) == 0:
                    text = ' and copied to new directory'
                    print(f'Parsed {cnt_images} images' + text)

            self.save_json_files(phase, fur_type)
            print(f'\nPhase:{phase}')
            print(f'JSON files directory:  {self.output_path_annotation[fur_type]}')

    def save_json_files(self, phase, fur_type):
        path_json = os.path.join(self.output_path_annotation[fur_type], 'keypoint5_'+phase+'.json')
        with open(path_json,'w') as outfile:
            json.dump(self.json_file_kp5_furniture[fur_type], outfile)

    def _process_image(self, fur_type, im_path):
        """Update image field in json file"""
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0]
        tmp_prefix = im_name.split(sep='_')[1]  # Type of the image
        tmp_im_id = int(im_name.split(sep='_')[2])  # Numeric code in the image
        im_id = 0
        if tmp_prefix == 'bed':
            im_id = 1000000 + tmp_im_id
        if tmp_prefix == 'chair':
            im_id = 2000000 + tmp_im_id
        if tmp_prefix == 'sofa':
            im_id = 3000000 + tmp_im_id
        if tmp_prefix == 'swivelchair':
            im_id = 4000000 + tmp_im_id
        if tmp_prefix == 'table':
            im_id = 5000000 + tmp_im_id
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
        self.json_file_kp5_furniture[fur_type]["images"].append(dict_ann)
        return (width, height), im_name, im_id

    def _process_annotation(self, fur_type, phase, im_path, cnt_ann, im_size, im_id):
        '''Process single instance'''
        
        # Enlarge box
        kps, num = self._transform_keypoints_10(fur_type, phase, im_path, cnt_ann)
        tmp_kps = kps
        box_tight = [np.min(tmp_kps[:, 0]), np.min(tmp_kps[:, 1]),np.max(tmp_kps[:, 0]), np.max(tmp_kps[:, 1])]

        kps = list(kps.reshape((-1,)))

        w, h = box_tight[2] - box_tight[0], box_tight[3] - box_tight[1]
        x_o = max(box_tight[0] - 0.1 * w, 0)
        y_o = max(box_tight[1] - 0.1 * h, 0)
        x_i = min(box_tight[0] + 1.1 * w, im_size[0])
        y_i = min(box_tight[1] + 1.1 * h, im_size[1])
        box = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]  # (x, y, w, h)

        furniture_id = int(str(im_id) + str(0))  # include the specific annotation id

        catego_id = 0
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0]
        tmp_prefix = im_name.split(sep='_')[1]  # Type of the image

        if tmp_prefix == 'bed':
            catego_id = 1
        elif tmp_prefix == 'chair':
            catego_id = 2
        elif tmp_prefix == 'sofa':
            catego_id = 3
        elif tmp_prefix == 'swivelchair':
            catego_id = 4
        elif tmp_prefix == 'table':
            catego_id = 5 
        
        self.json_file_kp5_furniture[fur_type]["annotations"].append({
            'image_id': im_id,
            'category_id': catego_id,
            'iscrowd': 0,
            'id': furniture_id,
            'area': box[2] * box[3],
            'bbox': box,
            'num_keypoints': num,
            'keypoints': kps,
            'segmentation': []})

        

    def _transform_keypoints_10(self, fur_type, phase, im_path, cnt_ann):
        '''keypoint format transform'''
        kps_out = np.zeros((len(FURNITURE_KEYPOINTS_10), 3))
        cnt = 0
        
        file_name = os.path.basename(im_path)
        im_name = os.path.splitext(file_name)[0]
        im_type = im_name.split(sep='_')[1]  # type of furniture

        if(im_type == 'bed'):
        
            kps_out[0,0] = self.annotations[fur_type][phase][cnt_ann][0,1]
            kps_out[0,1] = self.annotations[fur_type][phase][cnt_ann][1,1]
            kps_out[0,2] = 2
            
            kps_out[1,0] = self.annotations[fur_type][phase][cnt_ann][0,6]
            kps_out[1,1] = self.annotations[fur_type][phase][cnt_ann][1,6]
            kps_out[1,2] = 2
            
            kps_out[2,0] = self.annotations[fur_type][phase][cnt_ann][0,0]
            kps_out[2,1] = self.annotations[fur_type][phase][cnt_ann][1,0]
            kps_out[2,2] = 2
            
            kps_out[3,0] = self.annotations[fur_type][phase][cnt_ann][0,5]
            kps_out[3,1] = self.annotations[fur_type][phase][cnt_ann][1,5]
            kps_out[3,2] = 2
            
            kps_out[4,0] = self.annotations[fur_type][phase][cnt_ann][0,3]
            kps_out[4,1] = self.annotations[fur_type][phase][cnt_ann][1,3]
            kps_out[4,2] = 2
            
            kps_out[5,0] = self.annotations[fur_type][phase][cnt_ann][0,8]
            kps_out[5,1] = self.annotations[fur_type][phase][cnt_ann][1,8]
            kps_out[5,2] = 2
            
            kps_out[6,0] = self.annotations[fur_type][phase][cnt_ann][0,2]
            kps_out[6,1] = self.annotations[fur_type][phase][cnt_ann][1,2]
            kps_out[6,2] = 2
            
            kps_out[7,0] = self.annotations[fur_type][phase][cnt_ann][0,7]
            kps_out[7,1] = self.annotations[fur_type][phase][cnt_ann][1,7]
            kps_out[7,2] = 2
            
            kps_out[8,0] = self.annotations[fur_type][phase][cnt_ann][0,4]
            kps_out[8,1] = self.annotations[fur_type][phase][cnt_ann][1,4]
            kps_out[8,2] = 2
            
            kps_out[9,0] = self.annotations[fur_type][phase][cnt_ann][0,9]
            kps_out[9,1] = self.annotations[fur_type][phase][cnt_ann][1,9]
            kps_out[9,2] = 2
            
            cnt = 10

        elif(im_type == 'chair'):

            kps_out[0,0] = self.annotations[fur_type][phase][cnt_ann][0,0]
            kps_out[0,1] = self.annotations[fur_type][phase][cnt_ann][1,0]
            kps_out[0,2] = 2

            kps_out[1,0] = self.annotations[fur_type][phase][cnt_ann][0,1]
            kps_out[1,1] = self.annotations[fur_type][phase][cnt_ann][1,1]
            kps_out[1,2] = 2

            kps_out[2,0] = self.annotations[fur_type][phase][cnt_ann][0,3]
            kps_out[2,1] = self.annotations[fur_type][phase][cnt_ann][1,3]
            kps_out[2,2] = 2

            kps_out[3,0] = self.annotations[fur_type][phase][cnt_ann][0,2]
            kps_out[3,1] = self.annotations[fur_type][phase][cnt_ann][1,2]
            kps_out[3,2] = 2

            kps_out[4,0] = self.annotations[fur_type][phase][cnt_ann][0,4]
            kps_out[4,1] = self.annotations[fur_type][phase][cnt_ann][1,4]
            kps_out[4,2] = 2

            kps_out[5,0] = self.annotations[fur_type][phase][cnt_ann][0,5]
            kps_out[5,1] = self.annotations[fur_type][phase][cnt_ann][1,5]
            kps_out[5,2] = 2

            kps_out[6,0] = self.annotations[fur_type][phase][cnt_ann][0,7]
            kps_out[6,1] = self.annotations[fur_type][phase][cnt_ann][1,7]
            kps_out[6,2] = 2

            kps_out[7,0] = self.annotations[fur_type][phase][cnt_ann][0,6]
            kps_out[7,1] = self.annotations[fur_type][phase][cnt_ann][1,6]
            kps_out[7,2] = 2

            kps_out[8,0] = self.annotations[fur_type][phase][cnt_ann][0,8]
            kps_out[8,1] = self.annotations[fur_type][phase][cnt_ann][1,8]
            kps_out[8,2] = 2

            kps_out[9,0] = self.annotations[fur_type][phase][cnt_ann][0,9]
            kps_out[9,1] = self.annotations[fur_type][phase][cnt_ann][1,9]
            kps_out[9,2] = 2

            cnt = 10

        elif(im_type == 'sofa'):

            kps_out[0,0] = self.annotations[fur_type][phase][cnt_ann][0,1]
            kps_out[0,1] = self.annotations[fur_type][phase][cnt_ann][1,1]
            kps_out[0,2] = 2

            kps_out[1,0] = self.annotations[fur_type][phase][cnt_ann][0,8]
            kps_out[1,1] = self.annotations[fur_type][phase][cnt_ann][1,8]
            kps_out[1,2] = 2

            kps_out[2,0] = self.annotations[fur_type][phase][cnt_ann][0,0]
            kps_out[2,1] = self.annotations[fur_type][phase][cnt_ann][1,0]
            kps_out[2,2] = 2

            kps_out[3,0] = self.annotations[fur_type][phase][cnt_ann][0,7]
            kps_out[3,1] = self.annotations[fur_type][phase][cnt_ann][1,7]
            kps_out[3,2] = 2

            kps_out[4,0] = self.annotations[fur_type][phase][cnt_ann][0,3]
            kps_out[4,1] = self.annotations[fur_type][phase][cnt_ann][1,3]
            kps_out[4,2] = 2

            kps_out[5,0] = self.annotations[fur_type][phase][cnt_ann][0,10]
            kps_out[5,1] = self.annotations[fur_type][phase][cnt_ann][1,10]
            kps_out[5,2] = 2

            kps_out[6,0] = self.annotations[fur_type][phase][cnt_ann][0,2]
            kps_out[6,1] = self.annotations[fur_type][phase][cnt_ann][1,2]
            kps_out[6,2] = 2

            kps_out[7,0] = self.annotations[fur_type][phase][cnt_ann][0,9]
            kps_out[7,1] = self.annotations[fur_type][phase][cnt_ann][1,9]
            kps_out[7,2] = 2

            kps_out[8,0] = self.annotations[fur_type][phase][cnt_ann][0,6]
            kps_out[8,1] = self.annotations[fur_type][phase][cnt_ann][1,6]
            kps_out[8,2] = 2

            kps_out[9,0] = self.annotations[fur_type][phase][cnt_ann][0,13]
            kps_out[9,1] = self.annotations[fur_type][phase][cnt_ann][1,13]
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

            kps_out[4,0] = self.annotations[fur_type][phase][cnt_ann][0,7]
            kps_out[4,1] = self.annotations[fur_type][phase][cnt_ann][1,7]
            kps_out[4,2] = 2

            kps_out[5,0] = self.annotations[fur_type][phase][cnt_ann][0,8]
            kps_out[5,1] = self.annotations[fur_type][phase][cnt_ann][1,8]
            kps_out[5,2] = 2

            kps_out[6,0] = self.annotations[fur_type][phase][cnt_ann][0,10]
            kps_out[6,1] = self.annotations[fur_type][phase][cnt_ann][1,10]
            kps_out[6,2] = 2

            kps_out[7,0] = self.annotations[fur_type][phase][cnt_ann][0,9]
            kps_out[7,1] = self.annotations[fur_type][phase][cnt_ann][1,9]
            kps_out[7,2] = 2

            kps_out[8,0] = self.annotations[fur_type][phase][cnt_ann][0,11]
            kps_out[8,1] = self.annotations[fur_type][phase][cnt_ann][1,11]
            kps_out[8,2] = 2

            kps_out[9,0] = self.annotations[fur_type][phase][cnt_ann][0,12]
            kps_out[9,1] = self.annotations[fur_type][phase][cnt_ann][1,12]
            kps_out[9,2] = 2

            cnt = 6

        elif(im_type == 'table'):
            deviation_tmp = copy.copy(self.deviation['table'])

            kps_out[0,0] = self.annotations[fur_type][phase][cnt_ann][0,4]
            kps_out[0,1] = self.annotations[fur_type][phase][cnt_ann][1,4]
            kps_out[0,2] = 2

            kps_out[1,0] = self.annotations[fur_type][phase][cnt_ann][0,5]
            kps_out[1,1] = self.annotations[fur_type][phase][cnt_ann][1,5]
            kps_out[1,2] = 2

            kps_out[2,0] = self.annotations[fur_type][phase][cnt_ann][0,6]
            kps_out[2,1] = self.annotations[fur_type][phase][cnt_ann][1,6]
            kps_out[2,2] = 2

            kps_out[3,0] = self.annotations[fur_type][phase][cnt_ann][0,7]
            kps_out[3,1] = self.annotations[fur_type][phase][cnt_ann][1,7]
            kps_out[3,2] = 2

            kps_out[4,0] = self.annotations[fur_type][phase][cnt_ann][0,0]
            kps_out[4,1] = self.annotations[fur_type][phase][cnt_ann][1,0]
            kps_out[4,2] = 2

            kps_out[5,0] = self.annotations[fur_type][phase][cnt_ann][0,1]
            kps_out[5,1] = self.annotations[fur_type][phase][cnt_ann][1,1]
            kps_out[5,2] = 2

            kps_out[6,0] = self.annotations[fur_type][phase][cnt_ann][0,2]
            kps_out[6,1] = self.annotations[fur_type][phase][cnt_ann][1,2]
            kps_out[6,2] = 2

            kps_out[7,0] = self.annotations[fur_type][phase][cnt_ann][0,3]
            kps_out[7,1] = self.annotations[fur_type][phase][cnt_ann][1,3]
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


    def initiate_json(self, fur_type):
        """
        Initiate Json for training and val phase for the 10 kp furniture skeleton
        """
        self.json_file_kp5_furniture[fur_type]["info"] = dict(url="https://github.com/openpifpaf/openpifpaf",
                                                    date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                                                 time.localtime()),
                                                    description=("Conversion of keypoint-5 dataset into MS-COCO"
                                                                 " format with 10 keypoints"))
        self.json_file_kp5_furniture[fur_type]["categories"] = [dict(name='bed',
                                                           id=1,
                                                           skeleton=FURNITURE_SKELETON_10,
                                                           supercategory='bed',
                                                           keypoints=FURNITURE_KEYPOINTS_10),
                                                           dict(name='chair',
                                                           id=2,
                                                           skeleton=FURNITURE_SKELETON_10,
                                                           supercategory='chair',
                                                           keypoints=FURNITURE_KEYPOINTS_10),
                                                           dict(name='sofa',
                                                           id=3,
                                                           skeleton=FURNITURE_SKELETON_10,
                                                           supercategory='sofa',
                                                           keypoints=FURNITURE_KEYPOINTS_10),
                                                           dict(name='swivelchair',
                                                           id=4,
                                                           skeleton=FURNITURE_SKELETON_10,
                                                           supercategory='swivelchair',
                                                           keypoints=FURNITURE_KEYPOINTS_10)]
        self.json_file_kp5_furniture[fur_type]["images"] = []
        self.json_file_kp5_furniture[fur_type]["annotations"] = []


def main():
    
    input_path_bed="./data-keypoint-5/renamed_dataset/bed/"
    input_path_chair="./data-keypoint-5/renamed_dataset/chair/"
    input_path_sofa="./data-keypoint-5/renamed_dataset/sofa/"
    input_path_swivelchair="./data-keypoint-5/renamed_dataset/swivelchair/"
    input_path_table="./data-keypoint-5/renamed_dataset/table/"
    input_path = [input_path_bed, input_path_chair, input_path_sofa, input_path_swivelchair, input_path_table]
    
    annotation_bed = "./data-keypoint-5/keypoint-5/bed/coords.mat"
    annotation_chair = "./data-keypoint-5/keypoint-5/chair/coords.mat"
    annotation_sofa = "./data-keypoint-5/keypoint-5/sofa/coords.mat"
    annotation_swivelchair = "./data-keypoint-5/keypoint-5/swivelchair/coords.mat"
    annotation_table = "./data-keypoint-5/keypoint-5/table/coords.mat"
    annotation_path = [annotation_bed, annotation_chair, annotation_sofa, annotation_swivelchair, annotation_table]

    output_path_image_bed = "./data-keypoint-5/images_bed/"
    output_path_image_chair = "./data-keypoint-5/images_chair/"
    output_path_image_sofa = "./data-keypoint-5/images_sofa/"
    output_path_image_swivelchair = "./data-keypoint-5/images_swivelchair/"
    output_path_image_table = "./data-keypoint-5/images_table/"
    output_path_image_all = "./data-keypoint-5/images_all/"
    output_path_image = {}
    output_path_image['bed'] = output_path_image_bed
    output_path_image['chair'] = output_path_image_chair
    output_path_image['sofa'] = output_path_image_sofa
    output_path_image['swivelchair'] = output_path_image_swivelchair
    output_path_image['table'] = output_path_image_table
    output_path_image['all'] = output_path_image_all

    output_path_annotation_bed = "./data-keypoint-5/annotations_bed/"
    output_path_annotation_chair = "./data-keypoint-5/annotations_chair/"
    output_path_annotation_sofa = "./data-keypoint-5/annotations_sofa/"
    output_path_annotation_swivelchair = "./data-keypoint-5/annotations_swivelchair/"
    output_path_annotation_table = "./data-keypoint-5/annotations_table/"
    output_path_annotation_all = "./data-keypoint-5/annotations_all/"
    output_path_annotation = {}
    output_path_annotation['bed'] = output_path_annotation_bed
    output_path_annotation['chair'] = output_path_annotation_chair
    output_path_annotation['sofa'] = output_path_annotation_sofa
    output_path_annotation['swivelchair'] = output_path_annotation_swivelchair
    output_path_annotation['table'] = output_path_annotation_table
    output_path_annotation['all'] = output_path_annotation_all
    kp5_to_coco = Keypoint5ToCoco(input_path, annotation_path, output_path_image, output_path_annotation)
    
    kp5_to_coco.process('bed')
    kp5_to_coco.process('chair')
    kp5_to_coco.process('sofa')
    kp5_to_coco.process('swivelchair')
    kp5_to_coco.process('table')
    kp5_to_coco.process('all')

    deviation_bed = np.zeros((len(kp5_to_coco.deviation['bed'][:,0]),10))
    deviation_bed[:,0] = kp5_to_coco.deviation['bed'][:,1]
    deviation_bed[:,1] = kp5_to_coco.deviation['bed'][:,6]
    deviation_bed[:,2] = kp5_to_coco.deviation['bed'][:,0]
    deviation_bed[:,3] = kp5_to_coco.deviation['bed'][:,5]
    deviation_bed[:,4] = kp5_to_coco.deviation['bed'][:,3]
    deviation_bed[:,5] = kp5_to_coco.deviation['bed'][:,8]
    deviation_bed[:,6] = kp5_to_coco.deviation['bed'][:,2]
    deviation_bed[:,7] = kp5_to_coco.deviation['bed'][:,7]
    deviation_bed[:,8] = kp5_to_coco.deviation['bed'][:,4]
    deviation_bed[:,9] = kp5_to_coco.deviation['bed'][:,9]
    savemat('./data-keypoint-5/annotations_bed/deviation.mat',{'bed':deviation_bed})

    deviation_chair = np.zeros((len(kp5_to_coco.deviation['chair'][:,0]),10))
    deviation_chair[:,0] = kp5_to_coco.deviation['chair'][:,0]
    deviation_chair[:,1] = kp5_to_coco.deviation['chair'][:,1]
    deviation_chair[:,2] = kp5_to_coco.deviation['chair'][:,3]
    deviation_chair[:,3] = kp5_to_coco.deviation['chair'][:,2]
    deviation_chair[:,4] = kp5_to_coco.deviation['chair'][:,4]
    deviation_chair[:,5] = kp5_to_coco.deviation['chair'][:,5]
    deviation_chair[:,6] = kp5_to_coco.deviation['chair'][:,7]
    deviation_chair[:,7] = kp5_to_coco.deviation['chair'][:,6]
    deviation_chair[:,8] = kp5_to_coco.deviation['chair'][:,8]
    deviation_chair[:,9] = kp5_to_coco.deviation['chair'][:,9]
    savemat('./data-keypoint-5/annotations_chair/deviation.mat',{'chair':deviation_chair})

    deviation_sofa = np.zeros((len(kp5_to_coco.deviation['sofa'][:,0]),10))
    deviation_sofa[:,0] = kp5_to_coco.deviation['sofa'][:,1]
    deviation_sofa[:,1] = kp5_to_coco.deviation['sofa'][:,8]
    deviation_sofa[:,2] = kp5_to_coco.deviation['sofa'][:,0]
    deviation_sofa[:,3] = kp5_to_coco.deviation['sofa'][:,7]
    deviation_sofa[:,4] = kp5_to_coco.deviation['sofa'][:,3]
    deviation_sofa[:,5] = kp5_to_coco.deviation['sofa'][:,10]
    deviation_sofa[:,6] = kp5_to_coco.deviation['sofa'][:,2]
    deviation_sofa[:,7] = kp5_to_coco.deviation['sofa'][:,9]
    deviation_sofa[:,8] = kp5_to_coco.deviation['sofa'][:,6]
    deviation_sofa[:,9] = kp5_to_coco.deviation['sofa'][:,13]
    savemat('./data-keypoint-5/annotations_sofa/deviation.mat',{'sofa':deviation_sofa})

    deviation_swivelchair = np.zeros((len(kp5_to_coco.deviation['swivelchair'][:,0]),10))
    deviation_swivelchair[:,0] = 100
    deviation_swivelchair[:,1] = 100
    deviation_swivelchair[:,2] = 100
    deviation_swivelchair[:,3] = 100
    deviation_swivelchair[:,4] = kp5_to_coco.deviation['swivelchair'][:,7]
    deviation_swivelchair[:,5] = kp5_to_coco.deviation['swivelchair'][:,8]
    deviation_swivelchair[:,6] = kp5_to_coco.deviation['swivelchair'][:,10]
    deviation_swivelchair[:,7] = kp5_to_coco.deviation['swivelchair'][:,9]
    deviation_swivelchair[:,8] = kp5_to_coco.deviation['swivelchair'][:,11]
    deviation_swivelchair[:,9] = kp5_to_coco.deviation['swivelchair'][:,12]
    savemat('./data-keypoint-5/annotations_swivelchair/deviation.mat',{'swivelchair':deviation_swivelchair})

if __name__ == "__main__":
    main()