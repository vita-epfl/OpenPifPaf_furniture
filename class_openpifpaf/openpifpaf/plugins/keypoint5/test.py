from scipy.io import loadmat
import numpy as np
import os
import time
from shutil import copyfile
import json
from PIL import Image

tmp_mat_bed = loadmat("./data-keypoint-5/keypoint-5/bed/coords.mat")
a = tmp_mat_bed['coords'][:,0,0,0]
b = tmp_mat_bed['coords'][:,0,1,0]
c = tmp_mat_bed['coords'][:,0,2,0]
print(a)
print(b)
print(c)
m = (a + b + c)/3
print(m)
d = pow((pow(a[0]-m[0],2)+pow(a[1]-m[1],2)+pow(b[0]-m[0],2)+pow(b[1]-m[1],2)+pow(c[0]-m[0],2)+pow(c[1]-m[1],2))/3,0.5)
print(d)
