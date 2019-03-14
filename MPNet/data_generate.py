# Format of the csv file
# 0-255: histogram
# 256-257: label
# 258-261: crop_box
# 262: Num_pre
# 263: gamm alue
import os
import cv2
import csv
import random
import numpy as np
from skimage import  exposure,io
from PIL import Image


file_path = './data/train/'
csv_file = './data/train.csv'
sample_begin = 0

file_name = os.listdir(file_path)
n_sample = len(file_name)
k = 0
ith_sample = 0

with open(csv_file,'w') as f:
    f_csv = csv.writer(f)
    for filename in file_name:
        k = k +1
        print(k, filename)
        im = cv2.imread(file_path + filename)
        name = filename.split('.')[0]
        for i in range(10):
            imname = name + str(i) + '.jpg'

            [w, d, c] = np.shape(im)
            w1 = random.randint(0, int(w / 4.0))
            w2 = random.randint(int(w * 3.0 / 4), w)
            d1 = random.randint(0, int(d / 4.0))
            d2 = random.randint(int(d * 3.0 / 4), d)
            im_cut = im[w1:w2, d1: d2]
            cv2.imwrite(save_p1 + imname, im_cut)

            im_reload = cv2.imread(save_p1 + imname)
            if len(np.shape(im_reload)) == 3:
                im_cut_h = im_reload[:, :, 0]
            [h, bins] = np.histogram(im_cut_h, bins=256, normed=True)

            write_line = h.tolist() + [0, 1, w1, w2, d1, d2, 0, 0, filename]
            f_csv.writerow(write_line)
            ith_sample += 1

            Num_pre = random.randint(1, 2)
            # 1 for gamma correction, 2 for histogram equalization
            if Num_pre == 1:
                gamma_value= random.uniform(0.4, 2.1)
                im_eh = exposure.adjust_gamma(im_cut, gamma_value)
            else:
                im_eh = np.zeros(np.shape(im_cut))
                for i in range(3):
                    im_eh[:,:,i] = cv2.equalizeHist(im_cut[:,:,i])
            cv2.imwrite(save_p2 + imname, im_eh)
            im_eh_h = cv2.imread(save_p2 + imname)

            if len(np.shape(im_eh_h)) == 3:
                im_eh_h = im_eh_h[:, :, 0]
            [h_eh, bins_eh] = np.histogram(im_eh_h, bins=256, normed=True)

            if Num_pre ==1:
                write_line = h_eh.tolist() + [1, 0] + [w1, w2, d1, d2, 1, gamma_value] + [filename]
                f_csv.writerow(write_line)
                ith_sample += 1
            else:
                write_line = h_eh.tolist() + [1, 0] + [w1, w2, d1, d2, 2, 0] + [filename]
                f_csv.writerow(write_line)
                ith_sample += 1
