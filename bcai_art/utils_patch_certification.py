#
# BCAI ART : Bosch Center for AI Adversarial Robustness Toolkit
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import random
import math
import scipy.special
import torch
import numpy as np
from bcai_art.utils_patch import area2xy_scale

class CertificationProb:
    """
        Given class_ratio, crop size, patch_scale (as in percentage of image area), width and height of an image,
        and if the patch is placed at random location or the worst location, return the probability that this image
        can be certifiably robust under randomized crop.

        class_ratio is a 1xN array of float suming up to 1, where N is the number of classes for the classfication task.
         ith item in class_ratio is the percentage of crops of the image be classified as the ith class.

       get_certification_prob is the main function for computing certification probability
    """
    def __init__(self,patch_scale, crop_width, crop_height, num_crops, img_width = None, img_height = None, aspect_ratio = 1, random_location=True):
        self.aspect_ratio = aspect_ratio
        self.patch_scale = patch_scale
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.num_crops = num_crops
        
        if (not img_width is None) or (not img_height is None):
            self.set_img_sizes(img_width, img_height)
        self.random_location = random_location
        self.random_draw_time = 100
        
    def set_img_sizes(self, img_width, img_height):
        flag_change_img_size = False
        
        try:
            flag_change_img_size = not ((self.img_width == img_width) and (self.img_height == img_height))
        except:
            flag_change_img_size = True
        
        if flag_change_img_size:
            patch_x_scale, patch_y_scale = area2xy_scale(self.patch_scale, self.aspect_ratio)
            self.patch_width = int(patch_x_scale*img_width)
            self.patch_height = int(patch_y_scale*img_height)
            print(self.patch_width, self.patch_height)
            self.img_width = img_width
            self.img_height = img_height
            self.pa = self.get_pa()
            self.precompute_combintorial_pa()
        
        
    
    def get_certification_prob(self,class_ratio):
        class_ratio_1, class_1 = torch.max(class_ratio, dim=0)
        tmp_ratio = class_ratio_1
        class_ratio[class_1] = min(class_ratio - 1)
        class_ratio_2, class_2 = torch.max(class_ratio, dim=0)
        
        class_ratio[class_1] = tmp_ratio
        assert class_ratio_1 >0 and class_ratio_2>=0
        
        class_diff = math.floor((class_ratio_1 - class_ratio_2)*self.num_crops/2) + 1
        
        #print(class_diff)
        if not self.random_location:
            #return self.get_cert_prob_given_pa(self.pa, class_diff, self.num_crops)
            return sum(self.pa_combintorial[:class_diff]), class_diff
        
        p_cert = 0.0
        
        pa_values = np.sum(self.pa_combintorial[:, :class_diff], axis=1)
        #print(pa_values)
        #pdb.set_trace()
        for ii in range(self.random_draw_time):
            
            p_cert = p_cert + random.choices(pa_values, self.pa[1])[0]
            
            #p_cert = p_cert + self.get_cert_prob_given_pa(pa, class_diff, self.num_crops)
        
        #print(av_pa/self.random_draw_time)
        
        return p_cert/float(self.random_draw_time), class_diff
    
    def precompute_combintorial_pa(self):
        try:
            pa=self.pa
            
        except:
            print("this function should be called after pa is set")
            return
        
        if self.random_location:
            possible_pa = self.pa[0]
            pa_cnt = len(possible_pa)
            self.pa_combintorial = np.zeros((pa_cnt, self.num_crops + 1))
            
            
            for ii in range(pa_cnt):
                self.pa_combintorial[ii] = self.combintorial_pa_given(possible_pa[ii])
                
            
        else:
            self.pa_combintorial = self.combintorial_pa_given(pa)


    def combintorial_pa_given(self, pa):
        pa_combintorial = np.zeros(self.num_crops + 1)       
        if pa < 0.5:
            start_idx = 0
            end_idx = self.num_crops +1
            step = 1
            start_prob = (1-pa)**self.num_crops
        else:
            start_idx = self.num_crops
            end_idx = -1
            step = -1
            start_prob = (pa)**self.num_crops
        
        if start_prob == 0.0:
            raise Exception("fail to compute combintorial numbers")
        
        ratio = pa/(1-pa)
        pa_combintorial[start_idx] = start_prob
        
        for ii in range(start_idx+step, end_idx, step):
            if pa < 0.5:                
                facto = ratio * (self.num_crops - ii + 1) / ii
            else:
                facto = (ii+1) /(self.num_crops - ii) /ratio
                
            pa_combintorial[ii] = pa_combintorial[ii-step] * facto
        return pa_combintorial

    def get_cert_prob_given_pa(self,pa, class_diff, num_crops):

        p_cert = 0.0
        for ii in range(class_diff + 1):
            p_cert = p_cert + scipy.special.comb(num_crops, ii) * (pa**ii) * (1-pa)**(num_crops - ii)

        return p_cert

    def get_pa(self):
        n_all = (self.img_width - self.patch_width + 1)*(self.img_height - self.patch_height + 1)
        
        if not self.random_location:
            #pa is deterministic
            n_adv = (min(self.img_width - self.crop_width +1, self.crop_width + self.patch_width - 1))*(min(self.img_height - self.crop_height +1, self.crop_height + self.patch_height - 1))
            
            return float(n_adv)/float(n_all)
        
        #pa is a distribution, n_adv is a dict
        n_adv = {}
        overlap_width = 0
        overlap_height = 0
        
        for ii in range(self.img_width - self.patch_width +1):
            for jj in range(self.img_height - self.patch_height +1):
                #(ii,jj) is the left and top boundary of patch
                
                overlap_width = self.get_overlap(ii, self.img_width, self.crop_width, self.patch_width)
                overlap_height = self.get_overlap(jj, self.img_height, self.crop_height, self.patch_height)
                n_adv_instance = overlap_width * overlap_height
                
                if n_adv_instance not in n_adv:
                    n_adv[n_adv_instance] = 1
                    continue
                
                n_adv[n_adv_instance] = n_adv[n_adv_instance] + 1
        
        pa_values = []
        pa_prob = []
        max_prob = 0.0
        max_prob_instance = -1
        for n_adv_instance, n_adv_cnt in n_adv.items():
            pa_values.append(float(n_adv_instance)/float(n_all))
            prob_tmp = float(n_adv_cnt)/float(n_all)
            pa_prob.append(prob_tmp)
            
            if prob_tmp > max_prob:
                max_prob = prob_tmp
                max_prob_instance =n_adv_instance

        pa_prob[-1] = 1 - sum(pa_prob[:-1])
        
        return [pa_values, pa_prob]
                
                    
    def get_overlap(self,patch_boundary, img_len, crop_len, patch_len):
        
        if patch_boundary >= crop_len:
            
            if patch_boundary <= img_len - patch_len - crop_len:
                #not on boundaries
                overlap_len = patch_len + crop_len -1
            else:
                overlap_len = img_len - patch_boundary
        else:
            overlap_len = patch_boundary + patch_len

        return overlap_len
                   


