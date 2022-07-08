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

import torch
import numpy as np
import math
import random
import torch.nn.functional as F
import pdb

from bcai_art.utils_tensor import TensorList, get_frame_channel_height_width,get_frame_shape, DATASET_TYPE_IMAGE, DATASET_TYPE_VIDEO_FIX_SIZE

MASK_CIRCLE = 'circle'
MASK_RECTANGLE = 'rectangle'


def get_rect_mask_random(patch_shape, images):
    """
    given patch_shape=[patch_h, patch_w], and images BxCxWxD (image tensor), return masks with dimension BxWxD. Values for mask is 1 or 0, representing whether the patch is visible or not. Patches will be visible at random locations, but no further transform such as scaling applied
    """
    batch_size, channel, img_h, img_w = get_frame_shape(DATASET_TYPE_IMAGE, images)
    assert len(patch_shape) == 2
    
    if torch.is_tensor(images):
        masks = torch.zeros(batch_size, img_h,img_w)
        h_mins = np.random.randint(img_h-patch_shape[0],size= batch_size)
        w_mins = np.random.randint(img_w-patch_shape[1],size= batch_size)
        for ii in range(batch_size):
            masks[ii,h_mins[ii]:h_mins[ii]+patch_shape[0], w_mins[ii]:w_mins[ii]+patch_shape[1]] = 1
        
        return masks
    else:
        masks = []
        for ii in range(batch_size):
            
            channel,img_h,img_w = get_frame_channel_height_width(images[ii])
            masks.append(torch.zeros(img_h,img_w))
            h_min = random.randint(0,img_h-patch_shape[0]-1)
            w_min = random.randint(0,img_w-patch_shape[1]-1)
            masks[ii][h_min:h_min+patch_shape[0], w_min:w_min+patch_shape[1]] = 1

        masks = TensorList(masks)

        return masks
        
    
def circle_mask(shape, sharpness = 40):
    """Return a circular mask of a given shape.

    :param shape:       a circle shape, he last two dimensions must be equal
    :param sharpness:   a sharpness parameter.
    :return: the mask of a given shape.
    """
    assert shape[1] == shape[2], "circle_mask received a bad shape: " + shape

    diameter = shape[1]  
    x = np.linspace(-1, 1, diameter)
    y = np.linspace(-1, 1, diameter)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx**2 + yy**2) ** sharpness

    mask = 1 - np.clip(z, -1, 1)
    mask = np.expand_dims(mask, axis=0)
    mask = np.broadcast_to(mask, shape).astype(np.float32)
    return mask


def rect_mask(shape):
    """Return a rectangular mask of a given shape.

    :param shape: mask shape.
    :return: the mask of a given shape.
    """
    assert shape[1] == shape[2], "mask received a bad shape: " + shape
    mask = np.ones((shape[1], shape[2]))
    mask = np.expand_dims(mask, axis=0)
    mask = np.broadcast_to(mask, shape).astype(np.float32)
    return mask


def get_mask(shape, kind):
    if kind == MASK_CIRCLE:
        return circle_mask(shape)
    elif kind == MASK_RECTANGLE:
        return rect_mask(shape)
    else:
        raise Exception('Unsupported mask type: ' + MASK_CIRCLE)


def random_transform(masks, patches, scale_min, scale_max, rotate_max, aspect_ratio, X):
    """Generate and apply a random transform to a patch

    :param masks:
    :param patches:     patches
    :param scale_min:   a minimum random scale factor.
    :param scale_max:   a maximum random scale factor.
    :param rotate_max:  a maximum random rotation in degrees.
    :return:
    """
    assert len(masks) == len(patches)

    _, patch_h, patch_w = get_frame_channel_height_width(patches)

    tform_mat_batch = []
    centroid_batch = []
    ALIGN_CORNERS = True
    flag_resize = False

    for i in range(len(masks)):
        _, img_h, img_w = get_frame_channel_height_width(X[i])
        t_mat, centroid, flag_resize = get_random_tform_matrix(scale_min, scale_max, rotate_max, aspect_ratio,
                                                               patch_w=patch_w,
                                                               patch_h=patch_h,
                                                               img_w=img_w,
                                                               img_h=img_h,
                                                               flag_resize=flag_resize)
        tform_mat_batch.append(t_mat)
        centroid_batch.append(centroid)

    tform_tensor = torch.FloatTensor(np.array(tform_mat_batch)).to(device=masks.device)

    grid = F.affine_grid(tform_tensor, masks.size(),
                         align_corners=ALIGN_CORNERS)

    masks = F.grid_sample(masks, grid, align_corners=ALIGN_CORNERS)
    patches = F.grid_sample(patches, grid, align_corners=ALIGN_CORNERS)

    if flag_resize:
        masks_l = []
        patches_l = []

        for ii in range(len(patches)):
            patch_resized, mask_resized = rescale_patch_mask(patches[ii], masks[ii], X[ii])
            patches_l.append(patch_resized)
            masks_l.append(mask_resized)

        return TensorList(masks_l), TensorList(patches_l), centroid_batch

    return masks, patches, centroid_batch


def scale_and_pad(in_tensor, scale, pad):
    in_tensor = in_tensor.unsqueeze(0)
    scaled = F.interpolate(in_tensor,
                           scale_factor=scale,
                           # the interpretation of the None recompute_scale_factor will change
                           # in the future warnings. So setting recompute_scale_factor to False
                           # makes sure the scale factor is not recomputed in the future.
                           # It also makes the respective warning disappear.
                           recompute_scale_factor=False)
    padded = F.pad(scaled, pad)
    padded = padded.squeeze(0)
    
    return padded


def get_random_padding(pad):
    rand_pad = np.random.uniform(0, pad)
    return (int(rand_pad), pad-int(rand_pad))


def rescale_patch_mask(patch, mask, img):
    _, patch_h, patch_w = get_frame_channel_height_width(patch)
    _, img_h, img_w = get_frame_channel_height_width(img)

    scale = min(img_h / patch_h, img_w / patch_w)

    pad_w = get_random_padding(img_w - int(patch_w * scale))
    pad_h = get_random_padding(img_h - int(patch_h * scale))
    # Padding for the last dimension goes first,
    # see https://pytorch.org/docs/stable/nn.functional.html?highlight=functional%20pad#torch.nn.functional.pad
    pad_dims = pad_w + pad_h
    return scale_and_pad(patch, scale, pad_dims), scale_and_pad(mask, scale, pad_dims)


def area2xy_scale(scale, aspect_ratio):
    x_scale = math.sqrt(scale/aspect_ratio)
    y_scale = aspect_ratio * x_scale
    
    return x_scale, y_scale


def get_random_tform_matrix(scale_min, scale_max, rotate_max, aspect_ratio, patch_w, patch_h, img_w, img_h, flag_resize):
    """Generate a random transform matrix.

    :param scale_min:   a minimum random scale factor.
    :param scale_max:   a maximum random scale factor.
    :param rotate_max:  a maximum random rotation in degrees.
    :param patch_w:     a patch width.
    :param patch_h:     a patch height.
    :param img_w:       an image width.
    :param img_h:       an image height.

    :return:
    """
   
    random_rotation = np.random.uniform(-rotate_max, rotate_max) * np.pi / 180.
     
    random_scale = np.random.uniform(scale_min, scale_max)

    if not (img_w == patch_w and img_h == patch_h):
        flag_resize = True
        rescale =  (img_w*img_h) / (patch_w*patch_h)            
        random_scale = random_scale * rescale
    
    random_x_scale, random_y_scale = area2xy_scale(random_scale, aspect_ratio)
    x_padding_after_scaling = ((1 - random_x_scale) * patch_w) / 2.
    y_padding_after_scaling = ((1 - random_y_scale) * patch_h) / 2.

    random_x_delta = np.random.uniform(-x_padding_after_scaling, x_padding_after_scaling)
    random_y_delta = np.random.uniform(-y_padding_after_scaling, y_padding_after_scaling)

    centroid = [int((1/2* patch_w - random_x_delta)), int((1/2* patch_h-random_y_delta))] 

    # scale translations to -1,1 (due to how translation works with affine_grid)
    random_x_delta /= (patch_w/2.)
    random_y_delta /= (patch_h/2.)

    # rotation matrix
    rotate_mat = np.eye(3)
    rotate_mat[:2,:2] = np.array([
      [np.cos(random_rotation), -np.sin(random_rotation)],
      [np.sin(random_rotation), np.cos(random_rotation)]
    ])

    # scale matrix
    scale_mat = np.eye(3)
    scale_mat[0, 0] = 1. / random_x_scale
    scale_mat[1, 1] = 1. / random_y_scale
    scale_mat[2, 2] = 1.

    # translation matrix
    translate_mat = np.eye(3)
    translate_mat[:2,2] = np.array([random_x_delta, random_y_delta])

    # build transformation matrix
    tform_mat = scale_mat.dot(rotate_mat.dot(translate_mat))

    # keep only the top two rows
    return tform_mat[:2,:], centroid, flag_resize


def compute_ablation_area(im_size, block_size, pos):
    x_end = pos[0] + block_size[0]
    y_end = pos[1] + block_size[1]
    if x_end < im_size[0] and y_end < im_size[1]:
        return [((pos[0], pos[1]), (x_end, y_end))]
    elif x_end >= im_size[0] and y_end < im_size[1]:
        return [((pos[0], pos[1]) , (im_size[0], y_end)) , ((0, pos[1]) , (x_end - im_size[0], y_end))]
    elif x_end < im_size[0] and y_end >= im_size[1]: 
        return [((pos[0], pos[1]) , (x_end, im_size[1])) , ((pos[0], 0) , (x_end, y_end - im_size[1]))]
    else:
        return [((pos[0], pos[1]) , (im_size[0], im_size[1])) , ((0 , 0) , (x_end - im_size[0], y_end - im_size[1])),
               ((0, pos[1]) , (x_end - im_size[0], im_size[1])) , ((pos[0], 0) , (im_size[0], y_end - im_size[1]))]


def derandomized_ablate_batch(X, block_size, pos='random', reuse_noise = True):
    assert not (pos !='random' and reuse_noise == False)
    ablated_half1 = torch.zeros_like(X)
    ablated_half2 = torch.zeros_like(X)
    if reuse_noise:
        if pos == 'random':
            pos = (random.randint(0, X.shape[2]-1), random.randint(0, X.shape[3]-1))
        for area in compute_ablation_area((X.shape[2], X.shape[3]), block_size, pos):
            ablated_half1[:, :, area[0][0]:area[1][0], area[0][1]:area[1][1]] = X[:,:, area[0][0]:area[1][0], area[0][1]:area[1][1]]
            ablated_half2[:, :, area[0][0]:area[1][0], area[0][1]:area[1][1]] = 1.0 - X[:,:, area[0][0]:area[1][0], area[0][1]:area[1][1]]
    else:
        for i in range(X.shape[0]):
            pos = (random.randint(0, X.shape[2]-1), random.randint(0, X.shape[3]-1))
            for area in compute_ablation_area((X.shape[2], X.shape[3]), block_size, pos):
                ablated_half1[i, :, area[0][0]:area[1][0], area[0][1]:area[1][1]] = X[i, :, area[0][0]:area[1][0], area[0][1]:area[1][1]]
                ablated_half2[i, :, area[0][0]:area[1][0], area[0][1]:area[1][1]] = 1.0 - X[i, :, area[0][0]:area[1][0], area[0][1]:area[1][1]]
    return torch.cat((ablated_half1, ablated_half2), 1)


def gen_crops(imgTensor, num_crops, crops_size_x, crops_size_y, position_embedding = None, combine = False, num_samples = 1):
    if len(imgTensor.shape) == 4:
        datatype = DATASET_TYPE_IMAGE
    elif len(imgTensor.shape) == 5:
        datatype = DATASET_TYPE_VIDEO_FIX_SIZE
    else:
        raise TypeError("input tensor dimension should be 4 or 5")
    #cat_img = torch.cat(num_crops*[imgTensor[None,:]])
    
    batch, channel, w, h = get_frame_shape(datatype, imgTensor)
    cat_img = torch.repeat_interleave(imgTensor, num_crops, 0)
    if crops_size_x >= w and crops_size_y >= h:
        if position_embedding is None:
            return cat_img
        else:
            return cat_img, torch.repeat_interleave(position_embedding, num_crops, 0)
    
    idx_x = torch.randint(w-crops_size_x, (num_crops*batch,))
    idx_y = torch.randint(h-crops_size_y, (num_crops*batch,))
    all_idx_x = idx_x[:,None] + np.arange(crops_size_x)
    all_idx_y = idx_y[:,None] + np.arange(crops_size_y)
    
    if datatype == DATASET_TYPE_IMAGE:
        return cat_img[np.arange(num_crops*batch)[:,None,None,None],np.arange(channel)[None,:,None,None], all_idx_x[:,None,:,None], all_idx_y[:,None,None,:]]
    
    if datatype == DATASET_TYPE_VIDEO_FIX_SIZE:
        if position_embedding is None:
            return cat_img[np.arange(num_crops*batch)[:,None,None,None,None],np.arange(imgTensor.shape[1])[None,:,None,None,None],np.arange(channel)[None,None,:,None,None], all_idx_x[:,None,None,:,None], all_idx_y[:,None,None,None,:]]
        else:
            #pe is always emb_lenxwxh
            emb_len, _,_ = position_embedding.shape
            batch_pe = torch.unsqueeze(position_embedding, 0).repeat(batch,1,1,1)
            cat_pe = torch.repeat_interleave(batch_pe, num_crops, 0)
            return cat_img[np.arange(num_crops*batch)[:,None,None,None,None],np.arange(imgTensor.shape[1])[None,:,None,None,None],np.arange(channel)[None,None,:,None,None], all_idx_x[:,None,None,:,None], all_idx_y[:,None,None,None,:]],cat_pe[np.arange(num_crops*batch)[:,None,None,None],np.arange(emb_len)[None,:,None,None], all_idx_x[:,None,:,None], all_idx_y[:,None,None,:]]


def gen_crops_resize(imgTensor, num_crops, crops_size_x, crops_size_y, crop_size_x_scaled, crop_size_y_scaled, combine = False, num_samples = 1):
    import torch.nn as nn
    
    crop_tensor = gen_crops(imgTensor, num_crops, crops_size_x, crops_size_y,combine,num_samples)
    crop_tensor = nn.functional.interpolate(crop_tensor, (crop_size_x_scaled,crop_size_y_scaled),mode='bilinear')

    return crop_tensor
