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

import numpy as np
import torch

# Image data
DATASET_TYPE_IMAGE = "image"
# Video data split into fixed-size chunks
DATASET_TYPE_VIDEO_FIX_SIZE = "video_chunk_fixed_size"
# Audio split into fixed-size chunks
DATASET_TYPE_AUDIO_FIX_SIZE = "audio_chunk_fixed_size"

DEFAULT_TOL=1e-100

START_RANDOM = 'random'
START_ZERO = 'zero'

def get_device_list():
    """return:   a list of available devices"""
    return ['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]


def clear_device_cache(device_name):
    """clear device cache unless the device is CPU."""
    if device_name.lower() == 'cpu':
        return
    with torch.cuda.device(device_name):
        #print('Clearing cache on:', device_name)
        torch.cuda.empty_cache()


def get_batched(tensor, batch_size):
    """Torch expand might be more efficient as it doesn't copy memory.
       -1 means that the dim size remains unchanged.
    """
    return tensor.expand([batch_size] + [-1] * len(tensor.shape))


def get_frame_channel_height_width(X):
    """Just a tiny wrapper to obtain image dimensions (and # of channels)
       from a tensor generator in ToTensor-compatible way, i.e.,
       whose last three dimensions are exactly: channel, height, width.

        :param  X:     tensor
        :return a tuple  (channel number, height, width)

    """
    x_shape = X.shape
    assert len(x_shape) >= 3

    return x_shape[-3], x_shape[-2], x_shape[-1]


def get_frame_shape(dataset_type, X):
    """Computes shape of the frames in the image or video data.
       It *expects* data to have the batch dimension!
       If x is a TensorList, then width and height are going to be None.

    :param dataset_type: dataset type
    :param X:            tensor
    :return: a tuple: (batch, channel number, height, width)
    """
    if dataset_type == DATASET_TYPE_IMAGE:
        batch_size, channel, h, w = X.shape
    elif dataset_type == DATASET_TYPE_VIDEO_FIX_SIZE:
        batch_size, _, channel, h, w = X.shape
    else:
        raise Exception('Unsupported dataset type: ' + dataset_type)

    return batch_size, channel, h, w


def clamp(X, lower_limit, upper_limit):
    """
    Clamp a given tensor to a specified range
    """
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize_idx(idx):
    """Make the indexing interface polymorphic,
       so lists and ranges and tensors can be
       processed in the same fashion.
       Important, currently, singleton indices
       remain such, it is not clear what is a canonical
       way of dealing with singleton indices

      :param  idx:   an index that can be tensor, numpy, list, range, tuple, or integer.
      :return  a normalized index that is either a list or an integer.
    """
    if isinstance(idx, slice):
        step = idx.step
        if step is None:
            step = 1
        idx = range(idx.start, idx.stop, step)

    if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
        idx = idx.tolist()
    elif isinstance(idx, tuple) or isinstance(idx, range):
        idx = list(idx)

    return idx


def get_last_data_dim(is_matr):
    return (-2, -1) if is_matr else (-1)


def project(x, eps, norm_p,
            is_matr,
            tol=DEFAULT_TOL):
    """
    Project `x` on the L_p norm ball of size `eps`.

    :param  x: input tensor
    :type   x: `torch.tensor`
    :param  eps: Maximum norm allowed.
    :type   eps: `float`
    :param  norm_p: L_p norm to use for clipping. A positive integer value or `np.Inf`.
    :type   norm_p: `int` or np.inf
    :param  is_matr:        if True, the norm is computed using the last
                            two dimensions. Oherwise, we use only one last dimension.
    :type    `bool'
    :return: Values of `x` after projection.
    :rtype: `torch.tensor`
    """
    if norm_p == np.inf:
       return torch.clamp(x, -eps, eps)

    # If the norm is smaller than a given number we don't have to do anything
    x_norm = torch.norm(x, p=norm_p, dim=get_last_data_dim(is_matr), keepdim=True)
    scale_by_norm = torch.max(x_norm, torch.ones_like(x_norm) * eps)

    return x * (eps / (scale_by_norm + tol))


def get_start_delta(X, start, epsilon, norm_p,
                    requires_grad,
                    is_matr,
                    tol=DEFAULT_TOL):
    """Get starting delta/perturbation that is guaranteed
       to be within a ball of a given radius for a given norm.
       Sampling is stratified by the value of the norm,
       but not random uniform.

    :param X:               a tensor for which delta is computed.
    :param start:           a start type (START_RANDOM or START_ZERO)
    :param epsilon:         the radius of (attack) ball
    :param norm_p:          a norm
    :param is_matr:         True, if a single-channel data is 2-dim and False otherwise.
                            For two-dimensional data the norm is computed using the last
                            two dimensions.
    :param requires_grad:   does the result needs gradients?

    :return: a delta of the same shape as X, which is within a given ball.
    """

    if start == START_RANDOM:
        # This is a random uniform sampling in the LINF ball
        delta = 2 * epsilon * torch.rand_like(X) - epsilon

        # Let's actually rescal eval for LINF norm as well as otherwise
        # norm values tend to concentrate close to one

        # First make sure it sits on a eps-radius sphere
        start_norm = torch.norm(delta, p=norm_p, dim=get_last_data_dim(is_matr), keepdim=True)
        delta *= (epsilon / (start_norm + tol))
        # Then scale norms by randomly selected uniform numbers
        # It's important to have the same shape as the norm tensor itself:
        # we want the random number to be the same (if all but last dimensions
        # are kept fixed)
        delta *= torch.rand_like(start_norm)
    else:
        delta = torch.zeros_like(X)

    delta.requires_grad = requires_grad

    return delta


def get_abs_max_batched(x):
    """Return the maximum absolute  while reducing all dimensions except the first (batch) dimension.

    :param  x: input tensor

    :return a tensor of the shape B, where element K is the maximum absolute value accross all dimensions in batch element K.
    """
    B = x.shape[0]
    x = x.view(B, -1)
    return torch.max(torch.abs(x), -1)[0]


def get_max_norm_batched(x, norm_p, is_matr):
    """

    :param x:        input tensor
    :param norm_p:   a norm
    :param is_matr:  True, if a single-channel data is 2-dim and False otherwise.
                     For two-dimensional data the norm is computed using the last
                     two dimensions.

    :return a tensor of the shape B. Each
    """
    x_norm = torch.norm(x, p=norm_p, dim=get_last_data_dim(is_matr), keepdim=False)

    return get_abs_max_batched(x_norm)


def torch_solve(comp_obj, x_start, is_matr, y,
                loss_class, learn_rate,
                max_iters,
                optim_class=torch.optim.Adam,
                max_avg_loss=None,
                proj_eps=None, proj_norm=None,
                debug=False):
    """Trying to solve an equation f(x) = y using gradient descent and a PyTorch optimizer.
       The only optimizer parameters supported are the learning rate, but it's easy to change.

    :param comp_obj:        a (differentiable) function or an object with the overloaded ca
    :param x_start:         a starting pint
    :param is_matr:         this is needed only for the projection (see the comment for the function project).
    :param y:               y in f(x) = y
    :param loss_class:      a loss class
    :param learn_rate:      a learning rate
    :param optim_class:     an optimizer class
    :param max_iters:       a maximum number of iterations to make
    :param max_avg_loss:    an optional value of the (average) loss to continue iteration, if the average loss
                            becomes smaller than this threshold, the algorithm terminates.
    :param proj_eps:        an optional radius of a projection ball
    :param proj_norm:       an optional projection norm
    :param debug:           true to print the debug information
    :return:
    """
    # This procedure is inspired by the code in this tweet by Antonio Valerio Miceli Barone.
    # However, the actual implementation is quite different (in several important aspects).
    # https://twitter.com/AVMiceliBarone/status/1394733995099893767

    loss_obj = loss_class(reduction='mean')

    x_curr = torch.clone(x_start).detach()
    optimizer = optim_class([x_curr], lr=learn_rate)
    x_curr.requires_grad = True

    def closure():
        avg_loss = loss_obj(comp_obj(x_curr), y)

        optimizer.zero_grad()
        avg_loss.backward(retain_graph=True)
        return avg_loss

    for i in range(max_iters):
        avg_loss = optimizer.step(closure)

        if debug:
           print(f'solver iter: {i} loss: {avg_loss}')
        if max_avg_loss is not None:
            if avg_loss < max_avg_loss:
                break

        if proj_eps is not None:
            assert proj_norm is not None
            # Modify data directly or else, autograd will go crazy
            x_curr.data = project(x_curr.detach(), eps=proj_eps, norm_p=proj_norm, is_matr=is_matr)

    return x_curr.detach()


def apply_func_squeeze(func, obj, *args, **kwargs):

    if torch.is_tensor(obj):
        return func(obj, *args, **kwargs)
    elif type(obj) == TensorList:
        return TensorList([func(tensor.unsqueeze(0), *args, **kwargs).squeeze(0) for tensor in obj])
    else:
        raise TypeError("Inavlid Type "+type(obj))


def apply_func(func, obj, *args, **kwargs):
    """Apply function to an object that can be a tensor or a TensorList.

    :param func:   a function to apply
    :param obj:    an object (tensor or TensorList)
    :param args:   non-keyword function arguments
    :param kwargs: keyword function arguments
    :return:
    """

    if torch.is_tensor(obj):
        return func(obj, *args, **kwargs)
    elif type(obj) == TensorList:
        return TensorList([func(tensor, *args, **kwargs) for tensor in obj])
    else:
        raise TypeError("Unsupported object type " + type(obj))


def assert_property(obj, prop_name, nt):
    if torch.is_tensor(obj):
        if nt:
            assert not getattr(obj, prop_name)
        else:
            assert getattr(obj, prop_name)
    elif type(obj)==TensorList:
        for tensor in obj:
            if nt:
                assert not getattr(tensor, prop_name)
            else:
                assert getattr(tensor, prop_name)
    else:
        raise TypeError("Invalid Type "+type(obj))


class TensorList(object):
    """This is a utility class that wraps a list of Tensor, and then enables a limited set of tensor-like
       operations on the list" 
    """
    dumb_tensor = torch.tensor(1.0)

    def __init__(self, inner_list):
        """
        Constractor
        :param inner_list:      a list of tensors.
        """
        self.inner_list = list(inner_list)
        self.shape = (len(inner_list),3,None,None)

    def size(self, dim=None):
        """
        Return shape of list
        :param dim:      return size of the given dim only.
        """
        if dim == None:
            return self.shape
        else:
            return self.shape[dim]

    def iterative_operator(self, others):
        has_innerlist = getattr(others,"inner_list", False )
        
        match_tensor_shape = False
        if torch.is_tensor(others):
            match_tensor_shape = (len(self.inner_list) == others.shape[0])
            
            for ii in range(len(self.inner_list)):
                match_tensor_shape = match_tensor_shape and (self.inner_list[ii].shape == others[0].shape)
            
        return  (has_innerlist or match_tensor_shape)
        
    def __getattr__(self, name):
        """
        Retrieve a property or a function to call on all the tensors inside the inner list.
        Only a limited set of properties or functions is supported but it could be extended.
        :param name: name of the property or function
        """
        if name in ['sign', 'detach', 'grad', 'to']:
            x= getattr(TensorList.dumb_tensor, name)
            if callable(x):
                return lambda *args, **kwargs: self.apply_on_all(name,*args, **kwargs)
            else:
                return TensorList([getattr(tensor, name) for tensor in self.inner_list])
        else:
            raise AttributeError(name)
    
    def __setattr__(self, name, value):
        """
        set a property on all the tensors inside the inner list.
        Only a limited set of properties or functions is supported but it could be extended.
        :param name: name of the property or function
        :param value: value to set
        """
        if name in ['requires_grad']:
            for tensor in self.inner_list:
                setattr(tensor, name, value)
        else:
            self.__dict__[name] = value

    def apply_on_all(self, method, *args, **kwargs):
        """
        call a function by its name on all the tensors inside the inner list given the args.
        :param method: name of the property or function
        """
        if isinstance(method, str): 
            return TensorList([getattr(tensor, method)(*args, **kwargs) for tensor in self.inner_list])
        else:
            raise TypeError("Inavlid Type "+type(method))

    def __add__(self, others):
        """
        add two list of tensors by adding the corresponding elements (from left)
        :param others: another TensorList
        """
        return TensorList([tensor+other for tensor,other in zip(self.inner_list,others)])

    def __radd__(self, others):
        """
        add two list of tensors by adding the corresponding elements (from right)
        :param others: another TensorList
        """
        return TensorList([tensor+other for tensor,other in zip(self.inner_list,others)])

    def __sub__(self, others):
        """
        subtract two list of tensors by subtractinh the corresponding elements (from left)
        :param others: another TensorList
        """
        return TensorList([tensor-other for tensor,other in zip(self.inner_list,others)])

    def __rsub__(self, others):
        """
        subtract two list of tensors by subtracting the corresponding elements (from right)
        :param others: another TensorList
        """

        if self.iterative_operator(others):
            return TensorList([other-tensor for tensor,other in zip(self.inner_list,others)])
     
        return TensorList([others-tensor for tensor in self.inner_list])
    
    def __mul__(self, mult):
        """
        multiply two list of tensors by multiplying the corresponding elements (from left)
        :param mult: another TensorList
        """
        
        return TensorList([tensor*mult_i for tensor,mult_i in zip(self.inner_list, mult.inner_list)])

    def __rmul__(self, mult):
        """
        multiply two list of tensors by multiplying the corresponding elements (from right)
        :param mult: another TensorList
        """
               
        if self.iterative_operator(mult):
            return TensorList([tensor*mult_i for tensor,mult_i in zip(self.inner_list, mult)])
        
        return TensorList([tensor*mult for tensor in self.inner_list])

    def __getitem__(self, index):
        """
        retrieve a tensor given its index
        :param index: index
        """
        return self.inner_list[index]
    
    def __setitem__(self, index, value):
        """
        set a tensor given its index
        :param index: index
        :param val: value to set
        """
        self.inner_list[index]= value
    
    def __iter__(self):
        """
        enable iterating over the list
        """
        yield from self.inner_list

    def __len__(self):
        """
        return the length of the list
        """
        return len(self.inner_list)


class DictTensorList:
    """
        This is a utility class that wraps a list of Dictionaries, whose values are tensors.
        It enable operations on such collections of tensors.
    """
    def __init__(self, inner_list):
        """
        Constractor
        :param inner_list:      a list of dictionaries.
        """
        self.inner_list = list(inner_list)
        self.shape = (len(inner_list),)
        self.device = self.inner_list[0]['boxes'].device if len(self.inner_list)>0 else 'cpu'

    def __repr__(self):
        return '[' + ','.join([str(e) for e in self.inner_list]) + ']'

    def to(self, device):
        """
        set the device to the specified device.
        :param device: destination 
        """
        return DictTensorList([{k: v.to(device) for k, v in t.items()} for t in self.inner_list])

    def size(self, dim=None):
        """
        Return shape of list
        :param dim:      return size of the given dim only.
        """
        if dim == None:
            return self.shape
        else:
            return self.shape[dim]

    def __len__(self):
        """
        return the length of the list
        """
        return len(self.inner_list)

    def __iter__(self):
        """
        enable iterating over the list
        """
        yield from self.inner_list

    def __getitem__(self, index):
        """
        retrieve a tensor given its index
        :param index: index
        """
        return self.inner_list[index]

