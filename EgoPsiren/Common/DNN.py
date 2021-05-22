import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#from Common import DataReader #should import remaprange


def RemapRange (value, low1, high1, low2, high2):
  return low2 + (value - low1) * (high2 - low2) / (high1 - low1)


def CountParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    # print(type(y))
    # print(type(x))
    # print('y', y.shape)
    # print('x', x.shape)
    if grad_outputs is None:
        #print('in')
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


LATENT_DIMENSION = 64

# https://github.com/Fdevmsy/PyTorch-Soft-Argmax/blob/master/soft-argmax.py
def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert voxels.dim()==5
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0 
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0)
    if voxels.get_device() >= 0:
        indices_kernel.to(voxels.get_device())
    indices_kernel = indices_kernel.view((H,W,D))
    conv = soft_max*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices%D
    y = (indices/D).floor()%W
    x = (((indices/D).floor())/W).floor()%H
    coords = torch.stack([x,y,z],dim=2)
    return coords

# https://github.com/MWPainter/cvpr2019/blob/master/stitched/soft_argmax.py
def _make_radial_window(width, height, cx, cy, fn, window_width=10.0):
    """
    Returns a grid, where grid[i,j] = fn((i**2 + j**2)**0.5)
    :param width: Width of the grid to return
    :param height: Height of the grid to return
    :param cx: x center
    :param cy: y center
    :param fn: The function to apply
    :return:
    """
    # The length of cx and cy is the number of channels we need
    channels = cx.size(0)

    # Explicitly tile cx and cy, ready for computing the distance matrix below, because pytorch doesn't broadcast very well
    # Make the shape [channels, height, width]


    cx = cx.repeat(height, width, 1).permute(2, 0, 1)
    cy = cy.repeat(height, width, 1).permute(2, 0, 1)

    # Compute a grid where dist[i,j] = (i-cx)**2 + (j-cy)**2, need to view and repeat to tile and make shape [channels, height, width]
    xs = torch.arange(width).view((1, width)).repeat(channels, height, 1).float()
    ys = torch.arange(height).view((height, 1)).repeat(channels, 1, width).float()
    
    if(cx.get_device() >= 0):
        xs = xs.to(cx.get_device())
        ys = ys.to(cx.get_device())

    delta_xs = xs - cx
    delta_ys = ys - cy
    dists = torch.sqrt((delta_ys ** 2) + (delta_xs ** 2))

    # apply the function to the grid and return it
    return fn(dists, window_width)


def _parzen_scalar(delta, width):
    """For reference"""
    del_ovr_wid = math.abs(delta) / width
    if delta <= width/2.0:
        return 1 - 6 * (del_ovr_wid ** 2) * (1 - del_ovr_wid)
    elif delta <= width:
        return 2 * (1 - del_ovr_wid) ** 3


def _parzen_torch(dists, width):
    """
    A PyTorch version of the parzen window that works a grid of distances about some center point.
    See _parzen_scalar to see the 
    :param dists: The grid of distances
    :param window: The width of the parzen window
    :return: A 2d grid, who's values are a (radial) parzen window
    """
    hwidth = width / 2.0
    del_ovr_width = dists / hwidth

    near_mode = (dists <= hwidth/2.0).float()
    in_tail = ((dists > hwidth/2.0) * (dists <= hwidth)).float()

    return near_mode * (1 - 6 * (del_ovr_width ** 2) * (1 - del_ovr_width)) \
        + in_tail * (2 * ((1 - del_ovr_width) ** 3))


def _uniform_window(dists, width):
    """
    A (radial) uniform window function
    :param dists: A grid of distances
    :param width: A width for the window
    :return: A 2d grid, who's values are 0 or 1 depending on if it's in the window or not
    """
    hwidth = width / 2.0
    return (dists <= hwidth).float()


def _identity_window(dists, width):
    """
    An "identity window". (I.e. a "window" which when multiplied by, will not change the input).
    """
    return torch.ones(dists.size())



class SoftArgmax1D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        """
        super(SoftArgmax1D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax(x) = \sum_i (i * softmax(x)_i)
        :param x: The input to the soft arg-max layer
        :return: Output of the soft arg-max layer
        """
        smax = self.softmax(x)
        end_index = self.base_index + x.size()[1] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size)
        return torch.matmul(smax, indices)


class SoftArgmax2D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1, window_fn=None, window_width=10, softmax_temp=1.0):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 3D tensors (so a 4D tensor).
        For input shape (B, C, W, H), we apply softmax across the W and H dimensions.
        We use a softmax, over dim 2, expecting a 3D input, which is created by reshaping the input to (B, C, W*H)
        (This is necessary because true 2D softmax doesn't natively exist in PyTorch...
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        :param window_function: Specify window function, that given some center point produces a window 'landscape'. If
            a window function is specified then before applying "soft argmax" we multiply the input by a window centered
            at the true argmax, to enforce the input to soft argmax to be unimodal. Window function should be specified
            as one of the following options: None, "Parzen", "Uniform"
        :param window_width: How wide do we want the window to be? (If some point is more than width/2 distance from the
            argmax then it will be zeroed out for the soft argmax calculation, unless, window_fn == None)
        """
        super(SoftArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=2)
        self.softmax_temp = softmax_temp
        self.window_type = window_fn
        self.window_width = window_width
        self.window_fn = _identity_window
        if window_fn == "Parzen":
            self.window_fn = _parzen_torch
        elif window_fn == "Uniform":
            self.window_fn = _uniform_window



    def _softmax_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.
        :param x: A 4D tensor of shape (B, C, W, H) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H)) / temp
        x_softmax = self.softmax(x_flat)
        return x_softmax.view((B, C, W, H))


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))
        :param x: The input to the soft arg-max layer
        :return: Output of the 2D soft arg-max layer, x_coords and y_coords, in the shape (B, C, 2), which are the soft
            argmaxes per channel
        """
        # Compute windowed softmax
        # Compute windows using a batch_size of "batch_size * channels"
        batch_size, channels, height, width = x.size()
        argmax = torch.argmax(x.view(batch_size * channels, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, width).float(), torch.floor(torch.div(argmax.float(), float(width)))
        windows = _make_radial_window(width, height, argmax_x, argmax_y, self.window_fn, self.window_width)
        windows = windows.view(batch_size, channels, height, width)
        if(x.get_device() >= 0):
            windows = windows.to(x.get_device())
        smax = self._softmax_2d(x, self.softmax_temp) * windows
        smax = smax / torch.sum(smax.view(batch_size, channels, -1), dim=2).view(batch_size,channels,1,1)

        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size)
        if(x.get_device() >= 0):
            x_indices = x_indices.to(x.get_device())
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)

        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size)
        if(x.get_device() >= 0):
            y_indices = y_indices.to(x.get_device())
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)

        # For debugging (testing if it's actually like the argmax?)
        # argmax_x = argmax_x.view(batch_size, channels)
        # argmax_y = argmax_y.view(batch_size, channels)
        # print("X err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_x - x_coords))))
        # print("Y err in soft argmax: {err}".format(err=torch.mean(torch.abs(argmax_y - y_coords))))

        # Put the x coords and y coords (shape (B,C)) into an output with shape (B,C,2)
        return torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)


class SineLayer(nn.Module):
  # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
  
  # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
  # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
  # hyperparameter.
  
  # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
  # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
  
  def __init__(self, in_features, out_features, bias=True,
                is_first=False, omega_0=30):
    super().__init__()
    self.omega_0 = omega_0
    self.is_first = is_first
    
    self.in_features = in_features
    self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    self.init_weights()
  
  def init_weights(self):
    with torch.no_grad():
      if self.is_first:
          self.linear.weight.uniform_(-1 / self.in_features, 
                                        1 / self.in_features)      
      else:
          self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                        np.sqrt(6 / self.in_features) / self.omega_0)
      
  def forward(self, input):
    return torch.sin(self.omega_0 * self.linear(input))
  
  def forward_with_intermediate(self, input): 
    # For visualization of activation distributions
    intermediate = self.omega_0 * self.linear(input)
    return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
  def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                first_omega_0=30, hidden_omega_0=30.):
    super().__init__()
    
    self.net = []
    self.net.append(SineLayer(in_features, hidden_features, 
                              is_first=True, omega_0=first_omega_0))

    for i in range(hidden_layers):
      self.net.append(SineLayer(hidden_features, hidden_features, 
                                is_first=False, omega_0=hidden_omega_0))

    if outermost_linear:
      final_linear = nn.Linear(hidden_features, out_features)
      
      with torch.no_grad():
        final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                      np.sqrt(6 / hidden_features) / hidden_omega_0)
          
      self.net.append(final_linear)
    else:
      self.net.append(SineLayer(hidden_features, out_features, 
                                is_first=False, omega_0=hidden_omega_0))
    
    self.net = nn.Sequential(*self.net)
  
  def forward(self, coords):
    coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
    output = self.net(coords)
    return output, coords        

  def forward_with_activations(self, coords, retain_grad=False):
    '''Returns not only model output, but also intermediate activations.
    Only used for visualizing activations later!'''
    activations = OrderedDict()

    activation_count = 0
    x = coords.clone().detach().requires_grad_(True)
    activations['input'] = x
    for i, layer in enumerate(self.net):
      if isinstance(layer, SineLayer):
        x, intermed = layer.forward_with_intermediate(x)
        
        if retain_grad:
          x.retain_grad()
          intermed.retain_grad()
            
        activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
        activation_count += 1
      else: 
        x = layer(x)
        
        if retain_grad:
          x.retain_grad()
              
      activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
      activation_count += 1

    return activations

class ConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution, latent_dimension = 256):
        super().__init__()
        self.sub_latent_dimension = latent_dimension//2
        self.initial_kernel = 21

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, self.sub_latent_dimension, self.initial_kernel, 1, self.initial_kernel//2)
        self.relu = nn.ReLU(inplace=True)
        self.latent_dimension = latent_dimension

        self.cnn = nn.Sequential(
            nn.Conv2d(self.sub_latent_dimension, self.latent_dimension, 5, 1, 2),
            nn.ReLU(),
            Conv2dResBlock(self.latent_dimension, self.latent_dimension),
            #Conv2dResBlock(256, 256),
            #Conv2dResBlock(256, 256),
            #Conv2dResBlock(256, 256),
            nn.Conv2d(self.latent_dimension, self.latent_dimension, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)

        fc_dim = image_resolution[0] * image_resolution[1]
        self.fc = nn.Linear(fc_dim, 1) #TODO: Why 1??

        self.image_resolution = image_resolution

    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.cnn(o)
        intermediate = self.relu_2(o).view(o.shape[0], self.latent_dimension, -1)
        o = self.fc(intermediate).squeeze(-1)
        return o


class ConvImgIntensity(nn.Module):
    def __init__(self, channel, image_resolution, latent_dimension = 256):
        super().__init__()
        self.sub_latent_dimension = latent_dimension//2
        self.initial_kernel = 21

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, self.sub_latent_dimension, self.initial_kernel, 1, self.initial_kernel//2)
        self.relu = nn.ReLU(inplace=True)
        self.latent_dimension = latent_dimension
        self.sig = nn.Sigmoid()

        self.cnn = nn.Sequential(
            nn.Conv2d(self.sub_latent_dimension, self.latent_dimension, 5, 1, 2),
            nn.ReLU(),
            Conv2dResBlock(self.latent_dimension, self.latent_dimension),
            #Conv2dResBlock(256, 256),
            #Conv2dResBlock(256, 256),
            #Conv2dResBlock(256, 256),
            nn.Conv2d(self.latent_dimension, 1, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)

        fc_dim = image_resolution[0] * image_resolution[1]
        self.fc = nn.Linear(fc_dim, 1) #TODO: Why 1??

        self.image_resolution = image_resolution

    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.cnn(o)
        o = self.sig(o)
        #intermediate = self.relu_2(o).view(o.shape[0], self.latent_dimension, -1)
        #o = self.fc(intermediate).squeeze(-1)
        return o


class Conv2dResBlock(nn.Module):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''
    def __init__(self, in_channel, out_channel=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            nn.ReLU()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output


#def channel_last(x):
#    return x.transpose(1, 2).transpose(2, 3)

from collections import OrderedDict
import re

#from collections import OrderedDict

def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)

class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.
    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param


class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                    '(inheriting from `nn.Module`), or a `MetaModule`. '
                    'Got type: `{0}`'.format(type(module)))
        return input
    



class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output



########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


############################
# Initialization schemes
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)



class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, outermost_sigmoid=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        elif not outermost_sigmoid:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))
        else:
            snl, _, _ = nls_and_inits[nonlinearity]
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), snl
            ))


        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations



class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        '''
        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                                 outermost_linear=False, nonlinearity='tanh')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)
        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class SirenMM(MetaModule):
    '''A sinusoidal representation network.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        #if self.mode == 'rbf':
        #    self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
        #    in_features = kwargs.get('rbf_centers', 1024)
        #elif self.mode == 'nerf':
        #    self.positional_encoding = PosEncodingNeRF(in_features=in_features,
        #                                               sidelength=kwargs.get('sidelength', None),
        #                                               fn_samples=kwargs.get('fn_samples', None),
        #                                               use_nyquist=kwargs.get('use_nyquist', True))
        #    in_features = self.positional_encoding.out_dim

        #self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
        #                                            downsample=kwargs.get('downsample', False))
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        # various input processing methods for different applications
        #if self.image_downsampling.downsample:
        #    coords = self.image_downsampling(coords)
        #if self.mode == 'rbf':
        #    coords = self.rbf_layer(coords)
        #elif self.mode == 'nerf':
        #    coords = self.positional_encoding(coords)

        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

class ConvolutionalNeuralProcessImplicit2DHypernet(nn.Module):
    def __init__(self, in_features, out_features, image_resolution=None, partial_conv=False):
        super().__init__()
        latent_dim = LATENT_DIMENSION

        if partial_conv:
            self.encoder = PartialConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        else:
            self.encoder = ConvImgEncoder(channel=in_features, image_resolution=image_resolution, latent_dimension = latent_dim)
        
        self.hypo_net = SirenMM(in_features=2,out_features=out_features,num_hidden_layers=3,hidden_features=32,outermost_linear=True,nonlinearity='sine')  #Siren(in_features=2, out_features=out_features, #modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution, in_features=2)
        
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False



class ConvolutionalNeuralProcessImplicit2DHypernetWithMultiplier(nn.Module):
    def __init__(self, in_features, out_features, image_resolution=None, partial_conv=False):
        super().__init__()
        latent_dim = LATENT_DIMENSION

        if partial_conv:
            self.encoder = PartialConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        else:
            self.encoder = ConvImgEncoder(channel=in_features, image_resolution=image_resolution, latent_dimension = latent_dim)
        
        self.hypo_net = SirenMM(in_features=2,out_features=out_features,num_hidden_layers=3,hidden_features=32,outermost_linear=True,nonlinearity='sine')  #Siren(in_features=2, out_features=out_features, #modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution, in_features=2)
        
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)

        self.multiplier_net = ConvImgIntensity(channel=in_features, image_resolution=image_resolution, latent_dimension = latent_dim)

        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        siren_output = self.hypo_net(model_input, params=hypo_params)

        intensity = self.multiplier_net(model_input['img_sparse'])
        intensity = torch.flatten(intensity,start_dim=2)
        intensity = torch.transpose(intensity,1,2)

        intensity =  intensity*.9 + .1 # clamping
        

        siren = -nn.Sigmoid()(-siren_output['model_out'])*.9+.1 # clamping




        model_output = siren * intensity



        return {'model_in': siren_output['model_in'], 'model_out':model_output, 'siren_out': siren_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params, 'intensity': intensity}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False




class ConvolutionalAutoEncoderToPath(nn.Module):
    def __init__(self, in_features, path_length, image_resolution=None, partial_conv=False):
        super().__init__()
        self.latent_dim = LATENT_DIMENSION
        out_features = 2 * path_length

        if partial_conv:
            self.encoder = PartialConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        else:
            self.encoder = ConvImgEncoder(channel=in_features, image_resolution=image_resolution, latent_dimension = self.latent_dim)
        
        self.decoder =  FCBlock(in_features=self.latent_dim//2, out_features=out_features, num_hidden_layers=3,hidden_features=128,outermost_linear=True,nonlinearity='relu')  #Siren(in_features=2, out_features=out_features, #modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution, in_features=2)
        
        #self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
        #                              hypo_module=self.hypo_net)

        #self.multiplier_net = ConvImgIntensity(channel=in_features, image_resolution=image_resolution, latent_dimension = latent_dim)



        print(self)

    def reparameterize(self, mu, logVar):
        std=torch.exp(logVar/2)
        eps=torch.randn_like(std)
        return mu + std * eps


    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        #hypo_params = self.hyper_net(embedding)

        mu = embedding[:,:self.latent_dim//2]
        
        logVar = embedding[:,self.latent_dim//2:]

        z = self.reparameterize(mu,logVar)

        model_output = self.decoder(z)
        model_output = torch.reshape(model_output,(model_output.shape[0],2,-1))


        return {'model_in': model_input, 'model_out':model_output, 'latent_vec': embedding,
                'mu': mu, 'logVar': logVar, 'sample':z}




def train2(network,  data_generators, loss_functions, optimizer, epoch):
  network.train() #updates any network layers that behave differently in training and execution 
  avg_loss = 0
  num_batches = 0

  data_generator_A = data_generators[0]
  data_generator_B = data_generators[1]
  loss_function_A  = loss_functions[0]
  loss_function_B  = loss_functions[1]

  #print('training3')
  data_generator_iterator = iter(data_generator_B)
  for i, (input_data_A, target_output_A) in enumerate(data_generator_A):
    try:
      (input_data_B, target_output_B) = next(data_generator_iterator)
    except StopIteration:
      data_generator_iterator = iter(data_generator_B)
      (input_data_B, target_output_B) = next(data_generator_iterator)

    if True:
        for key, val in input_data_A.items():
            val = val.cuda()
            input_data_A[key] = val
        target_output_A = target_output_A.cuda()

    if True:
        for key, val in input_data_B.items():
            val = val.cuda()
            input_data_B[key] = val
        target_output_B = target_output_B.cuda()

    #print(i)
    optimizer.zero_grad()                               # Gradients need to be reset each batch
    prediction_A = network(input_data_A)             # Forward pass: compute the output class given a image
    prediction_B = network(input_data_B)
    #zero = torch.tensor(np.zeros(target_output.shape, dtype=np.float), dtype=torch.float32)
    loss_A = loss_function_A(prediction_A, target_output_A, epoch)   # Compute the loss: difference between the output and correct result
    loss_B = loss_function_B(prediction_B, target_output_B, epoch)   # Compute the loss: difference between the output and correct result
    loss_A.backward()                                     # Backward pass: compute the gradients of the model with respect to the loss
    loss_B.backward()                                     # Backward pass: compute the gradients of the model with respect to the loss
    #print('optimizing')
    optimizer.step()
    avg_loss += loss_A.item() + loss_B.item()

    num_batches += 1
  return avg_loss/num_batches


def test_with_grad2(network, test_loaders, loss_functions, epoch):
  #print('hit test?!?!?')
  network.eval() #updates any network layers that behave differently in training and execution 
  test_loss = 0
  num_batches = 0

  test_loader_A = test_loaders[0]
  test_loader_B = test_loaders[1]
  loss_function_A = loss_functions[0]
  loss_function_B = loss_functions[1]

  data_generator_iterator = iter(test_loader_B)
  #with torch.no_grad():
  for (data_A, target_A) in test_loader_A:

    try:
      (data_B, target_B) = next(data_generator_iterator)
    except StopIteration:
      data_generator_iterator = iter(test_loader_B)
      (data_B, target_B) = next(data_generator_iterator)
      
    if True:
        for key, val in data_A.items():
            val = val.cuda()
            data_A[key] = val
        target_A = target_A.cuda()

    if True:
        for key, val in data_B.items():
            val = val.cuda()
            data_B[key] = val
        target_B = target_B.cuda()

    output_A = network(data_A)
    output_B = network(data_B)
    #zero = torch.tensor(np.zeros(target.shape, dtype=np.float), dtype=torch.float32)
    test_loss += loss_function_A(output_A, target_A, epoch).item() + loss_function_B(output_B, target_B, epoch).item()
    num_batches += 1
  test_loss /= num_batches
  #print('\nTest set: Avg. loss: {:.4f})\n'.format(test_loss))
  return test_loss








def train(network,  data_generator, loss_function, optimizer, epoch):
  network.train() #updates any network layers that behave differently in training and execution 
  avg_loss = 0
  num_batches = 0
  #print('training')
  for i, (input_data, target_output) in enumerate(data_generator):
    #print(i)
    if True:
        for key, val in input_data.items():
            val = val.cuda()
            input_data[key] = val
        #where_wrong_way2 = np.where(target_output.numpy()[0,0,:,1] > 0.00001)
        #if len(where_wrong_way2[0]) > 0 :
        ##print(where_wrong_way2[0].shape)
        #    print(where_wrong_way2[0])
        if (type(target_output) is dict):
            for key, val in target_output.items():
                val = val.cuda()
                target_output[key] = val
        else:
            target_output = target_output.cuda()

    

    optimizer.zero_grad()                               # Gradients need to be reset each batch
    prediction = network(input_data)             # Forward pass: compute the output class given a image
    #zero = torch.tensor(np.zeros(target_output.shape, dtype=np.float), dtype=torch.float32)
    loss = loss_function(prediction, target_output, epoch)   # Compute the loss: difference between the output and correct result
    loss.backward()                                     # Backward pass: compute the gradients of the model with respect to the loss
    #print('optimizing')
    optimizer.step()
    avg_loss += loss.item()

    num_batches += 1
  return avg_loss/num_batches

def test_with_grad(network, test_loader, loss_function, epoch):
  #print('hit test??')
  network.eval() #updates any network layers that behave differently in training and execution 
  test_loss = 0
  num_batches = 0
  #with torch.no_grad():
  for data, target in test_loader:
    if True:
        for key, val in data.items():
            val = val.cuda()
            data[key] = val
            
        if (type(target) is dict):
            for key, val in target.items():
                val = val.cuda()
                target[key] = val
        else:
            target = target.cuda()
        #target = target.cuda()
    output = network(data)
    #zero = torch.tensor(np.zeros(target.shape, dtype=np.float), dtype=torch.float32)
    test_loss += loss_function(output, target, epoch).item()
    num_batches += 1
  test_loss /= num_batches
  #print('\nTest set: Avg. loss: {:.4f})\n'.format(test_loss))
  return test_loss





def graphLoss(epoch_counter, train_loss_hist, test_loss_hist, start = 0):
  fig = plt.figure()
  plt.plot(epoch_counter[start:], train_loss_hist[start:], color='blue')
  plt.plot(epoch_counter[start:], test_loss_hist[start:], color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('#Epochs')
  plt.ylabel('MSE')



def logResults(epoch, num_epochs, train_loss, train_loss_history, test_loss, test_loss_history, epoch_counter, print_interval=1):
  if (epoch%print_interval == 0):  print('Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f' %(epoch+1, num_epochs, train_loss, test_loss))
  train_loss_history.append(train_loss)
  test_loss_history.append(test_loss)
  epoch_counter.append(epoch)


def trainAndGraph(network, training_generator, testing_generator, loss_function, optimizer, num_epochs, learning_rate, output_path, overfit_output_path, logging_rate=1, train = train, test = test_with_grad, graph = True):
  
  #print('training and graphing')
  #Arrays to store training history
  test_loss_history = []
  epoch_counter = []
  train_loss_history = []
  
  best_test_loss = np.inf
  best_train_loss = np.inf

  for epoch in range(num_epochs):                           
    avg_loss = train(network, training_generator, loss_function, optimizer, epoch)
    if avg_loss < best_train_loss:
        torch.save(network, overfit_output_path)
        best_train_loss = avg_loss
    test_loss = test(network, testing_generator, loss_function, epoch)
    if test_loss < best_test_loss:
        torch.save(network, output_path)
        best_test_loss = test_loss

    logResults(epoch, num_epochs, avg_loss, train_loss_history, test_loss, test_loss_history, epoch_counter, logging_rate)
      
  print('Best test loss: {:.10f}'.format(best_test_loss))
  #Run the model on the input data for analysis of the results and error
  #with torch.no_grad():
  #  predicted = network(feature_tensor)
  if graph:
    graphLoss(epoch_counter, train_loss_history, test_loss_history)

  #saveState(network,optimizer)





trainAndGraphDerivative2 = lambda net, train_gen, test_gen, loss_fn, opt, n_epochs, lr, log_rate = 1 : trainAndGraph(net, train_gen, test_gen, loss_fn, opt, n_epochs, lr, log_rate, train2, test_with_grad2)



trainAndGraphDerivative = lambda net, train_gen, test_gen, loss_fn, opt, n_epochs, lr, outputPath, overfit_outputPath, log_rate = 1, verbose = False : trainAndGraph(net, train_gen, test_gen, loss_fn, opt, n_epochs, lr, outputPath, overfit_outputPath, log_rate, train, test_with_grad, graph = verbose)







def gradients_mse(model_outputs, coords, gt_gradients, epoch):
    # print('mo',model_outputs.shape)
    # print('co',coords.shape)
    # print('gt',gt_gradients.shape)
    # compute gradients on the model

    #print(gt_gradients[0,0,0,:])

    gradients = gradient(model_outputs, coords)
    #laplacians = laplace(model_outputs, coords)
    
    #laplacians_loss = torch.nn.SmoothL1Loss()(laplacians,torch.zeros(laplacians.shape).cuda())#torch.mean((torch.abs(laplacians)).sum(-1))
    
    
    # compare them with the ground-truth
   # gt_grads = torch.ones(gradients.shape)
    #gt_grads[0,0,:,0] = 0
    #gt_grads[0,0,:,1] = -1
    gradients_loss = torch.nn.SmoothL1Loss()(gradients,gt_gradients) #torch.mean((gradients - gt_gradients).pow(2).sum(-1))
    #if epoch > 350:
    return 1 * gradients_loss #+ .001 * laplacians_loss
    #return 0 * gradients_loss

#gradients_mse_with_coords = lambda preds,gt,epoch: gradients_mse(preds[0],preds[1], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling
gradients_mse_with_coords = lambda preds,gt,epoch: gradients_mse(preds['model_out'],preds['model_in'], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling




#current_epoch = 0




def laplacian_mse(model_outputs, coords, gt_laplacian, epoch):


    laplacians = laplace(model_outputs, coords)
    gradients = gradient(model_outputs, coords)
    #gt_laplacians = -1*torch.ones(laplacians.shape)
    #laplacians_loss = torch.mean((laplacians - gt_laplacian).pow(2).sum(-1)) # laplacians - 0
    #laplacians_loss = torch.mean((torch.max(-laplacians)).pow(2).sum(-1))
    #laplacians_loss = torch.mean((torch.nn.ReLU()(laplacians)).sum(-1))
    
    #big_laplacians = torch.where(torch.abs(laplacians) < float(0.2), 1.0, float(0.0))
    #laplacians *= big_laplacians
    #gt_laplacian *= big_laplacians
    laplacians_loss = torch.nn.SmoothL1Loss()(laplacians,gt_laplacian)#torch.mean((torch.abs(laplacians)).sum(-1))
    
    relued = torch.nn.ReLU()(model_outputs)

    value_loss = torch.nn.SmoothL1Loss()(relued,torch.zeros(model_outputs.shape).cuda())
    value_loss2 = torch.nn.SmoothL1Loss()(model_outputs,torch.zeros(model_outputs.shape).cuda())

    #relued = torch.nn.ReLU()(laplacians)
    #laplacians_loss = torch.nn.SmoothL1Loss()(relued,gt_laplacian)


    grad_norms = torch.norm(gradients,dim=3).cuda()


    #gradients_loss = torch.mean((grad_norms - torch.ones(grad_norms.shape)).pow(2).sum(-1))
    gradients_loss = torch.nn.SmoothL1Loss()(grad_norms,torch.ones(grad_norms.shape).cuda())
    #gradients_loss = torch.nn.L1Loss()(grad_norms,torch.ones(grad_norms.shape))
    #laplacians_loss = torch.mean(torch.sigmoid(torch.nn.ReLU()(-laplacians))).sum(-1)
    
    #laplacians_loss = torch.mean(torch.sigmoid(5*(laplacians))).sum(-1)

    #print('coord shape:',coords.shape)
    
    #ys = torch.mean(torch.nn.ReLU()(-gradients[0,0,:,1]).pow(2).sum(-1))
    #xs = torch.mean(gradients[0,0,:,1].pow(2).sum(-1))

    #minus1 = gradients - torch.ones(gradients.shape)

    ##ys = torch.mean(torch.abs(minus1[0,0,:,1]).sum(-1))
    #gt_grads = torch.ones(gradients.shape)
    #gt_grads[0,0,:,0] = 0
    #gt_grads[0,0,:,1] = -1
    ##gt_grads = gt_grads.cuda()
    ##ys = torch.mean((gradients - gt_grads).pow(2).sum(-1))
    #ys = torch.nn.SmoothL1Loss()(gradients, gt_grads)

    #return 1 * .001 * laplacians_loss + 0.1 * gradients_loss# + 0 * ys #+ 0 * .01 * gradients_loss #.1 * 0.5 * (np.cos(epoch*.2)+1)* laplacians_loss + 0.0001 * 0.5 * (np.sin(epoch * .2)+1) * gradients_loss # 1 * ys  # +
    return 0.0 * 0.001 * laplacians_loss + 1.0 * value_loss + .1 * value_loss2

#laplacian_mse_with_coords = lambda preds,gt,epoch: laplacian_mse(preds[0],preds[1], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling
laplacian_mse_with_coords = lambda preds,gt,epoch: laplacian_mse(preds['model_out'],preds['model_in'], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling


def auto_encoder_loss(model_outputs_dict, model_input, gt_value, epoch ):
    
    mu = model_outputs_dict['mu']
    logVar = model_outputs_dict['logVar']
    
    kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())

    path_loss = torch.nn.L1Loss()(model_outputs_dict['model_out'],gt_value)
    loss = kl_divergence + path_loss
    #print('\tkl:', kl_divergence.item())
    #print('\tpath:' , path_loss.item())
    #print('\tgt max:' , torch.max(gt_value).item())
    #print('\tgt min:' , torch.min(gt_value).item())
    #print('\tpred max:' , torch.max(model_outputs_dict['model_out']).item())
    #print('\tpred min:' , torch.min(model_outputs_dict['model_out']).item())
    print('')
    
    return loss

auto_encoder_loss_with_coords = lambda preds,gt,epoch: auto_encoder_loss(preds,preds['model_in'], gt,epoch)

def value_mse(model_outputs_dict, coords, gt_value_dict, epoch, dim=(196,196)):
    #model_outputs = torch.squeeze(model_outputs,-1)
    #multiplier = torch.zeros(coords.shape)

    mask = model_outputs_dict['intensity']
   
    regularizer_batch = torch.mean(mask,dim=[1,2])
    regularizer = torch.mean(regularizer_batch)/20.0

    batch_size = model_outputs_dict['model_out'].shape[0]
    prediction = torch.reshape(model_outputs_dict['model_out'],(batch_size,1,dim[0],dim[1]))
    #resultA = model_outputs_dict['model_out'].get_device()
    #resultB = prediction.get_device()
    pred_goals = RemapRange(SoftArgmax2D(window_fn="Parzen")(-prediction),0,196,-1,1)
    goals = torch.unsqueeze(gt_value_dict['goal'],1)


    multiplier = torch.unsqueeze(torch.exp(0.5*(coords[:,:,1] + 1)),-1)
    #test = torch.mean(multiplier)/10 # maybe equal to above, but just to be safe using the two step version
    goal_loss = torch.nn.MSELoss()(pred_goals,goals)
    implicit_field_loss = torch.nn.L1Loss()(model_outputs_dict['model_out'] * multiplier, gt_value_dict['field'] * multiplier)
    value_loss = 0.1 * goal_loss + implicit_field_loss + regularizer
    #print("\tintensity loss:", regularizer.item())
    #print("\tgoal loss:", goal_loss.item())
    #print("\timplicit field loss:", implicit_field_loss.item())
    #print("")
    return value_loss

#def value_mse(model_outputs_dict, coords, gt_value, epoch):
#    #model_outputs = torch.squeeze(model_outputs,-1)
#    #multiplier = torch.zeros(coords.shape)

#    raw_mask = model_outputs_dict['intensity']
#    mask =  raw_mask*.9 + .1
#    regularizer_batch = torch.mean(raw_mask,dim=[1,2])
#    regularizer = torch.mean(regularizer_batch)/20.0

#    raw_siren = model_outputs_dict['siren_out']
#    siren = torch.relu(-raw_siren)+.1
#    #model_outputs = torch.relu(-model_outputs_dict['model_out']) + .1 # SIREN values should be between -inf and 0, although initially there could be positive values. This is essentially a ReLU
#    multiplier = torch.unsqueeze(torch.exp(0.5*(coords[:,:,1] + 1)),-1)
#    #test = torch.mean(multiplier)/10 # maybe equal to above, but just to be safe using the two step version
#    value_loss = torch.nn.L1Loss()(-1 * mask * siren * multiplier, gt_value * multiplier) + regularizer
#    print("intensity loss:", regularizer)
#    return value_loss

#laplacian_mse_with_coords = lambda preds,gt,epoch: laplacian_mse(preds[0],preds[1], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling
value_mse_with_coords = lambda preds,gt,epoch: value_mse(preds,preds['model_in'], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling





#laplacian_mse_with_coords = lambda preds,gt,epoch: laplacian_mse(preds['model_out'],preds['model_in'], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling



def old_value_mse(model_outputs_dict, coords, gt_value, epoch):
    #model_outputs = torch.squeeze(model_outputs,-1)
    #multiplier = torch.zeros(coords.shape)


    model_outputs = model_outputs_dict['model_out']
    multiplier = torch.unsqueeze(torch.exp(0.5*(coords[:,:,1] + 1)),-1)
    #test = torch.mean(multiplier)/10 # maybe equal to above, but just to be safe using the two step version
    value_loss = torch.nn.L1Loss()(model_outputs * multiplier, gt_value * multiplier)
    #print("intensity loss:", regularizer)
    return value_loss

old_value_mse_with_coords = lambda preds,gt,epoch: old_value_mse(preds,preds['model_in'], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling
