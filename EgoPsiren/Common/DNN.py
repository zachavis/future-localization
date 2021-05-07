import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, self.sub_latent_dimension, 21, 1, 21//2)
        self.relu = nn.ReLU(inplace=True)
        self.latent_dimension = latent_dimension

        self.cnn = nn.Sequential(
            nn.Conv2d(self.sub_latent_dimension, self.latent_dimension, 3, 1, 1),
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
        target = target.cuda()
    output = network(data)
    zero = torch.tensor(np.zeros(target.shape, dtype=np.float), dtype=torch.float32)
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


def trainAndGraph(network, training_generator, testing_generator, loss_function, optimizer, num_epochs, learning_rate, output_path, logging_rate=1, train = train, test = test_with_grad, graph = True):
  
  #print('training and graphing')
  #Arrays to store training history
  test_loss_history = []
  epoch_counter = []
  train_loss_history = []

  best_test_loss = np.inf

  for epoch in range(num_epochs):                           
    avg_loss = train(network, training_generator, loss_function, optimizer, epoch)
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



trainAndGraphDerivative = lambda net, train_gen, test_gen, loss_fn, opt, n_epochs, lr, outputPath, log_rate = 1, verbose = False : trainAndGraph(net, train_gen, test_gen, loss_fn, opt, n_epochs, lr, outputPath, log_rate, train, test_with_grad, graph = verbose)







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



def value_mse(model_outputs, coords, gt_value, epoch):


    #model_outputs = torch.squeeze(model_outputs,-1)
    #multiplier = torch.zeros(coords.shape)
    multiplier = torch.unsqueeze(torch.exp(0.5*(coords[:,:,1] + 1)),-1)
    value_loss = torch.nn.L1Loss()(model_outputs * multiplier, gt_value * multiplier)
    return value_loss

#laplacian_mse_with_coords = lambda preds,gt,epoch: laplacian_mse(preds[0],preds[1], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling
value_mse_with_coords = lambda preds,gt,epoch: value_mse(preds['model_out'],preds['model_in'], gt,epoch) # TODO: CAUTION: these positions are inneffective when using stochastic sampling





