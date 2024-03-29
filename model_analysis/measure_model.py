from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
import pdb

count_ops = 0
count_params = 0
conv_num = 0

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, *xs):
    global count_ops, count_params, conv_num
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    x = xs[0]
    ### ops_conv
    if type_name in ['Conv2d', 'Conv2dQuant']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        # print("========")
        # print(conv_num)
        # print(layer)
        # print('input:', x.size())
        # print('output:1 ', layer.out_channels, out_h, out_w)
        # print(layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1])     # print Parameters
        print(delta_ops/ 1e6)   # print MACs

        delta_params = get_layer_param(layer)
        conv_num += 1
        
    elif type_name in ['MatMul', 'ThreeMatMul']:

        delta_ops = layer.get_computation()
        delta_params = 0
        
    elif type_name in ['ConvTranspose2d']:
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                layer.kernel_size[1] * x.size()[2] * x.size()[3] / layer.groups * multi_add
        #print(layer)
        #print(delta_ops/ 1e6)
        delta_params = get_layer_param(layer)
    
    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        #delta_ops = x.numel()
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        #delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        #delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_ops = 0.0
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel() if layer.bias is not None else 0
        #delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_ops = weight_ops + bias_ops
        delta_params = get_layer_param(layer)
        # print(delta_ops / 1e6)  # print MACs

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'Upsample']:
        delta_params = get_layer_param(layer)
        delta_ops = 0.0

    elif type_name in ['Upsample', 'Hardtanh' ,'QuantizedHardtanh' , 'MaxPool2d' ]:
        delta_params = 0
        delta_ops = 0.0
    ### unknown layer type
    
    else:
        delta_params = 0
        delta_ops = 0.0
        #raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, inp=None, inp_shape=None):
    global count_ops, count_params, conv_num
    count_ops = 0
    count_params = 0
    if inp is None and inp_shape is not None:
        C, H, W = inp_shape
        inp = Variable(torch.zeros(1, C, H, W))

    #if torch.cuda.is_available():
    #    inp = inp.cuda()

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(*x):
                        ret = m.old_forward(*x)
                        measure_layer(m, *x)
                        return ret
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    hidden = model.forward(inp) 
    # for i in hidden:
    #    print(i.shape)
    restore_forward(model)
    print ('macs=', count_ops/1e6, 'M, params', count_params/1e6, 'M, layers=', conv_num)
    return count_ops, count_params, conv_num


