from resnet import build_resnet
import argparse
import torch
import math
import sys
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.append("../")

# from prune_utils import *


def add_parser_arguments(parser):
    parser.add_argument(
        "--mode",
        default='iou',
        type=str,
        help='select analysis mode: [extract, plot, iou, sparsity]'
    )
    parser.add_argument(
        "--model-path", metavar="<path>",
        default='ori_models/checkpoint-0.pth.tar',
        help="checkpoint filename"
    )
    parser.add_argument(
        "--weight-path", metavar="<path>",
        default='model_weight/checkpoint-0.npy',
        help="name of file in which to store weights"
    )
    parser.add_argument(
        "--fig-path", metavar="<path>",
        default='fig/checkpoint-0.png',
        help="name of file in which to store weight distribution fig"
    )
    parser.add_argument(
        "--iou-path-1", metavar="<path>",
        default='model_weight/checkpoint-0.npy',
        help="name of file to compare sparse mask 1"
    )
    parser.add_argument(
        "--iou-path-2", metavar="<path>",
        default='model_weight/checkpoint-60.npy',
        help="name of file to compare sparse mask 2"
    )
    parser.add_argument(
        "--widths",
        default='64-128-256-512-64',
        type=str, metavar='width',
        help='resnet width configurations'
    )



def save_harden_weight_single(args):
    print("=> loading checkpoint...")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model_state = checkpoint["state_dict"]
    new_model_state = OrderedDict()
    for k, v in model_state.items():
        k = k.replace('module.', '')
        new_model_state[k] = v
    print("=> loaded checkpoint...")

    print("=> creating model resnet50...")
    model = build_resnet(version="resnet50", config="fanin", num_classes=1000, args=args)
    model.load_state_dict(new_model_state)
    #
    # prune_init(args, model)
    # prune_harden()

    weight_save = {}
    for name, weight in model.named_parameters():
        weight = weight.cpu().detach().numpy()
        if (len(weight.shape) == 4):
            # weight1d = weight.reshape(1, -1)[0]
            weight_save[name] = weight

    np.save(args.weight_path, weight_save)
    return  weight_save

def save_harden_weight_batch(args):
    for i in range(0, 25):
        filename = "x-xxxxxx-x-x-x-x-x-x-x-x--xxxx-x-x" + str(i * 10) + ".pth.tar"
        args.sp_config_file = "../profiles/resnet_0.75.yaml"
        args.model_path = filename
        args.weight_path = "model_weight/xxx-x-x-x-xxx-x-x-x_checkpoint-" + str(i * 10) + ".npy"

        save_harden_weight_single(args)
        print("processed model: ".format(args.weight_path))


def check_sparsity(args):
    print("=> loading checkpoint...")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model_state = checkpoint["state_dict"]

    print("\n==> double check the sparsity from copied weight...")
    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model_state.items():
        if (len(weight.size()) == 4) or len(weight.size()) == 2:
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            non_zero = np.sum(weight.cpu().detach().numpy() != 0)
            # print(zeros / (zeros + non_zero))
            print("{}[{}]: irregular zeros: {}, irregular sparsity is: {:.4f}".format(name, weight.size(), zeros,
                                                                                      zeros / (zeros + non_zero)))


def check_overall_weight_distribution(args):
    print("=> loading checkpoint...")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model_state = checkpoint["state_dict"]

    all_w = None
    for key in model_state:
        # if "module.layer2.3.conv2.weight" in key:
        if 'weight' in key and "bn" not in key and "fc" not in key and "module.conv1.weight" not in key and "downsample.1" not in key:
            ww = model_state[key].detach().cpu().numpy().flatten()
            if all_w is None:
                all_w = ww
            else:
                all_w = np.concatenate((all_w, ww))

    fig = plt.figure()

    all_nonzero = all_w[np.where(all_w != 0)]
    xtick = np.linspace(-0.5, 0.5, 1000)
    plt.hist(all_nonzero, bins=xtick)
    # plt.hist(all_w)
    # plt.xlim(-0.5,0.5)
    plt.ylim(0,200000)
    plt.savefig("dist.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)




def plot_distribution(args, yscale='linear', figsize=(15,5), fontsize=5):
    print("=> loading weight npz file...")
    weight_dict = np.load(args.weight_path, allow_pickle=True)[()]
    print("=> loaded weight npz file...")

    total_zeros = 0
    total_nonzeros = 0
    for name, weight in weight_dict.items():
        zeros = np.sum(weight == 0)
        total_zeros += zeros
        non_zeros = np.sum(weight != 0)
        total_nonzeros += non_zeros
        zeros = np.sum(weight == 0)
        non_zero = np.sum(weight != 0)
        print("{}: irregular zeros: {}, irregular sparsity is: {:.4f}".format(name, zeros,
                                                                              zeros / (zeros + non_zero)))

    font = {'size': fontsize}

    plt.rc('font', **font)

    # fig = plt.figure(dpi=300)
    fig = plt.figure(figsize=figsize)
    i = 1
    for name, weight in weight_dict.items():
        ax = fig.add_subplot(6, 10, i)
        weight = weight.reshape(1, -1)[0]
        temp = weight[np.nonzero(weight)]
        xtick = np.linspace(-0.2, 0.2, 100)
        ax.hist(temp, bins=xtick)
        if yscale == 'linear':
            pass
        elif yscale == 'log':
            ax.set_yscale("log")
        ax.set_title(name)
        i += 1
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig(args.fig_path, format="png")
    # plt.show()


def IOU_calculator_one_pair(args):
    print("loading models...\n")

    weight_dict_1 = np.load(args.iou_path_1, allow_pickle=True)[()]
    weight_dict_2 = np.load(args.iou_path_2, allow_pickle=True)[()]

    print("weight loaded!\n")

    temp = []
    all_iou = {}
    for name in weight_dict_1.keys():
        tensor1 = np.array(weight_dict_1[name], dtype=bool)
        tensor2 = np.array(weight_dict_2[name], dtype=bool)

        union = tensor1 * tensor2
        overlap = tensor1 + tensor2

        zero_overlap = np.count_nonzero(overlap)
        zero_union = np.count_nonzero(union)

        IOU = zero_union / float(zero_overlap)
        if IOU != 1:
            temp.append(IOU)

        print("==> layer [{}], iou: {}".format(name, IOU))
        if IOU != 1:
            all_iou[name] = IOU
    print("\n* min IOU among all layers: {} {}".format(np.min(temp), [k for k,v in all_iou.items() if v == np.min(temp)]))
    print("* max IOU among all layers: {} {}".format(np.max(temp), [k for k,v in all_iou.items() if v == np.max(temp)]))
    print("* average IOU among all layers: {}".format(np.mean(temp)))

    return np.min(temp), np.max(temp), np.mean(temp)


def IOU_calculator_batch_util(filename1, filename2):
    weight_dict_1 = np.load(filename1, allow_pickle=True)[()]
    weight_dict_2 = np.load(filename2, allow_pickle=True)[()]

    temp = []
    all_iou = {}
    for name in weight_dict_1.keys():
        tensor1 = np.array(weight_dict_1[name], dtype=bool)
        tensor2 = np.array(weight_dict_2[name], dtype=bool)

        union = tensor1 * tensor2
        overlap = tensor1 + tensor2

        zero_overlap = np.count_nonzero(overlap)
        zero_union = np.count_nonzero(union)

        IOU = zero_union / float(zero_overlap)
        if IOU != 1:
            temp.append(IOU)
        if IOU != 1:
            all_iou[name] = IOU

    return np.min(temp), np.max(temp), np.mean(temp)


def IOU_calculator_batch():
    all_index = [(0, 0), (10, 1), (20, 2), (30, 3), (40, 4), (50, 5), (60, 6), (70, 7), (80, 8), (90, 9),
                 (100, 10), (110, 11), (120, 12), (130, 13), (140, 14), (150, 15), (160, 16), (170, 17),
                 (180, 18), (190, 19), (200, 20), (210, 21), (220, 22), (230, 23), (240, 24), (248, 25)]
    pair_combination = set(itertools.combinations(all_index, 2))

    mat_max = np.ones([26, 26])
    mat_min = np.ones([26, 26])
    mat_average = np.ones([26, 26])
    for pair in pair_combination:
        filename1 = "model_weight/0.9_checkpoint-" + str(pair[0][0]) + ".npy"
        filename2 = "model_weight/0.9_checkpoint-" + str(pair[1][0]) + ".npy"
        min, max, average = IOU_calculator_batch_util(filename1, filename2)
        print(pair, min, max, average)

        reverse_pair = (pair[1][1], pair[0][1])
        mat_max[(pair[0][1], pair[1][1])] = max
        mat_max[reverse_pair] = max

        mat_min[(pair[0][1], pair[1][1])] = min
        mat_min[reverse_pair] = min

        mat_average[(pair[0][1], pair[1][1])] = average
        mat_average[reverse_pair] = average

    im_max = plt.matshow(mat_max, cmap=plt.cm.summer_r, aspect="auto", vmin=0.0, vmax=1)
    plt.colorbar(im_max)
    plt.xticks(np.arange(0, 26, 5), [str(k[0]) for i, k in enumerate(all_index) if i % 5 == 0])
    plt.yticks(np.arange(0, 26, 5), [str(k[0]) for i, k in enumerate(all_index) if i % 5 == 0])
    plt.title("max IOU of models at different epochs")
    plt.savefig("0.9_max_iou.eps", format="eps")
    # plt.show()

    df1 = pd.DataFrame(mat_max)
    filepath1 = '0.9_max_data.xlsx'
    df1.to_excel(filepath1, index=False)
    print(mat_max)

    im_min = plt.matshow(mat_min, cmap=plt.cm.summer_r, aspect="auto", vmin=0.0, vmax=1)
    plt.colorbar(im_min)
    plt.xticks(np.arange(0, 26, 5), [str(k[0]) for i, k in enumerate(all_index) if i % 5 == 0])
    plt.yticks(np.arange(0, 26, 5), [str(k[0]) for i, k in enumerate(all_index) if i % 5 == 0])
    plt.title("min IOU of models at different epochs")
    plt.savefig("0.9_min_iou.eps", format="eps")
    # plt.show()
    df2 = pd.DataFrame(mat_min)
    filepath2 = '0.9_min_data.xlsx'
    df2.to_excel(filepath2, index=False)
    print(mat_min)

    im_average = plt.matshow(mat_average, cmap=plt.cm.summer_r, aspect="auto", vmin=0.0, vmax=1)
    plt.colorbar(im_average)
    plt.xticks(np.arange(0, 26, 5), [str(k[0]) for i, k in enumerate(all_index) if i % 5 == 0])
    plt.yticks(np.arange(0, 26, 5), [str(k[0]) for i, k in enumerate(all_index) if i % 5 == 0])
    plt.title("average IOU of models at different epochs")
    plt.savefig("0.9_average_iou.eps", format="eps")
    # plt.show()
    df3 = pd.DataFrame(mat_average)
    filepath3 = '0.9_average_data.xlsx'
    df3.to_excel(filepath3, index=False)
    print(mat_average)


def get_large_index(model, percent, include_layers=[]):
    idx_dict = {}
    for name, weight in model.items():

        if include_layers[0] == "all":
            if (len(weight.size()) == 4):
                weight = weight.cpu().detach().numpy()
                weight = weight.reshape(-1)
                weight = np.abs(weight)
                size = weight.shape[0]
                kth = math.ceil(size * (1 - percent)) - 1
                idx = np.argpartition(weight, kth=kth)
                idx_dict[name] = idx[kth:]
        else:
            if name in include_layers:
                weight = weight.cpu().detach().numpy()
                weight = weight.reshape(-1)
                weight = np.abs(weight)
                size = weight.shape[0]
                kth = math.ceil(size * (1 - percent)) - 1
                idx = np.argpartition(weight, kth=kth)
                idx_dict[name] = idx[kth:]

    return idx_dict

def overlap_percentage(dict_1, dict_2):
    match_cnt_dict = {}
    match_percent_dict = {}
    for layer in dict_1:
        size = dict_1[layer].shape[0]
        match_cnt = 0
        for idx in dict_1[layer]:
            if idx in dict_2[layer]:
                match_cnt += 1
        match_cnt_dict[layer] = match_cnt
        match_percent_dict[layer] = match_cnt/size*100
        print("In layer {}: {} matches over {} largest, match rate is {:.3f}".format(layer, match_cnt, size, match_cnt/size*100))

    return match_cnt_dict, match_percent_dict


def overlap(args):
    print("loading models...\n")

    checkpoint1 = torch.load(args.iou_path_1, map_location="cuda")
    model_state1 = checkpoint1["state_dict"]

    checkpoint2 = torch.load(args.iou_path_2, map_location="cuda")
    model_state2 = checkpoint2["state_dict"]

    print("weight loaded!\n")

    include_layer = ["all"]
    # include_layer = ["feature.0.weight", "feature.3.weight", "feature.7.weight", "feature.20.weight", "feature.30.weight", "feature.40.weight"]
    # ["feature.0.weight, \
    # feature.3.weight, \
    # feature.7.weight, \
    # feature.10.weight, \
    # feature.14.weight, \
    # feature.17.weight, \
    # feature.20.weight, \
    # feature.24.weight, \
    # feature.27.weight, \
    # feature.30.weight, \
    # feature.34.weight, \
    # feature.37.weight, \
    # feature.40.weight"]

    for name in include_layer:
        print(name)

    percentage = 0.1  # check % largest weights
    model_idx_dict = get_large_index(model_state1, percent=percentage, include_layers=include_layer)
    init_idx_dict = get_large_index(model_state2, percent=percentage, include_layers=include_layer)

    # for name, weight in model.named_parameters():
    #     if name == "feature.7.weight":
    #         weight = weight.reshape(-1)
    #         print("weight size", weight.shape)
    #         print(idx_dict[name].shape)
    #
    #         print(idx_dict[name])

    match_cnt_dict, match_percent_dict = overlap_percentage(model_idx_dict, init_idx_dict)

    for layer in match_percent_dict:
        print("{:.2f}%".format(match_percent_dict[layer]))

    print("this is percentage: {}".format(percentage))

def SNFS_ERK(module, density, tolerance: int = 5, growth_factor: float = 0.5):
    total_params = 0
    baseline_nonzero = 0
    masks = {}
    for e, (name, weight) in enumerate(module.named_parameters()):
        # Exclude first layer
        # if e == 0:
        #     continue
        # Exclude bias
        if "bias" in name:
            continue
        # Exclude batchnorm
        if "bn" in name:
            continue

        device = weight.device
        masks[name] = torch.zeros_like(
            weight, dtype=torch.float32, requires_grad=False
        ).to(device)

    for e, (name, weight) in enumerate(module.named_parameters()):
        if name not in masks:
            continue
        total_params += weight.numel()

    target_params = total_params * density
    current_params = 0
    # TODO: is the below needed
    # Can we do this more elegantly?
    # new_nonzeros = 0
    epsilon = 10.0

    # searching for the right epsilon for a specific sparsity level
    while abs(current_params - target_params) > tolerance:
        new_nonzeros = 0.0
        for name, weight in module.named_parameters():
            if name not in masks:
                continue
            # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
            # we adapt the same formula for convolutional weights
            growth = max(int(epsilon * sum(weight.shape)), weight.numel())
            new_nonzeros += growth
        current_params = new_nonzeros
        if current_params > target_params:
            epsilon *= 1.0 - growth_factor
        else:
            epsilon *= 1.0 + growth_factor
        growth_factor *= 0.95

    density_dict = {}
    for name, weight in module.named_parameters():
        if name not in masks:
            continue
        if "downsample.1." in name:
            continue
        growth = epsilon * sum(weight.shape)
        prob = growth / np.prod(weight.shape)
        density_dict[name] = prob
        print(f"ERK {name}: {weight.shape} prob {prob}")

        device = weight.device
        masks[name] = (torch.rand(weight.shape) < prob).float().data.to(device)
        baseline_nonzero += (masks[name] != 0).sum().int().item()
    print(f"Overall sparsity {baseline_nonzero/total_params}")

    return density_dict



def RigL_ERK(module, density, erk_power_scale: float = 1.0):
    """Given the method, returns the sparsity of individual layers as a dict.
    It ensures that the non-custom layers have a total parameter count as the one
    with uniform sparsities. In other words for the layers which are not in the
    custom_sparsity_map the following equation should be satisfied.
    # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
    Args:
      module:
      density: float, between 0 and 1.
      erk_power_scale: float, if given used to take power of the ratio. Use
        scale<1 to make the erdos_renyi softer.
    Returns:
      density_dict, dict of where keys() are equal to all_masks and individiual
        masks are mapped to the their densities.
    """
    # Obtain masks
    masks = {}
    total_params = 0
    for e, (name, weight) in enumerate(module.named_parameters()):
        # Exclude first layer
        # if e == 0:
        #     continue
        # Exclude bias
        if "bias" in name:
            continue
        # Exclude batchnorm
        if "bn" in name:
            continue
        # if "downsample.1." in name:
        #     continue


        device = weight.device
        masks[name] = torch.zeros_like(
            weight, dtype=torch.float32, requires_grad=False
        ).to(device)
        total_params += weight.numel()

    # We have to enforce custom sparsities and then find the correct scaling
    # factor.

    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    dense_layers = set()
    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in masks.items():
            n_param = np.prod(mask.shape)
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density

            if name in dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros

            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                raw_probabilities[name] = (
                    np.sum(mask.shape) / np.prod(mask.shape)
                ) ** erk_power_scale
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    print(f"Sparsity of var:{mask_name} had to be set to 0.")
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    density_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaning layers.
    i=0
    for name, mask in masks.items():
        n_param = np.prod(mask.shape)
        if name in dense_layers:
            density_dict[name] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            density_dict[name] = probability_one
        # print(f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}")
        print(i, name, density_dict[name])
        total_nonzero += density_dict[name] * mask.numel()
        i += 1
    print("Overall density {}".format(total_nonzero/total_params))
    return density_dict






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model analysis")

    add_parser_arguments(parser)
    # prune_parse_arguments(parser)
    args = parser.parse_args()

    # check saving directory exist or not
    if not os.path.exists('model_weight/'):
        os.system('mkdir -p ' + "model_weight/")
        print("New folder model_weight/ created...")
    if not os.path.exists('ori_models/'):
        os.system('mkdir -p ' + "ori_models/")
        print("New folder ori_models/ created...")
    if not os.path.exists('fig/'):
        os.system('mkdir -p ' + "fig/")
        print("New folder fig/ created...")

    ''' set the below argument to some meaningless value to make ADMM initialization works '''
    args.sp_admm = True
    args.rho = 0.1
    args.sp_admm_update_batch = 1
    ''' ================================================================================== '''

    # if args.mode == 'extract_single':
    #     save_harden_weight_single(args)
    # elif args.mode == 'extract_batch':
    #     save_harden_weight_batch(args)
    # elif args.mode == 'plot':
    #     plot_distribution(args, yscale='linear', figsize=(35,15), fontsize=7)
    # elif args.mode == 'iou_single_pair':
    #     IOU_calculator_one_pair(args)
    # elif args.mode == 'iou_batch':
    #     IOU_calculator_batch()
    # elif args.mode == 'sparsity':
    #     check_sparsity(args)
    # elif args.mode == "overlap":
    #     overlap(args)
    # elif args.mode == "overall_dist":
    #     check_overall_weight_distribution(args)

    model = build_resnet(version="resnet50", config="fanin", num_classes=1000, args=args)
    # from measure_model import measure_model
    # count_ops, count_params, count_conv = measure_model(model, inp_shape=(3, 224, 224))
    # print("MACs = %.2f M" % (count_ops / 1e6))
    # print("Params = %.2f M" % (count_params / 1e6))
    i=0
    for name, w in model.named_parameters():
        if len(w.size()) == 4 or "fc" in name or "downsample.1.weight" in name or "downsample.1.bias" in name:
            print(i, name)
            i += 1

    RigL_ERK(model, 0.1)