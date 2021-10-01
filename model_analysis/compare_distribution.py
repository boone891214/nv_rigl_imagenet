from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from vgg import VGG
from resnet_ma import ResNet18, ResNet50
# from resnet import resnet18
# from lenet import LeNet
from torchvision.models.resnet import resnet18, resnet50
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import admm
from testers import *
# from my_utils import *
import math
from resnet32_cifar10_grasp import resnet32

parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--sparsity_type', type=str, default='column',
                    help="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--config_file', type=str, default='config_vgg16',
                    help="define sparsity_type: [irregular,column,filter]")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * float(correct) / float(len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def layer_sparsity(weight):

    zeros = np.sum(weight.cpu().detach().numpy() == 0)
    non_zeros = np.sum(weight.cpu().detach().numpy() != 0)

    sparsity = zeros / (zeros + non_zeros)

    return sparsity


def get_layer_large_index(weight, percent):
    weight = weight.cpu().detach().numpy()
    weight = weight.reshape(-1)
    weight = np.abs(weight)
    size = weight.shape[0]
    kth = math.ceil(size * (1 - percent))
    idx = np.argpartition(weight, kth=kth)
    layer_idx = idx[kth:]

    return layer_idx


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
        print("In layer {}: {} matches over {} largest, match rate is {:.3f}%".format(layer, match_cnt, size, match_cnt/size*100))
    print("--------------------------------")
    return match_cnt_dict, match_percent_dict


def compare_overlap(model_1, model_2, include_layer="all"):

    # ---------------- obtain per layer sparsity ------------
    model_1_weight = {}
    model_2_weight = {}
    model_1_sparsity = {}
    model_2_sparsity = {}

    if include_layer[0] == "all":
        for name, weight in model_1.named_parameters():
            if (len(weight.size()) == 4):
                sparsity = layer_sparsity(weight)
                model_1_sparsity[name] = sparsity
                model_1_weight[name] = weight.data

        for name, weight in model_2.named_parameters():
            if (len(weight.size()) == 4):
                sparsity = layer_sparsity(weight)
                model_2_sparsity[name] = sparsity
                model_2_weight[name] = weight.data

    else:   # with include layers
        for name, weight in model_1.named_parameters():
            if name in include_layer:
                sparsity = layer_sparsity(weight)
                model_1_sparsity[name] = sparsity
                model_1_weight[name] = weight.data

        for name, weight in model_2.named_parameters():
            if name in include_layer:
                sparsity = layer_sparsity(weight)
                model_2_sparsity[name] = sparsity
                model_2_weight[name] = weight.data

    percentage_dict = {}
    for name in model_1_sparsity:
        layer_percent = min(1 - model_1_sparsity[name], 1 - model_2_sparsity[name])
        layer_percent = "{:.3f}".format(layer_percent)
        layer_percent = float(layer_percent)
        print("Non-zero weights in layer {}: {:.3f}%".format(name, 100*layer_percent))
        percentage_dict[name] = layer_percent
    print("--------------------------------")
    # ---------------- get per layer large index ------------
    idx_model1_dict = {}
    idx_model2_dict = {}

    for name in percentage_dict:
        idx_model1_dict[name] = get_layer_large_index(model_1_weight[name], percentage_dict[name])
        idx_model2_dict[name] = get_layer_large_index(model_2_weight[name], percentage_dict[name])

    # ---------------- show overlap ------------
    match_cnt_dict, match_percent_dict = overlap_percentage(idx_model1_dict, idx_model2_dict)

    for i, name in enumerate(match_percent_dict):
        print("{}: {} (overlap/top) --- {:.2f}% / {:.2f}%".format(i, name, match_percent_dict[name], 100*percentage_dict[name]))




def main():

    model = resnet32(depth=32, dataset="cifar10")
    model.load_state_dict(torch.load("./model/seed914_64_lr_0.1_resnet32_cifar10_acc_91.210_sgd_lr0.1_default_sp0.950_epoch160.pt"))

    model_init = resnet32(depth=32, dataset="cifar10")
    model_init.load_state_dict(torch.load("./model/seed914_64_lr_0.1_sp0.956_resnet32_cifar10_acc_91.430_sgd_lr0.1_default_sp0.949_epoch131.pt"))

    model.cuda()

    # ---------------- layer-wise percent version ---------------
    include_layer = ["all"]
    # include_layer = ["feature.0.weight", "feature.3.weight", "feature.7.weight", "feature.20.weight", "feature.30.weight", "feature.40.weight"]
 
    compare_overlap(model_1=model, model_2=model_init, include_layer=include_layer)





if __name__ == '__main__':
    main()
