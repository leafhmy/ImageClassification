import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import time
import argparse
import os
from utils.label_smoothing import LabelSmoothingCrossEntropy
from load_data.data_loader import Loader, save_file_path, AlbumentationsLoader
from net.effcientnet_pytorch.model import EfficientNet, efficientnet_params, get_model_params
from utils.train_model import train
from utils.sam import SAM
from utils.train_model_sum import train_sum
from utils.check_exists import check_path
from data_aug_test import trans2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    # base args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpus', type=str, default='0,1'),
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--checkpoint', type=int, default=20)
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--data_dir', type=str,
                        default='')
    parser.add_argument('--train_dir', type=str,
                        default='')
    parser.add_argument('--val_dir', type=str,
                        default='')
    parser.add_argument('--anno', type=str,
                        default='')
    parser.add_argument('--exp', type=str, default='./checkpoints/')
    parser.add_argument('--num_cls', type=int,
                        default=11)
    parser.add_argument('--model_name', required=True, type=str)

    # train args
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.5, help='mixup alpha')

    parser.add_argument('--commit', type=str, default='')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('[INFO ] use_gpu: {}'.format(use_cuda))

    check_path(args)

    # load data
    # data_loader = Loader(args)
    # loader = data_loader.get_data_loader()
    data_loader = AlbumentationsLoader(args)
    trans = trans2()
    data_loader.set_trans(trans)
    loader = data_loader.get_data_loader()

    # define model
    blocks_args, global_params = get_model_params('efficientnet-b0', {'num_classes': args.num_cls})
    model = EfficientNet(blocks_args, global_params).from_name('efficientnet-b0')

    device_list_os = [int(i) for i in args.gpus.strip().split(',')]
    device_list = [i for i in range(len(device_list_os))]
    print(f'[INFO] device_ids: {device_list}')

    model = model.to(device)
    model = torch.nn.DataParallel(model, device_list)
    model = model.module

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.SGD  # for train_sum
    optimizer = SAM(model.parameters(), optimizer, lr=args.lr, momentum=args.momentum)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
    criterion = F.cross_entropy
    # criterion = LabelSmoothingCrossEntropy()

    if args.resume:
        pretrained_dict = torch.load(args.resume)
        model.load_state_dict(pretrained_dict, strict=False)
        print('[INFO ] load from checkpoint: {}'.format(args.resume))

    # train(args, model, device, loader, optimizer, criterion, lr_scheduler, mixup=False, model_name=args.model_name)
    train_sum(args, model, device, loader, optimizer, criterion, lr_scheduler, mixup=False, model_name=args.model_name)
