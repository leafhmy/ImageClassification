"""
run test in local machine, modify the argparse settings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import argparse
from utils.label_smoothing import LabelSmoothingCrossEntropy
from load_data.data_loader import Loader, save_file_path, AlbumentationsLoader
from utils.conf_mat import *
from utils.draw_cam import DrawCam
from utils.save_error import SaveError
from net.effnetv2 import effnetv2_s
from data_aug_test import trans2
from net.effcientnet_pytorch.model import EfficientNet, efficientnet_params, get_model_params
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def predict(args, model, device, loader, criterion, layer=None):
    test_loader = loader['val']
    label_dict = loader['label_dict']  # key: int, value: cls
    model.eval()
    test_loss = 0
    correct = 0
    correct_top3 = 0
    error = []  # 'img_path pred target' type: str
    checkpoint_name = args.resume.split(r'/')[-1]
    exp = args.save_dir+'/'+checkpoint_name+'/'
    if not os.path.exists(exp):
        os.mkdir(exp)
    if args.conf_mat:
        conf_matrix = torch.zeros(args.num_cls, args.num_cls)
    # consume = 0.

    with torch.no_grad():
        for data, target, img_path in test_loader:
            data, target = data.to(device), target.to(device)
            # begin = time.time()
            output = model(data)
            # end = time.time()
            # consume += (end - begin)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            if args.save_error:
                pred_sq = pred.squeeze()
                assert len(pred_sq) == len(target)
                for i in range(len(pred_sq)):
                    if pred_sq[i] != target[i]:
                        error.append(img_path[i]+' '+label_dict[pred_sq[i].item()]+' '+label_dict[target[i].item()])

            if args.conf_mat:
                conf_matrix = confusion_matrix(pred, labels=target, conf_matrix=conf_matrix)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # top3 acc
            target_resize = target.view(-1, 1)
            _, pred = output.topk(3, 1, True, True)
            correct_top3 += torch.eq(pred, target_resize).sum().float().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    acc_top3 = 100. * correct_top3 / len(test_loader.dataset)
    print('[Test ] set: average loss: {:.4f}, acc: {:.6f}% acc_top3: {:.6f}%'
          .format(test_loss, acc, acc_top3))
    # print(f'ave consume {consume / 505}')
    if args.conf_mat:
        print('[INFO ] ploting confusion matrix...')
        plot_confusion_matrix(conf_matrix.numpy(), classes=list(label_dict.values()), normalize=False,
                              title='Normalized confusion matrix', metric=args.metric)
        print('[INFO ] save confusion matrix to conf_mat_pic')

    if args.save_error:
        print('[INFO ] error: {}'.format(len(error)))
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        with open(exp+'error.txt', 'w') as f:
            f.write('img_path predict true\n')
            for m in error:
                f.write(m+'\n')
            print('[INFO ] save error log to {}/error.txt'.format(exp))
            print('[INFO ] save error images...')

            saver = SaveError(error, save_dir=exp+'pic/',
                              show_cam=args.show_cam, model=model, size=(224, 224), num_cls=args.num_cls, layer=layer)
            saver.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default="/home/zhongyuning/HMY_PROS/ImgCls/checkpoints/Jul_12_20_13_52_2021/effcientnet_b0_epoch100_92.277228_0.2197.pth")
    parser.add_argument('--data_dir', type=str,
                        default="/home/zhongyuning/HMY_PROS/Pee/dataset/")
    parser.add_argument('--train_dir', type=str,
                        default='')
    parser.add_argument('--val_dir', type=str,
                        default='')
    parser.add_argument('--anno', type=str,
                        default='pee')
    parser.add_argument('--save_dir', type=str, default='./error/')
    parser.add_argument('--num_cls', type=int,
                        default=11)
    parser.add_argument('--conf_mat', type=bool, default=True, help='save confusion matrix')
    parser.add_argument('--save_error', type=bool, default=True, help='save misclassified images')
    parser.add_argument('--show_cam', type=bool, default=True, help='save misclassified images with activated map')
    parser.add_argument('--metric', type=bool, default=True, help='save model metric to .xls file')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('[INFO ] use_gpu: {}'.format(use_cuda))
    print('[INFO ] conf_mat: {} save_error: {}'.format(args.conf_mat, args.save_error))

    assert (not args.train_dir) == (not args.val_dir)
    assert not (args.train_dir and args.data_dir)

    anno_path = ''.join(['anno/', args.anno, '/'])
    args.anno = anno_path
    assert os.path.exists(args.anno + 'train.txt') and os.path.exists(args.anno + 'val.txt')

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    data_loader = Loader(args)
    loader = data_loader.get_data_loader()

    blocks_args, global_params = get_model_params('efficientnet-b0', {'num_classes': args.num_cls})
    model = EfficientNet(blocks_args, global_params).from_name('efficientnet-b0')

    model = model.to(device)
    layer = None
    model = torch.nn.DataParallel(model)
    model = model.module

    criterion = LabelSmoothingCrossEntropy()
    # criterion = F.cross_entropy

    if args.resume:
        pretrained_dict = torch.load(args.resume)
        model.load_state_dict(pretrained_dict, strict=False)
        print('[INFO ] load from checkpoint: {}'.format(args.resume))

    predict(args, model, device, loader, criterion, layer=layer)

