import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import copy
from tensorboardX import SummaryWriter
from utils.mixup import mixup_data, mixup_criterion


def train_sum(args, model, device, loader, optimizer, criterion, lr_scheduler, mixup=False, model_name=''):
    train_loader = loader['train']
    test_loader = loader['val']
    # best_model = copy.deepcopy(model)
    best_model_dict = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    writer = SummaryWriter(args.exp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target, _) in enumerate(train_loader):
            # data, target = torch.tensor(data), torch.tensor(target)
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if mixup:
                inputs, targets_a, targets_b, lam = mixup_data(data, target,
                                                               args.alpha, use_cuda=True)  # 对数据集进行mixup操作
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)  # 对loss#函数进行mixup操作
            else:
                output = model(data)
                loss = criterion(output, target)
            loss.backward()

            # first forward-backward pass
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            if mixup:
                inputs, targets_a, targets_b, lam = mixup_data(data, target,
                                                               args.alpha, use_cuda=True)  # 对数据集进行mixup操作
                outputs = model(inputs)
                mixup_criterion(criterion, outputs, targets_a, targets_b, lam)  # 对loss#函数进行mixup操作
            else:
                output = model(data)
                criterion(output, target).backward()

            optimizer.second_step(zero_grad=True)

            if batch_idx % args.log_interval == 0:
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

                writer.add_scalar('loss', loss, epoch)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().data.cpu().numpy(), epoch)
        lr_scheduler.step()

        if epoch % args.checkpoint == 0:
            model.eval()
            test_loss = 0
            correct = 0
            correct_top3 = 0

            with torch.no_grad():
                for data, target, _ in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    # top3 acc
                    target_resize = target.view(-1, 1)
                    _, pred = output.topk(3, 1, True, True)
                    correct_top3 += torch.eq(pred, target_resize).sum().float().item()

            test_loss /= len(test_loader.dataset)
            acc = 100. * correct / len(test_loader.dataset)
            acc_top3 = 100. * correct_top3 / len(test_loader.dataset)
            writer.add_scalar('accuracy', acc, epoch)
            print('[Test ] average loss: {:.4f}, current_lr: {}, acc: {:.6f}% acc_top3: {:.6f}%'
                  .format(test_loss, lr_scheduler.get_last_lr(), acc, acc_top3))

            if acc > best_acc:
                best_acc = acc
                best_model_dict = copy.deepcopy(model.state_dict())
                if args.save_model:
                    torch.save(best_model_dict, args.exp+"{}_epoch{}_{:.6f}_{:.4f}.pth".format(model_name, epoch, best_acc, test_loss))
                    print("[INFO ] Save checkpoint {}_epoch{}_{:.6f}_{:.4f}.pth".format(model_name, epoch, best_acc, test_loss))

    writer.close()
    print("[INFO ] train completely best acc: {}".format(best_acc))