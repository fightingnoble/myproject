from __future__ import print_function

import argparse
import os
import time

import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datasets.mnist import getmnist, NUM_TRAIN
from torchvision import datasets, transforms

from helper import accuracy, AverageMeter, save_checkpoint
from module.layer1 import crxb_Conv2d, crxb_Linear
# import pydevd_pycharm
# pydevd_pycharm.settrace('0.0.0.0', port=12346, stdoutToServer=True, stderrToServer=True)


class Net(nn.Module):
    def __init__(self, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_SAF, enable_ec_SAF, quantize):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4*4*50, 500)
        # self.fc2 = nn.Linear(500, 10)
        self.conv1 = crxb_Conv2d(1, 20, kernel_size=5, crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device, quantize=quantize)
        self.conv2 = crxb_Conv2d(20, 50, kernel_size=5, crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device, quantize=quantize)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = crxb_Linear(4*4*50, 500, crxb_size=crxb_size, scaler_dw=scaler_dw,
                               gmax=gmax, gmin=gmin, gwire=gwire, gload=gload, freq=freq, temp=temp,
                               vdd=vdd, ir_drop=ir_drop, device=device, enable_noise=enable_noise,
                               enable_ec_SAF=enable_ec_SAF, enable_SAF=enable_SAF, quantize=quantize)
        self.fc2 = crxb_Linear(500, 10, crxb_size=crxb_size, scaler_dw=scaler_dw,
                               gmax=gmax, gmin=gmin, gwire=gwire, gload=gload, freq=freq, temp=temp,
                               vdd=vdd, ir_drop=ir_drop, device=device, enable_noise=enable_noise,
                               enable_ec_SAF=enable_ec_SAF, enable_SAF=enable_SAF, quantize=quantize)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Define a Convolutional Neural Network

# from torchvision.models.alexnet import AlexNet, alexnet, model_urls


# Define loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:  # print every 2000 mini-batches
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * labels.size(0),
                                                                           NUM_TRAIN,
                                                                           100. * (batch_idx + 1) / len(train_loader),
                                                                           loss.item()), end="\r")
    print('\n')
    return loss


def test(args, model, device, test_loader):
    if test_loader.dataset.train:
        print("test on validation set\r\n")
    else:
        print("test on test set\r\n")

    # validate
    model.eval()
    # test_loss = 0
    # correct = 0
    # num_samples = 0
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            test_loss = criterion(outputs, labels).item()
            # pred = outputs.argmax(dim=1, keepdim=True)
            # correct += pred.eq(labels.view_as(pred)).sum().item()
            # # _, pred = torch.max(outputs, 1)
            # # correct += (pred == labels).sum().item()
            # num_samples += pred.size(0)
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(test_loss, labels.size(0))
            top1.update(prec1[0], labels.size(0))
            top5.update(prec5[0], labels.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: Prec@1:{}/{} ({:.2f}%) Prec@5:{}/{} ({:.2f}%)\n'.format(
        losses.avg, top1.sum // 100, top1.count, top1.avg, top5.sum // 100, top1.count, top5.avg))
    return top1.avg, top5.avg, losses.avg


def run(args, model, device, train_loader, test_loader, scheduler, optimizer):
    global best_prec1
    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        train(args, model, device, train_loader, optimizer, epoch)
        prec1, prec5, loss = test(args, model, device, test_loader)
        # scheduler
        if args.scheduler =="MultiStepLR":
            scheduler.step()
        elif args.scheduler =="ReduceLROnPlateau":
            scheduler.step(loss)
        else:
            pass

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if args.detail and is_best:
            save_checkpoint({
                'epoch': args.epochs + 1,
                'arch': args.model_type,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, args.checkpoint_path + '_' + args.model_type + '_' + str(args.model_structure))


best_prec1 = 0


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # model cfg
    parser.add_argument('--model-type', type=str, default="MNIST", help="type of the model.")
    parser.add_argument('--model-structure', type=int, default=0, metavar='N',
                        help='model structure to be trained (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint, (default: None)')
    parser.add_argument('--e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # dataset
    parser.add_argument('--dataset-root', type=str, default="../datasets", help="load dataset path.")
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    # train cfg
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful to restarts)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    # device init cfg
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # optimizer
    parser.add_argument('--optim', type=str, default="", help="optim type Adam/SGD")
    parser.add_argument('--resume-optim', action='store_true', default=False,
                        help='resume optim')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    # scheduler
    parser.add_argument('--scheduler', type=str, default="None", help="scheduler MultiStepLR/None/ReduceLROnPlateau")
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--decreasing-lr', default='10', help='decreasing strategy')

    # result output cfg
    parser.add_argument('--detail', action='store_true', default=False,
                        help='show log in detial')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--checkpoint-path', type=str, default="", help="save model path.")

    # crossbar cfg
    parser.add_argument('--Quantized', action='store_true', default=False,
                        help='use quantized model')
    parser.add_argument('--qbit', type=int, default=8, help='activation/weight qbit')

    parser.add_argument('--crxb-size', type=int, default=64, help='corssbar size')
    parser.add_argument('--vdd', type=float, default=3.3, help='supply voltage')
    parser.add_argument('--gwire', type=float, default=0.0357,
                        help='wire conductacne')
    parser.add_argument('--gload', type=float, default=0.25,
                        help='load conductance')
    parser.add_argument('--gmax', type=float, default=0.000333,
                        help='maximum cell conductance')
    parser.add_argument('--gmin', type=float, default=0.000000333,
                        help='minimum cell conductance')
    parser.add_argument('--ir-drop', action='store_true', default=False,
                        help='switch to turn on ir drop analysis')
    parser.add_argument('--scaler-dw', type=float, default=1,
                        help='scaler to compress the conductance')
    parser.add_argument('--test', action='store_true', default=False,
                        help='switch to turn inference mode')
    parser.add_argument('--enable_noise', action='store_true', default=False,
                        help='switch to turn on noise analysis')
    parser.add_argument('--enable_SAF', action='store_true', default=False,
                        help='switch to turn on SAF analysis')
    parser.add_argument('--enable_ec-SAF', action='store_true', default=False,
                        help='switch to turn on SAF error correction')
    parser.add_argument('--freq', type=float, default=10e6,
                        help='scaler to compress the conductance')
    parser.add_argument('--temp', type=float, default=300,
                        help='scaler to compress the conductance')

    args = parser.parse_args()
    print("+++", args)

    # Train the network on the training data
    # Test the network on the test data
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    net = Net(crxb_size=args.crxb_size, gmax=args.gmax, gmin=args.gmin, gwire=args.gwire, gload=args.gload,
                vdd=args.vdd, ir_drop=args.ir_drop, device=device, scaler_dw=args.scaler_dw, freq=args.freq, temp=args.temp,
                enable_SAF=args.enable_SAF, enable_noise=args.enable_noise, enable_ec_SAF=args.enable_ec_SAF, quantize=args.qbit).to(device)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    # for param in net.parameters():
    #     param = nn.init.normal_(param)

    # config
    milestones = list(map(int, args.decreasing_lr.split(',')))
    print(milestones)

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY) # not good enough 68%
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.optim == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    if args.scheduler =="MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    elif args.scheduler =="ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                               patience=2, verbose=True, threshold=0.5,
                                                               threshold_mode='rel', min_lr=1e-4)
    else:
        scheduler = None

    # optionlly resume from a checkpoint
    if args.resume:
        print("=> using pre-trained model '{}'".format(args.model_type))
    else:
        print("=> creating model '{}'".format(args.model_type))

    global best_prec1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            if args.resume_optim:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except KeyError:
                    print("saved optim not compatible")
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dataset_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.train_batch_size, shuffle=True, **kwargs)

    testloader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dataset_root, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print(len(trainloader), len(testloader))

    t_s = time.monotonic()

    if not args.evaluate:
        print("!!train!!")
        run(args, net, device, trainloader, testloader, scheduler, optimizer)
    print("!!test!!")
    test(args, net, device, testloader)
    t_e = time.monotonic()

    m, s = divmod(t_e-t_s, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))

    PATH = args.checkpoint_path + '_' + args.model_type + '_' + str(args.model_structure) + '_final.pth'
    torch.save({
        'epoch': args.epochs + 1,
        'arch': args.model_type,
        'state_dict': net.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict()
    }, PATH)
    print('Finished Training')

if __name__ == '__main__':
    main()