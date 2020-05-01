#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: learned_quantization.py
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn.modules.utils
from torch import nn
from torchvision.utils import save_image, make_grid

# from module.dac import _quantize_dac

if not os.path.exists('./mlp_quantize'):
    os.mkdir('./mlp_quantize')
import visdom

vis = visdom.Visdom(env="mlp_quantize")
import pydevd_pycharm

pydevd_pycharm.settrace('0.0.0.0', port=12346, stdoutToServer=True, stderrToServer=True)

# quantize_weight = _quantize_dac.apply

MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001
NORM_PPF_0_75 = 0.6745
import numpy as np


# def QuantizedWeight(weight, weight_qbit):
#     """
#     Quantize weight.
#     Args:
#         weight (torch.Tensor): a 4D tensor. [K x K x iC x oC] -> [oC x iC x K x K]
#             Must have known number of channels, but can have other unknown dimensions.
#         weight_qbit (int or double): variance of weight initialization.
#         nbit (int): number of bits of quantized weight. Defaults to 2.
#         training (bool):
#     Returns:
#         torch.Tensor with attribute `variables`.
#     Variable Names:
#     * ``basis``: basis of quantized weight.
#     Note:
#         About multi-GPU training: moving averages across GPUs are not aggregated.
#         Batch statistics are computed by main training tower. This is consistent with most frameworks.
#     """
#
#     h_lvl_w = 2 ** (weight_qbit - 1) - 1
#     with torch.no_grad():
#         delta_w = 1 / h_lvl_w
#     if delta_w == 0:
#         weight_quan = weight
#     else:
#         weight_quan = quantize_weight(weight, delta_w) * delta_w
#     return weight_quan


def normfun(x, mu, sigma):
    x = (x - mu) / sigma
    pdf = np.exp(-((x - 0) ** 2) / (2 * 1 ** 2)) / (1 * np.sqrt(2 * np.pi))
    return pdf


def plot_histogram(layer_activation_list, layer_activation_name_list,
                   basis=None, threholds=None,
                   bins_in=None, fit=False, norm=False):
    num_layer = len(layer_activation_list)
    plt_col = plt_row = math.ceil(np.sqrt(num_layer))
    fig = plt.figure(figsize=(10 * plt_col, 10 * plt_row))
    for i, (layer_act_val, name) in enumerate(zip(layer_activation_list, layer_activation_name_list)):
        ax_act = fig.add_subplot(plt_row, plt_col, i + 1)
        # ax_act.clear()

        # example data
        # print(layer_act_val.shape)
        data = layer_act_val.detach().cpu().numpy().ravel()
        sigma = np.std(data)
        mu = np.mean(data)

        # the histogram of the data
        # counts, bins = np.histogram(data, bins=bin)
        # n, bins, patches = plt.hist(bins[:-1], bins, weights=counts, density=True)
        if bins_in is None:
            bins = np.concatenate([np.array([data.min()]), threholds[i].detach().cpu().numpy(), np.array([data.max()])])
            plt.hist(data, bins=bins, density=norm, edgecolor='k', histtype='bar', rwidth=0.8, zorder=1)
            np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
            print('bin:', bins)
        else:
            counts, bins = np.histogram(data, bins=bins_in)
            plt.hist(data, bins=bins, density=norm, edgecolor='k', histtype='bar', rwidth=0.8, zorder=1)
        if basis is not None:
            ax_act.scatter(basis[i].cpu().data.numpy(), np.zeros_like(basis[i].cpu().data.numpy()), s=100,
                           edgecolors='g', c='r', marker='D', zorder=2)

        if fit:
            y = normfun(bins, mu, sigma)
            ax_act.plot(bins, y, '--')
        plt.tick_params(labelsize=20)
        plt.ticklabel_format(axis='x', style='sci')
        ax_act.set_xlabel('value', fontsize=20)
        ax_act.set_ylabel('number', fontsize=20)
        ax_act.xaxis.get_major_formatter().set_powerlimits((0, 0))
        # ax_act.xaxis.set_major_formatter()
        ax_act.set_title('%s, $\mu=%.3f$, $\sigma=%.3f$' % (name, mu.item(), sigma.item()), fontsize=20)
        # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def plot_error(layer_activation_list, layer_error_list, layer_activation_name_list):
    num_layer = len(layer_activation_list)
    plt_col = plt_row = math.ceil(np.sqrt(num_layer))
    fig = plt.figure(figsize=(10 * plt_col, 10 * plt_row))
    for i, (layer_act_val, layer_act_error, name) in enumerate(
            zip(layer_activation_list, layer_error_list, layer_activation_name_list)):
        ax_act = fig.add_subplot(plt_row, plt_col, i + 1)
        # ax_act.clear()

        # example data
        # print(layer_act_val.shape)
        data = layer_act_val.detach().cpu().numpy().ravel()
        error = layer_act_error.detach().cpu().numpy().ravel()
        sigma = np.std(data)
        mu = np.mean(data)

        ax_act.scatter(data, error, s=100,
                       edgecolors='g', c='r', marker='.', zorder=2)

        plt.tick_params(labelsize=20)
        plt.ticklabel_format(axis='x', style='sci')
        ax_act.set_xlabel('value', fontsize=20)
        ax_act.set_ylabel('error', fontsize=20)
        ax_act.xaxis.get_major_formatter().set_powerlimits((0, 0))
        # ax_act.xaxis.set_major_formatter()
        ax_act.set_title('%s, $\mu=%.3f$, $\sigma=%.3f$' % (name, mu.item(), sigma.item()), fontsize=20)
        # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


class _newrelu(torch.autograd.Function):
    '''
    This self-define function is used for mapping weight on positive
    and negative array. It will prevent close to zero weights trapped
    within the region that quantized into zero, which will never be
    updated by back-propagation, thus degrades the accuracy.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


x_relu = _newrelu.apply


class _newsign(torch.autograd.Function):
    '''
    This self-define function is used for mapping weight on positive
    and negative array. It will prevent close to zero weights trapped
    within the region that quantized into zero, which will never be
    updated by back-propagation, thus degrades the accuracy.
    '''

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


x_sign = _newsign.apply


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    # x = x.view(x.size(0), 1, 28, 28)
    return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# class autoencoder(nn.Module):
#     def __init__(self, qbit=4):
#         super(autoencoder, self).__init__()
#         self.qbit = qbit
#         self.encoder = nn.Sequential(
#             nn.Linear(16, 8),
#             nn.ReLU(True),
#             nn.Linear(8, qbit),
#             nn.Tanh())
#         # self.decoder = nn.Sequential(
#         #     nn.Linear(16, 16),
#         #     nn.ReLU(True),
#         #     nn.Linear(16, 16),
#         #     nn.Sigmoid())
#         self.decoder = nn.Sequential(
#             nn.Linear(16, 8),
#             nn.ReLU(True),
#             nn.Linear(8, qbit),
#             nn.Sigmoid())
#
#         h_level = 1 / (16 ** 2)
#         init_basis = [h_level * (2 ** j) for j in range(16)]
#         init_levels = [j * h_level for j in range(16 ** 2)]
#         self.basis = torch.tensor(init_basis).float().cuda()
#         # nn.Parameter(torch.tensor(init_basis).float().cuda(),True)
#         self.levels = torch.tensor(init_levels).float().cuda()
#         # nn.Parameter(torch.tensor(init_levels).float().cuda(),True)
#
#         self.num_levels = qbit ** 2
#         init_level_multiplier = []
#         for i in range(256):
#             level_multiplier_i = [-1. for j in range(16)]
#             level_number = i
#             for j in range(16):
#                 binary_code = level_number % 2
#                 if binary_code == 0:
#                     binary_code = -1
#                 level_multiplier_i[j] = float(binary_code)
#                 level_number = level_number // 2
#             # level_multiplier_i.append(1.)
#             init_level_multiplier.append(level_multiplier_i)
#         self.level_codes_16 = torch.tensor(init_level_multiplier).cuda()
#         self.level_codes_qbit = self.level_codes_16[:self.qbit,:self.qbit]
#         self.code = None
#
#
#     def forward(self, x: torch.Tensor):
#         # ----------initial coding---------
#         # assert len(x.shape) <= 2
#         a= x.shape
#         x_h = x.clone()
#         # [shape, 1] - [len] => [shape, len]
#         index = (x.unsqueeze(-1) - self.levels).abs().argmin(dim=-1)
#         bits_y = torch.full(a + torch.Size((16,)), -1., device=x.device)
#         axi = torch.meshgrid([torch.arange(i) for i in a])
#         # print(xx.shape,yy.shape,index.shape)
#         # print(bits_y.shape, self.level_codes.shape)
#         bits_y[axi] = self.level_codes_16[index[axi]]
#         x = bits_y.view(-1, 16)
#         # with torch.no_grad(): print(((torch.matmul(x, self.basis) - x_h.view(-1))**2).sum())
#         # ----------initial coding end---------
#
#         x = self.encoder(x)  #
#         # x = x_sign(x)
#         # self.code = x
#         x_basis = self.decoder(self.basis)
#         # print(len(set(x.view(-1).tolist())))
#         # print(len(set(x.view(-1).tolist())))
#
#         # print(self.level_codes_qbit.shape, x_basis.shape)
#         x_levels = torch.matmul(x, x_basis)
#         # [shape, 1] - [len] => [shape, len]
#         index = (x.unsqueeze(-1) - x_levels).abs().argmin(dim=-1)
#         bits_y = torch.full(a + torch.Size((self.qbit,)), -1., device=x.device)
#         axi = torch.meshgrid([torch.arange(i) for i in a])
#         # print(xx.shape,yy.shape,index.shape)
#         # print(bits_y.shape, self.level_codes.shape)
#         bits_y[axi] = self.level_codes_qbit[index[axi]]
#         x = bits_y.view(-1, self.qbit)
#         self.code = x
#
#         # (N,16)(16,)
#         x = torch.matmul(x, x_basis)
#         x = x.view(a)
#         return x


class autoencoder(nn.Module):
    def __init__(self, qbit=4):
        super(autoencoder, self).__init__()
        self.qbit = qbit
        self.encoder = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, qbit),
            nn.Tanh())
        # self.decoder = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.ReLU(True),
        #     nn.Linear(16, 16),
        #     nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, qbit),
            nn.Sigmoid())

        h_level = 1 / (2 ** 16)
        init_basis_16 = [h_level * (2 ** j) for j in range(16)]
        self.basis_16 = torch.tensor(init_basis_16).float().cuda()
        # nn.Parameter(torch.tensor(init_basis_16).float().cuda(),True)

        self.num_levels = 2 ** qbit
        init_level_multiplier = []
        for i in range(2 ** 16):
            level_multiplier_i = [-1. for j in range(16)]
            level_number = i
            for j in range(16):
                binary_code = level_number % 2
                if binary_code == 0:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            # level_multiplier_i.append(1.)
            init_level_multiplier.append(level_multiplier_i)
        self.level_codes = torch.tensor(init_level_multiplier).cuda()
        # init_levels = [j * h_level for j in range(-2**15, 2 ** 15)]
        self.levels = torch.matmul(self.level_codes, self.basis_16)
        # nn.Parameter(torch.tensor(init_levels).float().cuda(),True)
        self.code = None
        self.basis = None

    def forward(self, x: torch.Tensor):
        # ----------initial coding---------
        # assert len(x.shape) <= 2
        a = x.shape
        x_h = x.clone()
        # [shape, 1] - [len] => [shape, len]
        index = (x.unsqueeze(-1) - self.levels).abs().argmin(dim=-1)
        bits_y = torch.full(a + torch.Size((16,)), -1., device=x.device)
        axi = torch.meshgrid([torch.arange(i) for i in a])
        # print(xx.shape,yy.shape,index.shape)
        # print(bits_y.shape, self.level_codes.shape)
        bits_y[axi] = self.level_codes[index[axi]]
        x = bits_y.view(-1, 16)
        # with torch.no_grad(): print(((torch.matmul(x, self.basis_16) - x_h.view(-1))**2).sum())
        # ----------initial coding end---------

        x = self.encoder(x)  #
        x = x_sign(x)
        self.code = x
        x_basis = self.decoder(self.basis_16)
        self.basis = x_basis
        # print(len(set(x.view(-1).tolist())))
        # print(len(set(x.view(-1).tolist())))
        # (N,16)(16,)
        x = torch.matmul(x, x_basis)
        x = x.view(a)
        return x


# class autoencoder(nn.Module):
#     def __init__(self, qbit=4):
#         super(autoencoder, self).__init__()
#         self.qbit = qbit
#         self.encoder = nn.Sequential(
#             nn.Linear(qbit, qbit),
#             nn.ReLU(True),
#             nn.Linear(qbit, qbit),
#             nn.Tanh())
#         self.decoder = nn.Sequential(
#             nn.Linear(qbit, qbit),
#             nn.ReLU(True),
#             nn.Linear(qbit, qbit),
#             nn.Sigmoid())
#
#         h_level = 1 / (qbit ** 2)
#         init_basis = [h_level * (2 ** j) for j in range(qbit)]
#         init_levels = [j * h_level for j in range(qbit ** 2)]
#         self.basis = torch.tensor(init_basis).float().cuda()
#         # nn.Parameter(torch.tensor(init_basis).float().cuda(),True)
#         self.levels = torch.tensor(init_levels).float().cuda()
#         # nn.Parameter(torch.tensor(init_levels).float().cuda(),True)
#
#         self.num_levels = qbit ** 2
#         init_level_multiplier = []
#         for i in range(qbit ** 2):
#             level_multiplier_i = [-1. for j in range(qbit)]
#             level_number = i
#             for j in range(qbit):
#                 binary_code = level_number % 2
#                 if binary_code == 0:
#                     binary_code = -1
#                 level_multiplier_i[j] = float(binary_code)
#                 level_number = level_number // 2
#             # level_multiplier_i.append(1.)
#             init_level_multiplier.append(level_multiplier_i)
#         self.level_codes = torch.tensor(init_level_multiplier).cuda()
#         self.code = None
#
#     def forward(self, x: torch.Tensor):
#         # assert len(x.shape) <= 2
#         a= x.shape
#
#         x = self.encoder(x)  #
#         x = x_sign(x)
#         self.code = x
#         x_basis = self.decoder(self.basis)
#         # print(len(set(x.view(-1).tolist())))
#         # print(len(set(x.view(-1).tolist())))
#         # (N,self.qbit)(self.qbit,)
#         x = torch.matmul(x, x_basis)
#         x = x.view(a)
#
#         # ----------coding---------
#         # [shape, 1] - [len] => [shape, len]
#         index = (x.unsqueeze(-1) - self.levels).abs().argmin(dim=-1)
#         bits_y = torch.full(a + torch.Size((self.qbit,)), -1., device=x.device)
#         axi = torch.meshgrid([torch.arange(i) for i in a])
#         # print(xx.shape,yy.shape,index.shape)
#         # print(bits_y.shape, self.level_codes.shape)
#         bits_y[axi] = self.level_codes[index[axi]]
#         x = bits_y.view(-1, self.qbit)
#         # ---------- coding end---------
#
#         return x

def main():
    import time
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # model cfg
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint, (default: None)')
    parser.add_argument('--e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-l', '--reg_param', type=float, default=0.0003,
                        help='regularization parameter `lambda`')
    args = parser.parse_args()
    print("+++", args)

    # ---------------load model-----------
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']

    cnn = Net()
    model_dict = cnn.state_dict()
    pretrained_dict = checkpoint['state_dict']
    # pretrained_dict = torch.load(args.resume,map_location=torch.device('cpu'))['state_dict']
    print(model_dict.keys(), '\r\n')
    print(pretrained_dict.keys(), '\r\n')
    new_dict = {}
    for k, v in model_dict.items():
        # fit model saved in parallel model
        if 'module.' + k in pretrained_dict:
            new_dict[k] = pretrained_dict['module.' + k]
            print(k, ' ')
        elif k in pretrained_dict:
            new_dict[k] = pretrained_dict[k]
            print(k, ' ')
        else:
            new_dict[k] = v
            print(k, '!!')
    model_dict.update(new_dict)
    cnn.load_state_dict(model_dict)
    # ---------------load model-----------

    w = cnn.state_dict()['conv1.weight'].cuda()
    print("shape of the weight: ", w.shape)
    pic = to_img(w.view(20, 1, 5, 5).cpu().data)
    print(pic.shape)
    save_image(pic, './mlp_quantize/image_origin.png')
    vis.image(make_grid(pic), "pic_origin", opts=dict(title='pic_origin'))

    w_norm = w.view(5, 100).cuda() / w.abs().max()
    # print(set(w_norm.abs().max().view(-1).tolist()))

    num_epochs = 200
    batch_size = 128
    learning_rate = 1e-3
    qbit = 4

    model = autoencoder(qbit).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # # define the sparse loss function
    # def sparse_loss(autoencoder, images):
    #     model_children = list(autoencoder.children())
    #     loss = 0
    #     values = images
    #     for i in range(len(model_children)):
    #         values = F.relu((model_children[i](values)))
    #         loss += torch.mean(torch.abs(values))
    #     return loss

    t_s = time.monotonic()

    for epoch in range(num_epochs):
        # ===================forward=====================
        output = model(w_norm)
        loss = criterion(output, w_norm)  # + model.code.abs().sum() * args.reg_param
        # with torch.no_grad(): print(model.code.abs().sum(),criterion(output, w_norm))
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
        if epoch % 4 == 0:
            pic = to_img(output.view(20, 1, 5, 5).cpu().data)
            save_image(pic, './mlp_quantize/image_{}.png'.format(epoch))
            vis.image(make_grid(pic), "pic", opts=dict(title='pic'))

    t_e = time.monotonic()

    s, ms = divmod((t_e - t_s) * 1000, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d:%02d" % (h, m, s, ms))

    model.eval()
    print(len(set(w_norm.view(-1).tolist())))

    basis = model.basis
    level_codes = model.level_codes[:2 ** qbit, :qbit]
    levels, sort_id = torch.matmul(level_codes, basis).sort()
    print(basis, '\n', levels, '\n', level_codes)
    #
    x = w_norm
    output = torch.zeros_like(x)
    a = x.shape
    # [shape, 1] - [len] => [shape, len]
    index = (x.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    axi = torch.meshgrid([torch.arange(i) for i in a])
    output[axi] = levels[index[axi]]

    def forward(self, x: torch.Tensor):
        # ----------initial coding---------
        # assert len(x.shape) <= 2
        a = x.shape
        # [shape, 1] - [len] => [shape, len]
        index = (x.unsqueeze(-1) - self.levels).abs().argmin(dim=-1)
        bits_y = torch.full(a + torch.Size((16,)), -1., device=x.device)
        axi = torch.meshgrid([torch.arange(i) for i in a])
        # print(xx.shape,yy.shape,index.shape)
        # print(bits_y.shape, self.level_codes.shape)
        bits_y[axi] = self.level_codes[index[axi]]
        x = bits_y.view(-1, 16)
        # with torch.no_grad(): print(((torch.matmul(x, self.basis_16) - x_h.view(-1))**2).sum())
        # ----------initial coding end---------

        # x = self.encoder(x)  #
        # x = x_sign(x)
        # self.code = x
        # x_basis = self.decoder(self.basis_16)
        # self.basis = x_basis
        # # print(len(set(x.view(-1).tolist())))
        # # print(len(set(x.view(-1).tolist())))
        # # (N,16)(16,)
        x = torch.matmul(x, self.basis_16)
        print(self.basis_16.max())
        x = x.view(a)
        return x

    # output = forward(model,w_norm)
    # output = model(w_norm)
    print(len(set(output.view(-1).tolist())))

    torch.save(model.state_dict(), './AE_quantize.pth')

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    vis.histogram(w_norm.view(-1), 'w_norm', opts=dict(title='w_norm'))
    vis.histogram(output.view(-1), 'w_quan', opts=dict(title='w_quan'))

    x = torch.arange(-1., 1., 0.01).to(w.device)
    y = torch.zeros_like(x)
    a = x.shape
    # [shape, 1] - [len] => [shape, len]
    index = (x.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    axi = torch.meshgrid([torch.arange(i) for i in a])
    y[axi] = levels[index[axi]]
    # y = forward(model,x)
    # y = model(x)
    vis.line(X=x, Y=y, win='qcurve', opts=dict(title='qcurve'))
    pass

    # print('levels: ', levels)
    # print('threshold: ', threholds)
    #
    # plot_histogram([w_norm[i] for i in torch.arange(OC)], ['layer1.0.conv1, filter#%d' % i for i in range(OC)],
    #                bins_in=16)
    # print("======original weight=======")
    # plot_histogram([w_norm[i] for i in torch.arange(OC)], ['layer1.0.conv1, filter#%d' % i for i in range(OC)],
    #                basis=levels, threholds=threholds)
    # print("======Q1weight=======")
    # plot_histogram([wq[i] for i in torch.arange(OC)],
    #                ['Q1weight:layer1.0.conv1, filter#%d' % i for i in range(OC)],
    #                bins_in=16)
    # plot_histogram([w_norm - wq for i in torch.arange(OC)],
    #                ['Q1error layer1.0.conv1, filter#%d' % i for i in range(OC)],
    #                bins_in=16)
    # plot_error([w_norm for i in torch.arange(OC)], [w_norm - wq for i in torch.arange(OC)],
    #            ['Q1 transfer error:layer1.0.conv1, filter#%d' % i for i in range(OC)])
    # plot_error([w_norm for i in torch.arange(OC)], [wq for i in torch.arange(OC)],
    #            ['Q1 transfer:layer1.0.conv1, filter#%d' % i for i in range(OC)])
    # print(torch.norm(w_norm - wq).item())


if __name__ == "__main__":
    main()
