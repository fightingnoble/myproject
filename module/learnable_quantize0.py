#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: learned_quantization.py

import torch

from module.dac import _quantize_dac

quantize_weight = _quantize_dac.apply

MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001
NORM_PPF_0_75 = 0.6745


def QuantizedWeight(weight, weight_qbit):
    """
    Quantize weight.
    Args:
        weight (torch.Tensor): a 4D tensor. [K x K x iC x oC] -> [oC x iC x K x K]
            Must have known number of channels, but can have other unknown dimensions.
        weight_qbit (int or double): variance of weight initialization.
        nbit (int): number of bits of quantized weight. Defaults to 2.
        training (bool):
    Returns:
        torch.Tensor with attribute `variables`.
    Variable Names:
    * ``basis``: basis of quantized weight.
    Note:
        About multi-GPU training: moving averages across GPUs are not aggregated.
        Batch statistics are computed by main training tower. This is consistent with most frameworks.
    """

    h_lvl_w = 2 ** (weight_qbit - 1) - 1
    with torch.no_grad():
        delta_w = 1 / h_lvl_w
    if delta_w == 0:
        weight_quan = weight
    else:
        weight_quan = quantize_weight(weight, delta_w) * delta_w
    return weight_quan


class lq_weight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, levels, thrs, nbit):
        # type: (Any, torch.Tensor, torch.Tensor, torch.Tensor, int) -> torch.Tensor
        """
        Quantize weight.
        Args:
            ctx: (object)
            x (torch.Tensor): a 4D tensor. [K x K x iC x oC] -> [oC x iC x K x K]
                Must have known number of channels, but can have other unknown dimensions.
            levels: (torch.Tensor)
            thrs: (torch.Tensor)
            nbit: (int)
        Returns:
            torch.Tensor with attribute `variables`.
        """
        if x.dim() == 4:
            # [oC x iC x K x K] -> [K x K x iC x oC]
            xp = x.permute((3, 2, 1, 0))
            oc, ic, k1, k2 = x.shape
            num_levels = 2 ** nbit
            assert levels.shape == torch.Size((num_levels, oc))
            # print(thrs.shape,num_levels, oc)
            assert thrs.shape == torch.Size((num_levels - 1, oc))

            # calculate output y
            # y [K, K, iC, oC]
            # bits_y [K x K x iC, oC, lbit]
            y = torch.zeros_like(xp) + levels[0]  # output
            zero_y = torch.zeros_like(xp)

            for i in torch.arange(num_levels - 1):
                g = torch.ge(xp, thrs[i])
                # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
                y = torch.where(g, zero_y + levels[i + 1], y)
            return y.permute((3, 2, 1, 0))
        else:
            raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(x.dim()))

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def Quantizer_train(x, basis, level_codes, thrs_multiplier,
                    nbit=2, training=False):
    """
    Quantize weight.
    Args:
        x (torch.Tensor): a 4D tensor. [K x K x iC x oC] -> [oC x iC x K x K]
            Must have known number of channels, but can have other unknown dimensions.
        basis (torch.Tensor): a 2D tensor. [lbit, oc]
        level_codes (torch.Tensor): a 2D tensor. [2**nbit, lbit] or [num_levels, lbit]
        thrs_multiplier (torch.Tensor): a 2D tensor. [num_levels-1, num_levels]
        nbit (int): number of bits of quantized weight. Defaults to 2.
        training (bool):
    Returns:
        torch.Tensor with attribute `variables`.
    Variable Names:
    * ``basis``: basis of quantized weight.
    Note:
        About multi-GPU training: moving averages across GPUs are not aggregated.
        Batch statistics are computed by main training tower. This is consistent with most frameworks.
    """
    oc, ic, k1, k2 = x.shape
    device = x.device

    num_levels = 2 ** nbit
    lbit = nbit
    delta = EPS

    # [oC x iC x K x K] -> [K x K x iC x oC]
    xp = x.permute((3, 2, 1, 0))

    N = 1
    # training
    if training:
        for _ in torch.arange(N):
            # calculate levels and sort [2**nbit, oc] or [num_levels, oc]
            levels = torch.matmul(level_codes, basis)
            levels, sort_id = torch.sort(levels, 0)

            # calculate threshold
            # [num_levels-1, oc]
            thrs = torch.matmul(thrs_multiplier, levels)

            # calculate level codes per channel
            # ix:sort_id [num_levels, oc], iy: torch.arange(nbit)
            # level_codes = [num_levels,lbit]
            # level_codes_channelwise [num_levels, oc,lbit]
            for oc_idx in torch.arange(oc):
                if oc_idx == 0:
                    level_codes_t = level_codes[
                        torch.meshgrid(sort_id[:, oc_idx], torch.arange(lbit, device=sort_id.device))].unsqueeze(1)
                    level_codes_channelwise = level_codes_t
                else:
                    level_codes_t = level_codes[
                        torch.meshgrid(sort_id[:, oc_idx], torch.arange(lbit, device=sort_id.device))].unsqueeze(1)
                    level_codes_channelwise = torch.cat((level_codes_channelwise, level_codes_t), 1)

            # calculate output y and its binary code
            # y [K, K, iC, oC]
            # bits_y [K x K x iC, oC,lbit]
            reshape_x = torch.reshape(xp, [-1, oc])

            y = torch.zeros_like(xp) + levels[0]  # output
            zero_y = torch.zeros_like(xp)
            bits_y = torch.full([reshape_x.shape[0], oc, lbit], -1., device=device)
            zero_bits_y = torch.zeros_like(bits_y)

            # [K x K x iC x oC] [1, oC]
            for i in torch.arange(num_levels - 1):
                g = torch.ge(xp, thrs[i])
                # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
                y = torch.where(g, zero_y + levels[i + 1], y)
                # [K x K x iC, oC,lbit]
                bits_y = torch.where(g.view(-1, oc, 1), zero_bits_y + level_codes_channelwise[i + 1], bits_y)

            # calculate BTxB
            # [oC,lbit, K x K x iC] x [oC, K x K x iC,lbit] => [oC,lbit,lbit]
            BTxB = torch.matmul(bits_y.permute(1, 2, 0), bits_y.permute(1, 0, 2)) + delta * torch.eye(lbit,
                                                                                                      device=device)
            # calculate inverse of BTxB
            # [oC,lbit,lbit]
            if lbit > 2:
                BTxB_inv = torch.pinverse(BTxB)
            elif lbit == 2:
                det = torch.det(BTxB)
                BTxB_inv = torch.stack((BTxB[:, 1, 1], -BTxB[:, 0, 1], -BTxB[:, 1, 0], BTxB[:, 0, 0]),
                                       1).view(OC, lbit, lbit) / det.unsqueeze(-1).unsqueeze(-1)
            elif lbit == 1:
                BTxB_inv = 1 / BTxB
            else:
                BTxB_inv = None

            # calculate BTxX
            # bits_y [K x K x iC, oc, lbit] reshape_x [K x K x iC, oC]
            # [oC, lbit, K x K x iC] [oC, K x K x iC, 1] => [oC, lbit, 1]
            BTxX = torch.matmul(bits_y.permute(1, 2, 0), reshape_x.permute(1, 0).unsqueeze(-1))
            BTxX = BTxX + (delta * basis.permute(1, 0).unsqueeze(-1))  # + basis

            # calculate new basis
            # BTxB_inv: [oC, lbit, lbit] BTxX: [oC, lbit, 1]
            # [oC, lbit, lbit] x [oC, lbit, 1] => [oC, lbit, 1] => [lbit, oC]
            new_basis = torch.matmul(BTxB_inv, BTxX).squeeze(-1).permute(1, 0)
            # print(BTxB_inv.shape,BTxX.shape)
            # print(new_basis.shape)

            # create moving averages op
            basis -= (1 - MOVING_AVERAGES_FACTOR) * (basis - new_basis)
            # print("\nbasis:\n", basis)

    # calculate levels and sort [2**nbit, oc] or [num_levels, oc]
    levels = torch.matmul(level_codes, basis)
    levels, sort_id = torch.sort(levels, 0)

    # calculate threshold
    # [num_levels-1, oc]
    thrs = torch.matmul(thrs_multiplier, levels)

    # calculate level codes per channel
    # ix:sort_id [num_levels, oc], iy: torch.arange(nbit)
    # level_codes = [num_levels, lbit]
    # level_codes_channelwise [num_levels, oc, lbit]
    for oc_idx in torch.arange(oc):
        if oc_idx == 0:
            level_codes_t = level_codes[
                torch.meshgrid(sort_id[:, oc_idx], torch.arange(lbit, device=sort_id.device))].unsqueeze(1)
            level_codes_channelwise = level_codes_t
        else:
            level_codes_t = level_codes[
                torch.meshgrid(sort_id[:, oc_idx], torch.arange(lbit, device=sort_id.device))].unsqueeze(1)
            level_codes_channelwise = torch.cat((level_codes_channelwise, level_codes_t), 1)

    return basis.transpose(1, 0), levels.transpose(1, 0), thrs.transpose(1, 0), level_codes_channelwise.transpose(1, 0)


def QuantizedWeight_conductance(x, n, nbit=2, training=False):
    """
    Quantize weight.
    Args:
        x (torch.Tensor): a 4D tensor. [K x K x iC x oC] -> [oC x iC x K x K]
            Must have known number of channels, but can have other unknown dimensions.
        n (int or double): variance of weight initialization.
        nbit (int): number of bits of quantized weight. Defaults to 2.
        training (bool):
    Returns:
        torch.Tensor with attribute `variables`.
    Variable Names:
    * ``basis``: basis of quantized weight.
    Note:
        About multi-GPU training: moving averages across GPUs are not aggregated.
        Batch statistics are computed by main training tower. This is consistent with most frameworks.
    """
    oc, ic, k1, k2 = x.shape
    device = x.device

    init_basis = []
    base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbit - 1))
    for j in range(nbit):
        init_basis.append([(2 ** j) * base for i in range(oc)])
    # init_basis.append(x.mean((1, 2, 3)).tolist())

    num_levels = 2 ** nbit
    lbit = nbit

    # initialize level multiplier
    # binary code of each level:
    # shape: [num_levels, lbit]
    init_level_multiplier = []
    for i in range(num_levels):
        level_multiplier_i = [0. for j in range(nbit)]
        level_number = i
        for j in range(nbit):
            binary_code = level_number % 2
            if binary_code == 0:
                binary_code = -1
            level_multiplier_i[j] = float(binary_code)
            level_number = level_number // 2
        # level_multiplier_i.append(1.)
        init_level_multiplier.append(level_multiplier_i)

    # initialize threshold multiplier
    # shape: [num_levels-1, num_levels]
    # [[0,0,0,0,0,0,0.5,0.5]
    #  [0,0,0,0,0,0.5,0.5,0,]
    #  [0,0,0,0,0.5,0.5,0,0,]
    #  ...
    #  [0.5,0.5,0,0,0,0,0,0,]]
    init_thrs_multiplier = []
    for i in range(1, num_levels):
        thrs_multiplier_i = [0. for j in range(num_levels)]
        thrs_multiplier_i[i - 1] = 0.5
        thrs_multiplier_i[i] = 0.5
        init_thrs_multiplier.append(thrs_multiplier_i)

    # [lbit, oc]
    basis = torch.tensor(init_basis, dtype=torch.float32, requires_grad=False).cuda()
    # [2**nbit,lbit] or [num_levels,lbit]
    level_codes = torch.tensor(init_level_multiplier).cuda()
    # [num_levels-1, num_levels]
    thrs_multiplier = torch.tensor(init_thrs_multiplier).cuda()

    basis_t, levels_t, thrs_t, level_codes_t = Quantizer_train(x, basis,
                                                               level_codes, thrs_multiplier,
                                                               nbit, training)
    # alpha = alpha.detach()
    # basis = basis_t.transpose(1, 0).detach()
    levels = levels_t.transpose(1, 0).detach()
    thrs = thrs_t.transpose(1, 0).detach()
    level_codes_channelwise = level_codes_t.transpose(1, 0).detach()

    # calculate output y and its binary code
    # y [K, K, iC, oC]
    # bits_y [K x K x iC, oC, lbit]
    y = lq_weight.apply(x, levels, thrs, nbit)
    # [oC x iC x K x K] -> [K x K x iC x oC]
    xp = x.permute((3, 2, 1, 0))
    reshape_x = torch.reshape(xp, [-1, oc])
    bits_y = torch.full([reshape_x.shape[0], oc, lbit], -1., device=device)
    zero_bits_y = torch.zeros_like(bits_y)

    # [K x K x iC x oC] [1, oC]
    for i in torch.arange(num_levels - 1):
        g = torch.ge(xp, thrs[i])
        # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
        # [K x K x iC, oC, lbit]
        bits_y = torch.where(g.view(-1, oc, 1), zero_bits_y + level_codes_channelwise[i + 1], bits_y)

    return y, levels.permute(1, 0), thrs.permute(1, 0)


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

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.models as models
    import math
    import time

    np.random.seed(1)
    # K = 3; IC = 16; OC = 16
    # w = np.random.normal(0,1,(OC, IC, K, K))
    # w = w.clip(-3,3)
    # w = w/np.abs(w).max()
    # w = torch.as_tensor(w, dtype=torch.float)

    cnn = models.resnet18(pretrained=True)

    parm = {}
    for name, parameters in cnn.named_parameters():
        print(name, ':', parameters.size())
        parm[name] = parameters.detach()

    w = parm['layer1.0.conv1.weight'].view(4, 16 * 64, 3, 3).cuda()
    # w = w + torch.arange(4).view(4, 1, 1, 1).float() / 10
    # print(parm['layer1.0.conv1.weight'].view(4, 16 * 64, 3, 3).var(dim=(1, 2, 3), keepdim=True)
    #       * torch.arange(4).view(4,1,1,1))
    OC, IC, K1, K2 = w.shape
    print("shape of the weight: ", w.shape)

    w_norm = w / w.abs().max()
    t_s = time.monotonic()

    wq, levels, threholds = QuantizedWeight_conductance(w_norm, OC * K1 * K2, 4, True)

    t_e = time.monotonic()

    s, ms = divmod((t_e - t_s) * 1000, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d:%02d" % (h, m, s, ms))

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    print('levels: ', levels)
    print('threshold: ', threholds)

    plot_histogram([w_norm[i] for i in torch.arange(OC)], ['layer1.0.conv1, filter#%d' % i for i in range(OC)],
                   bins_in=16)
    print("======original weight=======")
    plot_histogram([w_norm[i] for i in torch.arange(OC)], ['layer1.0.conv1, filter#%d' % i for i in range(OC)],
                   basis=levels, threholds=threholds)
    print("======Q1weight=======")
    plot_histogram([wq[i] for i in torch.arange(OC)],
                   ['Q1weight:layer1.0.conv1, filter#%d' % i for i in range(OC)],
                   bins_in=16)
    plot_histogram([w_norm - wq for i in torch.arange(OC)],
                   ['Q1error layer1.0.conv1, filter#%d' % i for i in range(OC)],
                   bins_in=16)
    plot_error([w_norm for i in torch.arange(OC)], [w_norm - wq for i in torch.arange(OC)],
               ['Q1 transfer error:layer1.0.conv1, filter#%d' % i for i in range(OC)])
    plot_error([w_norm for i in torch.arange(OC)], [wq for i in torch.arange(OC)],
               ['Q1 transfer:layer1.0.conv1, filter#%d' % i for i in range(OC)])
    print(torch.norm(w_norm - wq).item())
