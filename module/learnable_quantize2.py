#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: learned_quantization.py

import torch

MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001
NORM_PPF_0_75 = 0.6745


def QuantizedWeight(x, n, nbit=2, training=False):
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

    num_levels = 2 ** nbit
    delta = EPS

    # initialize level multiplier
    # binary code of each level:
    # shape: [num_levels, nbit]
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

    # [nbit, oc]
    basis = torch.tensor(init_basis, dtype=torch.float32, requires_grad=False)
    # [2**nbit, nbit] or [num_levels, nbit]
    level_codes = torch.tensor(init_level_multiplier)
    # [num_levels-1, num_levels]
    thrs_multiplier = torch.tensor(init_thrs_multiplier)

    # [oC x iC x K x K] -> [K x K x iC x oC]
    xp = x.permute((3, 2, 1, 0))

    N = 3
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
            # level_codes = [num_levels, nbit]
            # level_codes_channelwise [num_levels, oc, nbit]
            for oc_idx in torch.arange(oc):
                if oc_idx == 0:
                    level_codes_t = level_codes[
                        torch.meshgrid(sort_id[:, oc_idx], torch.arange(nbit, device=sort_id.device))].unsqueeze(1)
                    level_codes_channelwise = level_codes_t
                else:
                    level_codes_t = level_codes[
                        torch.meshgrid(sort_id[:, oc_idx], torch.arange(nbit, device=sort_id.device))].unsqueeze(1)
                    level_codes_channelwise = torch.cat((level_codes_channelwise, level_codes_t), 1)

            # calculate output y and its binary code
            # y [K, K, iC, oC]
            # bits_y [K x K x iC, oC, nbit]
            reshape_x = torch.reshape(xp, [-1, oc])
            y = torch.zeros_like(xp) + levels[0]  # output
            zero_y = torch.zeros_like(xp)
            bits_y = torch.full([reshape_x.shape[0], oc, nbit], -1., device=device)
            zero_bits_y = torch.zeros_like(bits_y)

            # [K x K x iC x oC] [1, oC]
            for i in torch.arange(num_levels - 1):
                g = torch.ge(xp, thrs[i])
                # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
                y = torch.where(g, zero_y + levels[i + 1], y)
                # [K x K x iC, oC, nbit]
                bits_y = torch.where(g.view(-1, oc, 1), zero_bits_y + level_codes_channelwise[i + 1], bits_y)

            # calculate BTxB
            # [oC, nbit, K x K x iC] x [oC, K x K x iC, nbit] => [oC, nbit, nbit]
            BTxB = torch.matmul(bits_y.permute(1, 2, 0), bits_y.permute(1, 0, 2)) + delta * torch.eye(nbit,
                                                                                                      device=device)
            # calculate inverse of BTxB
            # [oC, nbit, nbit]
            if nbit > 2:
                BTxB_inv = torch.inverse(BTxB)
            elif nbit == 2:
                det = torch.det(BTxB)
                BTxB_inv = torch.stack((BTxB[:, 1, 1], -BTxB[:, 0, 1], -BTxB[:, 1, 0], BTxB[:, 0, 0]),
                                       1).view(OC, nbit, nbit) / det.unsqueeze(-1).unsqueeze(-1)
            elif nbit == 1:
                BTxB_inv = 1 / BTxB
            else:
                BTxB_inv = None

            # calculate BTxX
            # bits_y [K x K x iC, oc, nbit] reshape_x [K x K x iC, oC]
            # [oC, nbit, K x K x iC] [oC, K x K x iC, 1] => [oC, nbit, 1]
            BTxX = torch.matmul(bits_y.permute(1, 2, 0), reshape_x.permute(1, 0).unsqueeze(-1))
            BTxX = BTxX + (delta * basis.permute(1, 0).unsqueeze(-1))  # + basis

            # calculate new basis
            # BTxB_inv: [oC, nbit, nbit] BTxX: [oC, nbit, 1]
            # [oC, nbit, nbit] x [oC, nbit, 1] => [oC, nbit, 1] => [nbit, oC]
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
    # level_codes = [num_levels, nbit]
    # level_codes_channelwise [num_levels, oc, nbit]
    for oc_idx in torch.arange(oc):
        if oc_idx == 0:
            level_codes_t = level_codes[
                torch.meshgrid(sort_id[:, oc_idx], torch.arange(nbit, device=sort_id.device))].unsqueeze(1)
            level_codes_channelwise = level_codes_t
        else:
            level_codes_t = level_codes[
                torch.meshgrid(sort_id[:, oc_idx], torch.arange(nbit, device=sort_id.device))].unsqueeze(1)
            level_codes_channelwise = torch.cat((level_codes_channelwise, level_codes_t), 1)

    # calculate output y and its binary code
    # y [K, K, iC, oC]
    # bits_y [K x K x iC, oC, nbit]
    reshape_x = torch.reshape(xp, [-1, oc])
    y = torch.zeros_like(xp) + levels[0]  # output
    zero_y = torch.zeros_like(xp)
    bits_y = torch.full([reshape_x.shape[0], oc, nbit], -1., device=device)
    zero_bits_y = torch.zeros_like(bits_y)

    # [K x K x iC x oC] [1, oC]
    for i in torch.arange(num_levels - 1):
        g = torch.ge(xp, thrs[i])
        # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
        y = torch.where(g, zero_y + levels[i + 1], y)
        # [K x K x iC, oC, nbit]
        bits_y = torch.where(g.view(-1, oc, 1), zero_bits_y + level_codes_channelwise[i + 1], bits_y)

    return y.permute(3, 2, 1, 0), levels.permute(1, 0), thrs.permute(1, 0)


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
            # bits_y [K x K x iC, oC, nbit + 1]
            y = torch.zeros_like(xp) + levels[0]  # output
            zero_y = torch.zeros_like(xp)

            for i in torch.arange(num_levels - 1):
                g = torch.ge(xp, thrs[i])
                # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
                y = torch.where(g, zero_y + levels[i + 1], y)
            return y.permute((3, 2, 1, 0))
        else:
            raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(input.dim()))

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def Quantizer_train(x, basis, level_codes, thrs_multiplier, variation_multiplier, alpha, nbit=2, training=False):
    """
    Quantize weight.
    Args:
        x (torch.Tensor): a 4D tensor. [K x K x iC x oC] -> [oC x iC x K x K]
            Must have known number of channels, but can have other unknown dimensions.
        basis (torch.Tensor): a 2D tensor. [nbit+1, oc]
        level_codes (torch.Tensor): a 2D tensor. [2**nbit, nbit+1] or [num_levels, nbit+1]
        thrs_multiplier (torch.Tensor): a 2D tensor. [num_levels-1, num_levels]
        variation_multiplier (torch.Tensor): a 2D tensor. [num_levels-1, num_levels]
        alpha (torch.Tensor): a 2D tensor. [num_levels-1, oc]
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

    # [oC x iC x K x K] -> [K x K x iC x oC]
    xp = x.permute((3, 2, 1, 0))

    import torch.optim.lbfgs as lbfgs
    param = torch.cat((basis, alpha),dim=0)
    optimizer = lbfgs.LBFGS(param, lr=0.1)
    basis = param[0:nbit+1]
    alpha = param[nbit+1:]

    N = 4
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
            # level_codes = [num_levels,nbit+1]
            # level_codes_channelwise [num_levels, oc,nbit+1]
            for oc_idx in torch.arange(oc):
                if oc_idx == 0:
                    level_codes_t = level_codes[
                        torch.meshgrid(sort_id[:, oc_idx], torch.arange(nbit + 1, device=sort_id.device))].unsqueeze(1)
                    level_codes_channelwise = level_codes_t
                else:
                    level_codes_t = level_codes[
                        torch.meshgrid(sort_id[:, oc_idx], torch.arange(nbit + 1, device=sort_id.device))].unsqueeze(1)
                    level_codes_channelwise = torch.cat((level_codes_channelwise, level_codes_t), 1)

            # calculate output y and its binary code
            # y [K, K, iC, oC]
            # bits_y [K x K x iC, oC,nbit+1]
            reshape_x = torch.reshape(xp, [-1, oc])
            y = torch.zeros_like(xp) + levels[0]  # output
            zero_y = torch.zeros_like(xp)
            bits_y = torch.full([reshape_x.shape[0], oc, nbit + 1], -1., device=device)
            zero_bits_y = torch.zeros_like(bits_y)

            # [K x K x iC x oC] [1, oC]
            for i in torch.arange(num_levels - 1):
                g = torch.ge(xp, thrs[i])
                # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
                y = torch.where(g, zero_y + levels[i + 1], y)
                # [K x K x iC, oC,nbit+1]
                bits_y = torch.where(g.view(-1, oc, 1), zero_bits_y + level_codes_channelwise[i + 1], bits_y)

            # bits_y [K x K x iC, oc, nbit + 1] basis [nbit+1, oc] reshape_x [K x K x iC, oC]
            # B: [oC, K x K x iC, nbit + 1], v: [oC, nbit + 1, 1], w: [oC, K x K x iC, 1]
            B = bits_y.permute(1,0,2)
            BT = bits_y.permute(1,2,0)
            v = basis.permute(1, 0).unsqueeze(-1)
            w = reshape_x.permute(1, 0).unsqueeze(-1)
            # [oC, K x K x iC, nbit + 1] x [oC, nbit + 1, 1] - [oC, K x K x iC, 1]=> [oC, K x K x iC, 1]
            F = torch.matmul(B, v) - w
            loss1 = torch.norm(F,dim=(1,2),keepdim=True)

            # constrains
            # [num_levels-1, num_levels] x [num_levels, oc]
            # [oc, num_levels-1, 1]
            G = torch.matmul(variation_multiplier, levels).permute(1,0).unsqueeze(-1)
            # [oc, 1, num_levels-1] x [oc, num_levels-1, 1] => [oc,1,1]
            loss2 = torch.matmul(alpha.permute(1,0).unsqueeze(1), G)

            loss = torch.sum(loss1 + loss2)
            loss.backward()
            optimizer.step()

            # calculate BTxB
            # [oC, nbit, K x K x iC] x [oC, K x K x iC, nbit] => [oC, nbit, nbit]
            BTB = torch.matmul(BT, B)
            # [oC, nbit + 1, nbit + 1] x [oC, nbit + 1, 1] => [oC, nbit+1, 1]
            BTBv = torch.matmul(BTB, v)
            # calculate BTxX
            # bits_y [K x K x iC, oc, nbit] reshape_x [K x K x iC, oC]
            # [oC, nbit, K x K x iC] [oC, K x K x iC, 1] => [oC, nbit+1, 1]
            BTw = torch.matmul(BT, w)
            gF = 2 * BT.matmul(F)
            gloss =
            torch.matmul(variation_multiplier, levels)
            gradient_norm = torch

    # calculate levels and sort [2**nbit, oc] or [num_levels, oc]
    levels = torch.matmul(level_codes, basis)
    levels, sort_id = torch.sort(levels, 0)

    # calculate threshold
    # [num_levels-1, oc]
    thrs = torch.matmul(thrs_multiplier, levels)

    # calculate level codes per channel
    # ix:sort_id [num_levels, oc], iy: torch.arange(nbit)
    # level_codes = [num_levels, nbit + 1]
    # level_codes_channelwise [num_levels, oc, nbit + 1]
    for oc_idx in torch.arange(oc):
        if oc_idx == 0:
            level_codes_t = level_codes[
                torch.meshgrid(sort_id[:, oc_idx], torch.arange(nbit + 1, device=sort_id.device))].unsqueeze(1)
            level_codes_channelwise = level_codes_t
        else:
            level_codes_t = level_codes[
                torch.meshgrid(sort_id[:, oc_idx], torch.arange(nbit + 1, device=sort_id.device))].unsqueeze(1)
            level_codes_channelwise = torch.cat((level_codes_channelwise, level_codes_t), 1)

    return basis.transpose(1, 0), levels.transpose(1, 0), thrs.transpose(1, 0), level_codes_channelwise.transpose(1, 0)


def reshaped_QuantizedWeight(x, n, nbit=2, training=False):
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
    init_basis.append(x.mean((1, 2, 3)).tolist())

    num_levels = 2 ** nbit

    # initialize level multiplier
    # binary code of each level:
    # shape: [num_levels, nbit+1]
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
        level_multiplier_i.append(1.)
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


    # [nbit+1, oc]
    basis = torch.tensor(init_basis, dtype=torch.float32, requires_grad=False)
    # [2**nbit,nbit+1] or [num_levels,nbit+1]
    level_codes = torch.tensor(init_level_multiplier)
    # [num_levels-1, num_levels]
    thrs_multiplier = torch.tensor(init_thrs_multiplier)

    basis_t, levels_t, thrs_t, level_codes_t = Quantizer_train(x, basis, level_codes, thrs_multiplier, nbit, training)
    basis = basis_t.transpose(1, 0)
    levels = levels_t.transpose(1, 0)
    thrs = thrs_t.transpose(1, 0)
    level_codes_channelwise = level_codes_t.transpose(1, 0)

    # calculate output y and its binary code
    # y [K, K, iC, oC]
    # bits_y [K x K x iC, oC, nbit + 1]
    y = lq_weight.apply(x, levels, thrs, nbit)
    # [oC x iC x K x K] -> [K x K x iC x oC]
    xp = x.permute((3, 2, 1, 0))
    reshape_x = torch.reshape(xp, [-1, oc])
    bits_y = torch.full([reshape_x.shape[0], oc, nbit + 1], -1., device=device)
    zero_bits_y = torch.zeros_like(bits_y)

    # [K x K x iC x oC] [1, oC]
    for i in torch.arange(num_levels - 1):
        g = torch.ge(xp, thrs[i])
        # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
        # [K x K x iC, oC, nbit + 1]
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
        data = layer_act_val.numpy().ravel()
        sigma = np.std(data)
        mu = np.mean(data)

        # the histogram of the data
        # counts, bins = np.histogram(data, bins=bin)
        # n, bins, patches = plt.hist(bins[:-1], bins, weights=counts, density=True)
        if bins_in is None:
            bins = np.concatenate([np.array([data.min()]), threholds[i], np.array([data.max()])])
            plt.hist(data, bins=bins, density=norm, edgecolor='k', histtype='bar', rwidth=0.8, zorder=1)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('bin:', bins)
        else:
            counts, bins = np.histogram(data, bins=bins_in)
            plt.hist(data, bins=bins, density=norm, edgecolor='k', histtype='bar', rwidth=0.8, zorder=1)
        if basis is not None:
            ax_act.scatter(basis[i].data.numpy(), np.zeros_like(basis[i].data.numpy()), s=100,
                           edgecolors='g', c='r', marker='D', zorder=2)

        if fit:
            y = normfun(bins, mu, sigma)
            ax_act.plot(bins, y, '--')
        plt.tick_params(labelsize=20)
        ax_act.set_xlabel('value', fontsize=20)
        ax_act.set_ylabel('number', fontsize=20)
        ax_act.set_title('%s, $\mu=%.3f$, $\sigma=%.3f$' % (name, mu.item(), sigma.item()), fontsize=20)
        # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.models as models
    import math

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

    w = parm['layer1.0.conv1.weight'].view(4, 16 * 64, 3, 3)
    w = w + torch.arange(4).view(4, 1, 1, 1).float() / 10
    print(
        parm['layer1.0.conv1.weight'].view(4, 16 * 64, 3, 3).var(dim=(1, 2, 3), keepdim=True) * torch.arange(4).view(4,
                                                                                                                     1,
                                                                                                                     1,
                                                                                                                     1))
    OC, IC, K1, K2 = w.shape
    print("shape of the weight: ", w.shape)

    wq, levels, threholds = reshaped_QuantizedWeight(w, OC * K1 * K2, 2, True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    print('levels: ', levels)
    print('threshold: ', threholds)

    plot_histogram([w[i] for i in torch.arange(OC)], ['layer1.0.conv1, filter#%d' % i for i in range(OC)],
                   basis=levels, bins_in=16)
    print("=============")
    plot_histogram([w[i] for i in torch.arange(OC)], ['layer1.0.conv1, filter#%d' % i for i in range(OC)],
                   basis=levels, threholds=threholds)
    # plot_histogram([wq[i] for i in torch.arange(OC)], [str(i) for i in torch.arange(OC)])
    # print("w\n", w, "\nwq\n", wq)
