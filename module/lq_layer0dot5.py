import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adc import _adc
from .dac import _quantize_dac
from .w2g import w2g

quantize_input = _quantize_dac.apply
quantize_weight = _quantize_dac.apply
adc = _adc.apply
noise_amp = 0.015
from .learnable_quantize0dot5 import Quantizer_train, lq_weight

MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001
NORM_PPF_0_75 = 0.6745


# import visdom
# vis = visdom.Visdom(env ="reram_mnist_lq0")


class crxb_Conv2d(nn.Conv2d):
    """
    This is the custom conv layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers
    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It’s a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        vdd(float): supply voltage.
        enable_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 ir_drop, device, gmax, gmin, gwire, gload,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, input_qbit=32, weight_qbit=32, activation_qbit=32, scaler_dw=1,
                 is_first_layer=False, is_last_layer=False,
                 vdd=3.3, enable_noise=True, freq=10e6, temp=300, crxb_size=64,
                 enable_SAF=False, enable_ec_SAF=False):

        super(crxb_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.input_qbit = input_qbit
        self.weight_qbit = weight_qbit
        self.activation_qbit = activation_qbit
        self.scaler_dw = scaler_dw

        self.is_first_layer = torch.tensor([int(is_first_layer)])
        self.is_last_layer = torch.tensor([int(is_last_layer)])
        self.register_buffer('delta_in_sum', torch.zeros(1))
        self.register_buffer('delta_out_sum', torch.zeros(1))
        self.register_buffer('counter', torch.zeros(1))

        self.delta_x = self.delta_w = None
        self.delta_i = self.delta_y = None
        self.h_lvl_ma = 2 ** (input_qbit - 1) - 1 if self.is_first_layer.data else 2 ** (8 - 1) - 1
        self.h_lvl_w = 2 ** (weight_qbit - 1) - 1
        self.h_lvl_ia = 2 ** (activation_qbit - 1) - 1 if self.is_last_layer.data else 2 ** (8 - 1) - 1

        assert self.groups == 1, "currently not support grouped convolution for custom conv"

        self.ir_drop = ir_drop
        self.device = device

        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size
        self.enable_ec_SAF = enable_ec_SAF

        self.register_buffer('nchout_index', torch.arange(self.out_channels))
        weight_flatten_rows = self.in_channels * torch.cumprod(torch.tensor(self.kernel_size), 0)[-1].item()
        weight_flatten_cols = self.out_channels
        self.crxb_row, self.crxb_row_pads = self.num_pad(weight_flatten_rows, self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad(weight_flatten_cols, self.crxb_size)
        # p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        self.h_out = None
        self.w_out = None
        self.w_pad = [0, self.crxb_row_pads, 0, self.crxb_col_pads]
        self.input_pad = [0, 0, 0, self.crxb_row_pads]
        weight_crxb_shape = torch.Size((self.crxb_col, self.crxb_row,
                                        self.crxb_size, self.crxb_size))

        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl_v = 2 ** input_qbit
        self.h_lvl_g = 2 ** (weight_qbit - 1)
        # ReRAM cells
        self.Gmax = gmax  # max conductance
        self.Gmin = gmin  # min conductance
        self.delta_g = (self.Gmax - self.Gmin)  # conductance step
        self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax,
                       G_SA1=self.Gmin, weight_shape=weight_crxb_shape, enable_SAF=enable_SAF)
        self.Gwire = gwire
        self.Gload = gload
        # DAC
        self.Vdd = vdd  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl_v - 1)

        ################ Stochastic Conductance Noise setup #########################
        # parameters setup
        self.enable_noise = enable_noise
        self.freq = freq  # operating frequency
        self.kb = 1.38e-23  # Boltzmann const
        self.temp = temp  # temperature in kelvin
        self.q = 1.6e-19  # electron charge

        self.tau = 0.5  # Probability of RTN
        self.a = 1.662e-7  # RTN fitting parameter
        self.b = 0.0015  # RTN fitting parameter

        # =================== learn-able Quantized Weight ===================
        x = self.weight
        oc, ic, k1, k2 = x.shape
        nbit = self.weight_qbit
        n = oc * k1 * k2

        init_basis = []
        base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbit - 1))
        for j in range(nbit):
            init_basis.append([(2 ** j) * base for i in range(oc)])
        # init_basis.append(x.mean((1, 2, 3)).tolist())

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
        basis = torch.tensor(init_basis, dtype=torch.float32, requires_grad=False).to(device)
        self.register_buffer("basis", basis)
        # [2**nbit, lbit] or [num_levels, lbit]
        self.level_codes = torch.tensor(init_level_multiplier).to(device)
        # [num_levels-1, num_levels]
        self.thrs_multiplier = torch.tensor(init_thrs_multiplier).to(device)
        # inital level codes per channel [num_levels, oc,nbit+1]
        self.level_codes_channelwise = self.level_codes.unsqueeze(-2).expand(num_levels, oc, nbit + 1)

        # [num_levels, oc]
        self.register_buffer('levels', torch.matmul(self.level_codes, basis))
        # [num_levels-1, oc]
        self.register_buffer('thrs', torch.matmul(self.thrs_multiplier, self.levels))

    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() * self.scaler_dw
            if self.training:
                self.counter.data += 1
                self.delta_x = input.abs().max() / self.h_lvl_ma
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data / self.counter.data

        input_clip = F.hardtanh(input, min_val=-self.h_lvl_ma * self.delta_x.item(),
                                max_val=self.h_lvl_ma * self.delta_x.item())
        if self.delta_x == 0:
            input_quan = input
        else:
            input_quan = quantize_input(input_clip, self.delta_x) * self.delta_v  # convert to voltage

        # if self.delta_w == 0:
        #     self.weight_quan = self.weight
        # else:
        #     self.weight_quan = quantize_weight(self.weight, self.delta_w)

        # =================== train learn-able Quantized Weight start ===================
        if self.delta_w == 0:
            self.x = self.weight
        else:
            self.x = self.weight/self.delta_w

        oc, ic, k1, k2 = self.x.shape
        nbit = self.weight_qbit
        num_levels = 2 ** nbit
        lbit = nbit

        if self.training:
            basis_t, levels_t, thrs_t, level_codes_t = Quantizer_train(self.x.detach(), self.basis, self.levels,
                                                                       self.thrs, self.level_codes_channelwise,
                                                                       self.level_codes, self.thrs_multiplier,
                                                                       nbit, self.training)
            # alpha = alpha.detach()
            # basis = basis_t.transpose(1, 0).detach()
            self.levels = levels_t.transpose(1, 0).detach()
            self.thrs = thrs_t.transpose(1, 0).detach()
            self.level_codes_channelwise = level_codes_t.transpose(1, 0).detach()

        # calculate output y and its binary code
        # y [K, K, iC, oC]
        # bits_y [K x K x iC, oC, lbit]
        y = lq_weight.apply(self.x, self.levels, self.thrs, nbit)
        # # [oC x iC x K x K] -> [K x K x iC x oC]
        # xp = x.permute((3, 2, 1, 0))
        # reshape_x = torch.reshape(xp, [-1, oc])
        # bits_y = torch.full([reshape_x.shape[0], oc, lbit], -1., device=self.device)
        # zero_bits_y = torch.zeros_like(bits_y)
        #
        # # [K x K x iC x oC] [1, oC]
        # for i in torch.arange(num_levels - 1):
        #     g = torch.ge(xp, self.thrs[i])
        #     # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
        #     # [K x K x iC, oC, lbit]
        #     bits_y = torch.where(g.view(-1, oc, 1), zero_bits_y + self.level_codes_channelwise[i + 1], bits_y)

        self.weight_quan = y.contiguous()
        # =================== train learn-able Quantized Weight end===================

        # 2. Perform the computation between input voltage and weight conductance
        if self.h_out is None and self.w_out is None:
            self.h_out = int(
                (input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
            self.w_out = int(
                (input.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

        # 2.1 flatten and unfold the weight and input
        input_unfold = F.unfold(input_quan, kernel_size=self.kernel_size[0],
                                dilation=self.dilation, padding=self.padding,
                                stride=self.stride)
        weight_flatten = self.weight_quan.view(self.out_channels, -1)

        # 2.2. add paddings
        weight_padded = F.pad(weight_flatten, self.w_pad,
                              mode='constant', value=0)
        input_padded = F.pad(input_unfold, self.input_pad,
                             mode='constant', value=0)
        # 2.3. reshape to crxb size
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                       self.crxb_size, input_padded.shape[2])
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1, 2)
        # convert the floating point weight into conductance pair values
        G_crxb = self.w2g(weight_crxb)

        # 2.4. compute matrix multiplication followed by reshapes

        # this block is for introducing stochastic noise into ReRAM conductance
        if self.enable_noise:
            # rand_p = nn.Parameter(torch.Tensor(G_crxb.shape),
            #                       requires_grad=False)
            # rand_g = nn.Parameter(torch.Tensor(G_crxb.shape),
            #                       requires_grad=False)
            # if self.device.type == "cuda":
            #     rand_p = rand_p.cuda()
            #     rand_g = rand_g.cuda()
            # with torch.no_grad():
            #     input_reduced = (input_crxb.norm(p=2, dim=0).norm(p=2, dim=3).unsqueeze(dim=3)) / \
            #                     (input_crxb.shape[0] * input_crxb.shape[3])
            #     grms = torch.sqrt(
            #         G_crxb * self.freq * (4 * self.kb * self.temp + 2 * self.q * input_reduced) / (input_reduced ** 2) \
            #         + (self.delta_g / 3) ** 2)
            #
            #     grms[torch.isnan(grms)] = 0
            #     grms[grms.eq(float('inf'))] = 0
            #
            #     rand_p.uniform_()
            #     rand_g.normal_(0, 1)
            #     G_p = G_crxb * (self.b * G_crxb + self.a) / (G_crxb - (self.b * G_crxb + self.a))
            #     G_p[rand_p.ge(self.tau)] = 0
            #     G_g = grms * rand_g
            # G_crxb += (G_g.cuda() + G_p)
            # temp_var = torch.empty_like(G_crxb).normal_(0, 0.05/3)
            temp_var = torch.stack((torch.full_like(G_crxb[0], noise_amp),torch.full_like(G_crxb[1], -noise_amp)))
            # temp_var = torch.stack((torch.full_like(G_crxb[0], 0.15),torch.full_like(G_crxb[1], -0.15)))
            with torch.no_grad():
                G_crxb += G_crxb * temp_var

        # this block is to calculate the ir drop of the crossbar
        if self.ir_drop:
            from .IR_solver import IrSolver

            crxb_pos = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[0].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_pos.resetcoo()
            crxb_neg = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[1].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_neg.resetcoo()

            output_crxb = (crxb_pos.caliout() - crxb_neg.caliout())
            output_crxb = output_crxb.contiguous().view(self.crxb_col, self.crxb_row, self.crxb_size,
                                                        input.shape[0],
                                                        input_padded.shape[2])

            output_crxb = output_crxb.permute(3, 0, 1, 2, 4)

        else:
            output_crxb = torch.matmul(G_crxb[0], input_crxb) - \
                          torch.matmul(G_crxb[1], input_crxb)

        # 3. perform ADC operation (i.e., current to digital conversion)
        with torch.no_grad():
            if self.training:
                self.delta_i = output_crxb.abs().max() / self.h_lvl_ia
                self.delta_out_sum.data += self.delta_i
            else:
                self.delta_i = self.delta_out_sum.data / self.counter.data
            self.delta_y = self.delta_w * self.delta_x * \
                           self.delta_i / (self.delta_v * self.delta_g)
        #         print('adc LSB ration:', self.delta_i/self.max_i_LSB)
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl_ia * self.delta_i.item(),
                                 max_val=self.h_lvl_ia * self.delta_i.item())
        if self.delta_i == 0:
            output_adc = output_clip
        else:
            output_adc = adc(output_clip, self.delta_i, self.delta_y)

        if self.w2g.enable_SAF:
            if self.enable_ec_SAF:
                G_pos_diff, G_neg_diff = self.w2g.error_compensation()
                ec_scale = self.delta_y / self.delta_i
                output_adc += (torch.matmul(G_pos_diff, input_crxb)
                               - torch.matmul(G_neg_diff, input_crxb)) * ec_scale

        output_sum = torch.sum(output_adc, dim=2)
        output = output_sum.view(output_sum.shape[0],
                                 output_sum.shape[1] * output_sum.shape[2],
                                 self.h_out,
                                 self.w_out).index_select(dim=1, index=self.nchout_index)

        if self.bias is not None:
            bias_quan = quantize_weight(self.bias, self.delta_y) * self.delta_y
            a = bias_quan.unsqueeze(1).unsqueeze(1)
            output += a

        return output

    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0


class crxb_Linear(nn.Linear):
    """
    This is the custom linear layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers
    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It’s a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        vdd(float): supply voltage.
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        enable_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
    """

    def __init__(self, in_features, out_features,
                 ir_drop, device, gmax, gmin, gwire, gload,
                 bias=True, input_qbit=32, weight_qbit=32, activation_qbit=32, scaler_dw=1,
                 is_first_layer=False, is_last_layer=False,
                 vdd=3.3, enable_noise=True, freq=10e6, temp=300, crxb_size=64,
                 enable_SAF=False, enable_ec_SAF=False):
        super(crxb_Linear, self).__init__(in_features, out_features, bias)
        self.input_qbit = input_qbit
        self.weight_qbit = weight_qbit
        self.activation_qbit = activation_qbit
        self.scaler_dw = scaler_dw

        # self.register_buffer('is_first_layer', torch.tensor([int(is_first_layer)]))
        # self.register_buffer('is_last_layer', torch.tensor([int(is_last_layer)]))
        self.is_first_layer = torch.tensor([int(is_first_layer)])
        self.is_last_layer = torch.tensor([int(is_last_layer)])
        self.register_buffer('delta_in_sum', torch.zeros(1))
        self.register_buffer('delta_out_sum', torch.zeros(1))
        self.register_buffer('counter', torch.zeros(1))

        self.delta_x = self.delta_w = None
        self.delta_i = self.delta_y = None
        self.h_lvl_ma = 2 ** (input_qbit - 1) - 1 if self.is_first_layer.data else 2 ** (8 - 1) - 1
        self.h_lvl_w = 2 ** (weight_qbit - 1) - 1
        self.h_lvl_ia = 2 ** (activation_qbit - 1) - 1 if self.is_last_layer.data else 2 ** (8 - 1) - 1

        self.ir_drop = ir_drop
        self.device = device
        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size
        self.enable_ec_SAF = enable_ec_SAF

        self.register_buffer('out_index', torch.arange(out_features))
        self.crxb_row, self.crxb_row_pads = self.num_pad(self.weight.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad(self.weight.shape[0], self.crxb_size)
        # p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3) for last 1, 2, 3 dimension
        self.w_pad = [0, self.crxb_row_pads, 0, self.crxb_col_pads]
        self.input_pad = [0, self.crxb_row_pads]
        weight_crxb_shape = torch.Size((self.crxb_col, self.crxb_row,
                                        self.crxb_size, self.crxb_size))

        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl_v = 2 ** input_qbit
        self.h_lvl_g = 2 ** (weight_qbit - 1)
        # ReRAM cells
        self.Gmax = gmax  # max conductance
        self.Gmin = gmin  # min conductance
        self.delta_g = (self.Gmax - self.Gmin)   # conductance step
        self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax,
                       G_SA1=self.Gmin, weight_shape=weight_crxb_shape, enable_SAF=enable_SAF)
        self.Gwire = gwire
        self.Gload = gload
        # DAC
        self.Vdd = vdd  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl_v - 1)

        ################ Stochastic Conductance Noise setup #########################
        # parameters setup
        self.enable_noise = enable_noise
        self.freq = freq  # operating frequency
        self.kb = 1.38e-23  # Boltzmann const
        self.temp = temp  # temperature in kelvin
        self.q = 1.6e-19  # electron charge

        self.tau = 0.5  # Probability of RTN
        self.a = 1.662e-7  # RTN fitting parameter
        self.b = 0.0015  # RTN fitting parameter

        # =================== learn-able Quantized Weight ===================
        x = self.weight.unsqueeze(-1).unsqueeze(-1)
        oc, ic, k1, k2 = x.shape
        nbit = self.weight_qbit
        n = oc * k1 * k2

        init_basis = []
        base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbit - 1))
        for j in range(nbit):
            init_basis.append([(2 ** j) * base for i in range(oc)])
        # init_basis.append(x.mean((1, 2, 3)).tolist())

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
        basis = torch.tensor(init_basis, dtype=torch.float32, requires_grad=False).to(device)
        self.register_buffer("basis", basis)
        # [2**nbit, lbit] or [num_levels, lbit]
        self.level_codes = torch.tensor(init_level_multiplier).to(device)
        # [num_levels-1, num_levels]
        self.thrs_multiplier = torch.tensor(init_thrs_multiplier).to(device)
        # inital level codes per channel [num_levels, oc,nbit+1]
        self.level_codes_channelwise = self.level_codes.unsqueeze(-2).expand(num_levels, oc, nbit + 1)

        # [num_levels, oc]
        self.register_buffer('levels', torch.matmul(self.level_codes, basis))
        # [num_levels-1, oc]
        self.register_buffer('thrs', torch.matmul(self.thrs_multiplier, self.levels))

    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() * self.scaler_dw
            if self.training:
                self.counter.data += 1
                self.delta_x = input.abs().max() / self.h_lvl_ma
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data / self.counter.data

        input_clip = F.hardtanh(input, min_val=-self.h_lvl_ma * self.delta_x.item(),
                                max_val=self.h_lvl_ma * self.delta_x.item())
        if self.delta_x == 0:
            input_quan = input
        else:
            input_quan = quantize_input(input_clip, self.delta_x) * self.delta_v  # convert to voltage

        # if self.delta_w == 0:
        #     self.weight_quan = self.weight
        # else:
        #     self.weight_quan = quantize_weight(self.weight, self.delta_w)

        # =================== train learn-able Quantized Weight start ===================
        if self.delta_w == 0:
            self.x = self.weight.unsqueeze(-1).unsqueeze(-1)
        else:
            self.x = self.weight.unsqueeze(-1).unsqueeze(-1)/self.delta_w

        oc, ic, k1, k2 = self.x.shape
        nbit = self.weight_qbit
        num_levels = 2 ** nbit
        lbit = nbit

        if self.training:
            basis_t, levels_t, thrs_t, level_codes_t = Quantizer_train(self.x.detach(), self.basis, self.levels,
                                                                       self.thrs, self.level_codes_channelwise,
                                                                       self.level_codes, self.thrs_multiplier,
                                                                       nbit, self.training)
            # alpha = alpha.detach()
            # basis = basis_t.transpose(1, 0).detach()
            self.levels = levels_t.transpose(1, 0).detach()
            self.thrs = thrs_t.transpose(1, 0).detach()
            self.level_codes_channelwise = level_codes_t.transpose(1, 0).detach()

        # calculate output y and its binary code
        # y [K, K, iC, oC]
        # bits_y [K x K x iC, oC, lbit]
        y = lq_weight.apply(self.x, self.levels, self.thrs, nbit)
        # # [oC x iC x K x K] -> [K x K x iC x oC]
        # xp = x.permute((3, 2, 1, 0))
        # reshape_x = torch.reshape(xp, [-1, oc])
        # bits_y = torch.full([reshape_x.shape[0], oc, lbit], -1., device=self.device)
        # zero_bits_y = torch.zeros_like(bits_y)
        #
        # # [K x K x iC x oC] [1, oC]
        # for i in torch.arange(num_levels - 1):
        #     g = torch.ge(xp, self.thrs[i])
        #     # [K, K, iC, oC] + [1, oC], [K, K, iC, oC] => [K, K, iC, oC]
        #     # [K x K x iC, oC, lbit]
        #     bits_y = torch.where(g.view(-1, oc, 1), zero_bits_y + self.level_codes_channelwise[i + 1], bits_y)

        self.weight_quan = y.squeeze(-1).squeeze(-1).contiguous()
        # =================== train learn-able Quantized Weight end===================

        # 2. Perform the computation between input voltage and weight conductance
        # 2.1. skip the input unfold and weight flatten for fully-connected layers
        # 2.2. add padding
        weight_padded = F.pad(self.weight_quan, self.w_pad,
                              mode='constant', value=0)
        input_padded = F.pad(input_quan, self.input_pad,
                             mode='constant', value=0)
        # print(input_padded.shape,input_quan.shape,input.shape)
        # 2.3. reshape
        input_crxb = input_padded.view(
            input.shape[0], 1, self.crxb_row, self.crxb_size, 1)
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1, 2)
        # convert the floating point weight into conductance pair values
        G_crxb = self.w2g(weight_crxb)

        # 2.4. compute matrix multiplication
        # this block is for introducing stochastic noise into ReRAM conductance
        if self.enable_noise:
            # rand_p = nn.Parameter(torch.Tensor(G_crxb.shape),
            #                       requires_grad=False)
            # rand_g = nn.Parameter(torch.Tensor(G_crxb.shape),
            #                       requires_grad=False)
            # if self.device.type == "cuda":
            #     rand_p = rand_p.cuda()
            #     rand_g = rand_g.cuda()
            # with torch.no_grad():
            #     input_reduced = input_crxb.norm(p=2, dim=0).norm(p=2, dim=3).unsqueeze(dim=3) / (
            #             input_crxb.shape[0] * input_crxb.shape[3])
            #     grms = torch.sqrt(
            #         G_crxb * self.freq * (4 * self.kb * self.temp + 2 * self.q * input_reduced) / (input_reduced ** 2) \
            #         + (self.delta_g / 3 / 128) ** 2)  # G_crxb / 3 / 128
            #
            #     grms[torch.isnan(grms)] = 0
            #     grms[grms.eq(float('inf'))] = 0
            #
            #     rand_p.uniform_()
            #     rand_g.normal_(0, 1)
            #     G_p = G_crxb * (self.b * G_crxb + self.a) / (G_crxb - (self.b * G_crxb + self.a))
            #     G_p[rand_p.ge(self.tau)] = 0
            #     G_g = grms * rand_g
            #
            # G_crxb += (G_g + G_p)
            # temp_var = torch.empty_like(G_crxb).normal_(0, 0.05/3)
            temp_var = torch.stack((torch.full_like(G_crxb[0], noise_amp), torch.full_like(G_crxb[1], -noise_amp)))
            # temp_var = torch.stack((torch.full_like(G_crxb[0], 0.15),torch.full_like(G_crxb[1], -0.15)))
            with torch.no_grad():
                G_crxb += G_crxb * temp_var

        # this block is to calculate the ir drop of the crossbar

        if self.ir_drop:
            from .IR_solver import IrSolver

            crxb_pos = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[0].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_pos.resetcoo()
            crxb_neg = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[1].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_neg.resetcoo()

            output_crxb = (crxb_pos.caliout() - crxb_neg.caliout())
            output_crxb = output_crxb.contiguous().view(self.crxb_col,
                                                        self.crxb_row,
                                                        self.crxb_size,
                                                        input.shape[0],
                                                        1)

            output_crxb = output_crxb.permute(3, 0, 1, 2, 4)

        else:
            output_crxb = torch.matmul(G_crxb[0], input_crxb) \
                          - torch.matmul(G_crxb[1], input_crxb)

        # 3. perform ADC operation (i.e., current to digital conversion)
        with torch.no_grad():
            if self.training:
                self.delta_i = output_crxb.abs().max() / (self.h_lvl_ia)
                self.delta_out_sum.data += self.delta_i
            else:
                self.delta_i = self.delta_out_sum.data / self.counter.data
            self.delta_y = self.delta_w * self.delta_x * \
                           self.delta_i / (self.delta_v * self.delta_g)
        #         print('adc LSB ration:', self.delta_i/self.max_i_LSB)
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl_ia * self.delta_i.item(),
                                 max_val=self.h_lvl_ia * self.delta_i.item())
        if self.delta_i == 0:
            output_adc = output_crxb
        else:
            output_adc = adc(output_clip, self.delta_i, self.delta_y)

        if self.w2g.enable_SAF:
            if self.enable_ec_SAF:
                G_pos_diff, G_neg_diff = self.w2g.error_compensation()
                ec_scale = self.delta_y / self.delta_i
                output_adc += (torch.matmul(G_pos_diff, input_crxb)
                               - torch.matmul(G_neg_diff, input_crxb)) * ec_scale

        output_sum = torch.sum(output_adc, dim=2).squeeze(dim=3)
        output = output_sum.view(input.shape[0],
                                 output_sum.shape[1] * output_sum.shape[2]).index_select(dim=1, index=self.out_index)

        if self.bias is not None:
            bias_quan = quantize_weight(self.bias, self.delta_y) * self.delta_y
            output += bias_quan

        return output

    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0

# Change list:
# 2020/03/15: fixed the bias bug
# 2020/03/25: divide the quatization bits into three class, IA, W, MA.
# 2020/03/25: add layer number flag: is_first_layer, is_last_layer
# 2020/04/19: add layer number flag: is_first_layer, is_last_layer