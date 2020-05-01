import argparse

import numpy as np
import torch
from torch.utils.data import TensorDataset

from DAE.ir_drop_plot import irdrop_out_error_plot
from module.IR_solver import IrSolver

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

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
parser.add_argument('--freq', type=float, default=10e6,
                    help='scaler to compress the conductance')
parser.add_argument('--temp', type=float, default=300,
                    help='scaler to compress the conductance')

# device init cfg
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

from typing import Tuple

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")


class IR_solver(IrSolver):
    def __init__(self, Rsize: int, Csize: int, Gwire: float, Gload: int,
                 input_x: torch.Tensor, Gmat: torch.Tensor, device=torch.device("cuda:0")):
        # Gmat size:  (crxb_Rsize, crxb_Csize, crxb_col, crxb_row)
        # input size: (crxb_Rsize, N, 1,     crxb_row, L)
        super(IR_solver, self).__init__(Rsize, Csize, Gwire, Gload, input_x, Gmat, device=device)
        self.device = device
        self.node_sp = None
        self.node_i = self.node_v = None

    def resetcoo(self):
        super(IR_solver, self).resetcoo()

    def _add_data(self, row_data, col_data, data_data):
        super(IR_solver, self)._add_data(row_data, col_data, data_data)

    def _nodematgen(self):
        """This function generates the node conductance matrix. The node conductance matrix is batched
        according to dimension of the input tensors. The detailed descrapition of the node conductance matrix please to
        this link: https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA1.html

        Args:
            None

        Returns:
            The conductance matrix in coo format.
            current_mat (tensor.float): the current matrix.
        """

        # Gmat size:  (crxb_Rsize, crxb_Csize, crxb_col, crxb_row)
        # input size: (crxb_Rsize, N, 1,     crxb_row, L)
        # current_mat:(GRsize * 2 * GCsize, N, 1, crxb_row, L)
        # num_nonzero: GRsize*(GCsize-2)*4 + GRsize*2*3 + (GRsize-2)*GCsize*4 + 2*GCsize*3
        # mat_data:   (num_nonzero,crxb_col, crxb_row)
        # row_data:   (num_nonzero)
        # col_data:   (num_nonzero)
        # current_mat:(GRsize * 2 * GCsize, N, 1, crxb_row, L)
        # node_sp:    (GRsize * 2 * GCsize, GRsize * 2 * GCsize, crxb_col, crxb_row)

        # current_mat = torch.zeros(self.GRsize ** 2 * 2, self.input_x.shape[1], self.input_x.shape[2],
        #                           self.input_x.shape[3], self.input_x.shape[4])
        extender = torch.ones(self.Gmat.size()[2], self.Gmat.size()[3])
        # turn the matrix G into batches

        electrode = ['top', 'bot']
        counter = 0

        for row in range(self.GRsize):
            for ele in electrode:
                for col in range(self.GCsize):
                    if col == 0 and ele == 'top':  # edge type I
                        # current_mat[counter] = self.input_x[row] * self.Gload
                        self._add_data(counter, counter, self.Gload + self.Gmat[row][col] + self.Gwire)
                        self._add_data(counter, counter + 1, -self.Gwire * extender)
                        self._add_data(counter, counter + self.GRsize, -self.Gmat[row][col])

                    elif row == 0 and ele == 'bot':  # edge type II
                        self._add_data(counter, counter, self.Gmat[row][col] + self.Gwire)
                        self._add_data(counter, counter + 2 * self.GRsize, -self.Gwire * extender)
                        self._add_data(counter, counter - self.GRsize, -self.Gmat[row][col])

                    elif col == self.GCsize - 1 and ele == 'top':  # edge type III
                        self._add_data(counter, counter, self.Gmat[row][col] + self.Gwire)
                        self._add_data(counter, counter - 1, -self.Gwire * extender)
                        self._add_data(counter, counter + self.GRsize, -self.Gmat[row][col])

                    elif row == self.GRsize - 1 and ele == 'bot':  # edge type IV
                        self._add_data(counter, counter, self.Gload + self.Gmat[row][col] + self.Gwire)
                        self._add_data(counter, counter - 2 * self.GRsize, -self.Gwire * extender)
                        self._add_data(counter, counter - self.GRsize, -self.Gmat[row][col])

                    else:
                        if ele == 'top':
                            self._add_data(counter, counter, self.Gmat[row][col] + 2 * self.Gwire)
                            self._add_data(counter, counter + 1, -self.Gwire * extender)
                            self._add_data(counter, counter - 1, -self.Gwire * extender)
                            self._add_data(counter, counter + self.GCsize, -self.Gmat[row][col])

                        elif ele == 'bot':
                            self._add_data(counter, counter, self.Gmat[row][col] + 2 * self.Gwire)
                            self._add_data(counter, counter + (2 * self.GRsize), -self.Gwire * extender)
                            self._add_data(counter, counter - (2 * self.GRsize), -self.Gwire * extender)
                            self._add_data(counter, counter - self.GRsize, -self.Gmat[row][col])

                    counter += 1

        # sparse tensor
        self.node_i = torch.LongTensor([self.mat_row, self.mat_col]).to(self.device)
        self.node_v = torch.stack(self.mat_data).to(self.device)
        self.resetcoo()
        # Generate the node conductace G
        self.node_sp = torch.sparse.FloatTensor(self.node_i, self.node_v)
        print(torch.matrix_rank(self.node_sp.to_dense().permute(2, 3, 0, 1)[0,0]))
        pass

    def _nodemat_update(self, Gmat: torch.Tensor):
        self.Gmat = Gmat
        self._nodematgen()

    def _gen_mna_z(self, input_x):
        current_mat = torch.zeros(self.GRsize ** 2 * 2, input_x.shape[1], input_x.shape[2],
                                  input_x.shape[3], input_x.shape[4])

        electrode = ['top', 'bot']
        counter = 0

        for row in range(self.GRsize):
            for ele in electrode:
                for col in range(self.GCsize):
                    if col == 0 and ele == 'top':  # edge type I
                        current_mat[counter] = input_x[row] * self.Gload
                    counter += 1
        return current_mat

    def _solve(self, input_x, detial=False) -> Tuple[torch.Tensor, ...]:
        """This function is to calcuate the output of the current of the corssbar

        Args:
            None

        Retures:
            output current of the crossbar w.r.t. IR drop
        """

        # current_mat:(GRsize * 2 * GCsize, N, 1, crxb_row, L)
        # node_sp:    (GRsize * 2 * GCsize, GRsize * 2 * GCsize, crxb_col, crxb_row)

        self._nodematgen()
        # start1 = time.time()
        current_mat = self._gen_mna_z(input_x).to(self.device)
        # Generate the current array I of the MNA, which solve the node voltages using GV = I

        # b: (1,        crxb_row, GRsize * 2 * GCsize, N*L)
        # A: (crxb_col, crxb_row, GRsize * 2 * GCsize, GRsize * 2 * GCsize)
        nodes, _ = torch.solve(current_mat.permute(2, 3, 0, 1, 4).contiguous().view(current_mat.size()[2],
                                                                                    current_mat.size()[3],
                                                                                    current_mat.size()[0],
                                                                                    -1),
                               self.node_sp.to_dense().permute(2, 3, 0, 1))
        # nodes： (crxb_col, crxb_row, GRsize * 2 * GCsize, N*L)
        # Solve batched linear systems
        del _, current_mat
        temp = nodes.shape[2]
        # outcurrent： (crxb_col, crxb_row, GCsize, N*L)
        outcurrent = nodes[:, :, temp - self.GCsize:temp, :]
        try:
            outcurrent = outcurrent * self.Gload
        except:
            outcurrent = outcurrent * self.Gload
        if not detial:
            del nodes
            return outcurrent
        else:
            return outcurrent, nodes

    def _ideal(self, input_x):
        output_crxb_i = torch.matmul(G_crxb[0], input_crxb)
        return output_crxb_i

    def gen_data(self):
        x_data = torch.empty(0)
        y_data = torch.empty(0)
        for m in range(self.GRsize):
            vin = torch.zeros(self.GRsize)
            vin[m] = self.num_v_level
            ideal_vout = self.ideal_forward(vin)
            # dataset_size.append((vin, ideal_vout))
            x_data = torch.cat((x_data, vin.view(1, -1)))
            y_data = torch.cat((y_data, ideal_vout.view(1, -1)))

        # define input
        vin = torch.full((self.GRsize,), self.num_v_level)
        ideal_vout = self._ideal(vin)
        # dataset_size.append((vin, ideal_vout))
        x_data = torch.cat((x_data, vin.view(1, -1)))
        y_data = torch.cat((y_data, ideal_vout.view(1, -1)))
        np.save("dataset/vin_data.npy", x_data.numpy())
        np.save("dataset/ideal_vout", y_data.numpy())
        deal_dataset = TensorDataset(x_data, y_data)
        print(len(deal_dataset))
        return deal_dataset


print("+++", args)

N = 1
crxb_row = 1
crxb_col = 1
L = 1

# for logic:
# Gmat size:  (crxb_col, crxb_row, crxb_Csize, crxb_Rsize)
# input size: (N,1,      crxb_row, crxb_Rsize, L)
# ouput size: (N, crxb_col, crxb_row, crxb_Csize, L)

input_crxb = torch.rand(N, 1, crxb_row, args.crxb_size, L).to(device) * args.vdd  # generating input
weight = torch.rand(crxb_col, crxb_row, args.crxb_size, args.crxb_size).to((device))
G_p = torch.where(weight > 0, weight, torch.zeros_like(weight)) * (args.gmax - args.gmin) + args.gmin
G_n = torch.where(weight < 0, weight, torch.zeros_like(weight)) * (args.gmax - args.gmin) + args.gmin
G_crxb = torch.stack((G_p, G_n))

# for Irsolver:
# Gmat size:  (crxb_Rsize, crxb_Csize, crxb_col, crxb_row)
# input size: (crxb_Rsize, N, 1,     crxb_row, L)
# ouput size: (crxb_col, crxb_row, crxb_Csize, N, L)


crxb_pos = IR_solver(Rsize=args.crxb_size,
                     Csize=args.crxb_size,
                     Gwire=args.gwire,
                     Gload=args.gload,
                     input_x=input_crxb.permute(3, 0, 1, 2, 4),
                     Gmat=G_crxb[0].permute(3, 2, 0, 1),
                     device=device)
crxb_neg = IR_solver(Rsize=args.crxb_size,
                     Csize=args.crxb_size,
                     Gwire=args.gwire,
                     Gload=args.gload,
                     input_x=input_crxb.permute(3, 0, 1, 2, 4),
                     Gmat=G_crxb[1].permute(3, 2, 0, 1),
                     device=device)
i_p, v_p = crxb_pos._solve(input_crxb.permute(3, 0, 1, 2, 4), detial=True)
i_n, v_n = crxb_neg._solve(input_crxb.permute(3, 0, 1, 2, 4), detial=True)

# outcurrent： (crxb_col, crxb_row, GCsize, N*L)
# outcurrent： (N, crxb_col, crxb_row, GCsize, L)
i_crxb = (i_p - i_n).contiguous().view(crxb_col, crxb_row, args.crxb_size,
                                       input_crxb.shape[0],
                                       input_crxb.shape[-1])

i_crxb = i_crxb.permute(3, 0, 1, 2, 4)

i_crxb_ideal = torch.matmul(G_crxb[0], input_crxb) \
               - torch.matmul(G_crxb[1], input_crxb)

# nodes： (crxb_col, crxb_row, GRsize * 2 * GCsize, N*L) ->
# (2, N, crxb_col, crxb_row, GRsize, GCsize, L)
v_p_t = v_p.view(crxb_col, crxb_row,
                 args.crxb_size, 2, args.crxb_size,
                 input_crxb.shape[0],
                 input_crxb.shape[-1]).permute(3, 5, 0, 1, 2, 4, 6)
v_n_t = v_n.view(crxb_col, crxb_row,
                 args.crxb_size, 2, args.crxb_size,
                 input_crxb.shape[0],
                 input_crxb.shape[-1]).permute(3, 5, 0, 1, 2, 4, 6)

#    (N, crxb_col, crxb_row, GRsize, GCsize, L)/
#       (crxb_col, crxb_row, GCsize, GRsize, 1)
# -> (N, crxb_col, crxb_row, GRsize, GCsize, L)
# ->  (N, crxb_col, crxb_row, GCsize, L)
i_branch_p = (v_p_t[0] - v_p_t[1]) * G_p.unsqueeze(-1)
i_branch_n = (v_n_t[0] - v_p_t[1]) * G_n.unsqueeze(-1)
i_crxb_eq = i_branch_p.sum(-3) - i_branch_n.sum(-3)

# input_crxb：(N, 1,        crxb_row, GRsize, 1,      L)
# vnode_crxb: (N, crxb_col, crxb_row, GRsize, GCsize, L)
# gnode_crxb: (   crxb_col, crxb_row, GCsize, GRsize, 1)

# TODO: add least square
G_crxb_comp = torch.stack(((input_crxb.unsqueeze(-2) / (v_p_t[0] - v_p_t[1])).abs().sqrt() * G_p,
                           (input_crxb.unsqueeze(-2) / (v_n_t[0] - v_n_t[1])).abs().sqrt() * G_n))[:,0,:,:,:,:,0]

crxb_pos._nodemat_update(G_crxb_comp[0].permute(3, 2, 0, 1))
crxb_neg._nodemat_update(G_crxb_comp[1].permute(3, 2, 0, 1))

i_p, v_p = crxb_pos._solve(input_crxb.permute(3, 0, 1, 2, 4), detial=True)
i_n, v_n = crxb_neg._solve(input_crxb.permute(3, 0, 1, 2, 4), detial=True)

# outcurrent： (crxb_col, crxb_row, GCsize, N*L)
# outcurrent： (N, crxb_col, crxb_row, GCsize, L)
i_crxb_comp = (i_p - i_n).contiguous().view(crxb_col, crxb_row, args.crxb_size,
                                            input_crxb.shape[0],
                                            input_crxb.shape[-1])

i_crxb_comp = i_crxb_comp.permute(3, 0, 1, 2, 4)

irdrop_out_error_plot(args, args.crxb_size, v_p_t[:, 0, 0, 0, :, :, 0].cpu().numpy(),
                      input_crxb[0, 0, 0, :, 0].cpu().numpy(),
                      [i_crxb_ideal[0, 0, 0, :, 0].cpu().numpy(), i_crxb[0, 0, 0, :, 0].cpu().numpy(),
                       i_crxb_eq[0, 0, 0, :, 0].cpu().numpy(), i_crxb_comp[0, 0, 0, :, 0].cpu().numpy()],
                      ['i_ideal', 'i_crxb', 'i_eq', 'i_comp'])
