from typing import Optional

import torch


def train_backtracking(function: callable, weight: torch.Tensor, g_decent: torch.Tensor, g_search: torch.Tensor,
                       alpha: Optional[float] = 1., gama: Optional[float] = 0.8, cls: Optional[float] = 1.):
    """

    :param weight:
    :param g_decent: function gradient
    :param g_search:
    :param alpha: initial step back track rate
    :param gama: step back track rate
    :param cls: initial learning rate
    :return:
    """
    f_decent = alpha * cls * torch.matmul(g_decent.transpose(-1, -2), g_search).sum()
    weight_n = weight + alpha * g_decent
    f_next = function(weight_n)
    f_current = function(weight)

    flag = f_next - f_current - f_decent
    times = 0
    while (flag > 0):
        alpha = gama * alpha
        f_decent = alpha * cls * torch.matmul(g_decent.transpose(-1, -2), g_search).sum()
        f_next = function(weight + alpha * g_decent)
        f_current = function(weight)
        flag = f_next - f_current - f_decent
        times += 1
    # print("search %d times\n" % times)
    # print("f_next:", f_next.item())
    return alpha, f_next.item()


def lagrange_loss(loss_f, constrain_f, lambda_lag):
    """
    Args:
        loss_f: callable
        constrain_f: torch.Tensor
        lambda_lag: torch.Tensor
    Returns:
    """
    return lambda param: loss_f(param) + torch.matmul(lambda_lag.tranpose(-1, -2), constrain_f(param))


def is_better(a, best, mode='min', threshold=1e-4, threshold_mode='rel'):
    if mode == 'min' and threshold_mode == 'rel':
        rel_epsilon = 1. - threshold
        return a < best * rel_epsilon

    elif mode == 'min' and threshold_mode == 'abs':
        return a < best - threshold

    elif mode == 'max' and threshold_mode == 'rel':
        rel_epsilon = threshold + 1.
        return a > best * rel_epsilon

    else:  # mode == 'max' and epsilon_mode == 'abs':
        return a > best + threshold


def is_best(current, best, num_bad_epochs, patience=2):
    current = float(current)
    if is_better(current, best):
        best = current
        num_bad_epochs = 0
    else:
        num_bad_epochs += 1
    return num_bad_epochs > patience, best, num_bad_epochs


def gradient_decent(weight: torch.Tensor, f_gradient: callable, epi: float, lr: float = 1., line_search: bool = False,
                    f: Optional[callable] = None, gama: Optional[float] = 0.8, cls: Optional[float] = 1):
    """
    :param cls:
    :param gama:
    :param g_function:
    :param g_search:
    :param f:
    :param weight:
    :param f_gradient:
    :param epi:
    :param lr:
    :param line_search:
    :return:
    """
    g_current = f_gradient(weight)
    l_current = l_best = float('inf')
    num_bad_epochs = 0
    is_best_flag = False
    while g_current.norm() > epi and not is_best_flag:
        if line_search:
            lr, l_current = train_backtracking(f, weight, g_current, g_current, lr, gama, cls)
        weight = weight - lr * g_current
        g_current = f_gradient(weight)
        is_best_flag, l_best, num_bad_epochs = is_best(l_current, l_best, num_bad_epochs)
        # print('g_current norm', g_current.norm())
        # print('weight:',weight)
    return weight


def LU_decomposition(A):
    device = A.device
    M = A.shape[-2]
    N = A.shape[-1]
    L = torch.diag(torch.ones(M)).to(device)
    R = A.clone()
    for i in range(N):
        G = torch.diag(torch.ones(M)).to(device)
        G_inv = torch.diag(torch.ones(M)).to(device)
        # TODO: replace R[i, i] by first non-zero element at ith column
        if R[i, i]:
            # print("normal\n")
            G[i + 1:, i] = -R[i + 1:, i] / R[i, i]
            G_inv[i + 1:, i] = R[i + 1:, i] / R[i, i]
            L = torch.matmul(L, G_inv)
            # print("G:", G, '\n')
            # print("R:", R, '\n')
            R = torch.matmul(G, R)
        else:
            ix = torch.nonzero(R[i:, i])[0]
            if ix.size():
                # print("permute\n")
                P = torch.diag(torch.ones(M)).to(device)
                P[ix + i, ix + i] = P[i, i] = 0
                P[ix + i, i] = P[i, ix + i] = 1
                # print('P:',P,'\n')
                # print("R:", R, '\n')
                L = torch.matmul(L, P)
                R = torch.matmul(P, R)
                G[i + 1:, i] = -R[i + 1:, i] / R[i, i]
                G_inv[i + 1:, i] = R[i + 1:, i] / R[i, i]
                L = torch.matmul(L, G_inv)
                # print("G:", G, '\n')
                # print("R:", R, '\n')
                R = torch.matmul(G, R)
            else:
                pass
    return L, R


if __name__ == "__main__":
    # A = torch.tensor([[17, 24, 1, 8, 15],
    #                   [23, 5, 7, 14, 16],
    #                   [4, 6, 13, 20, 22],
    #                   [10, 12, 19, 21, 3],
    #                   [11, 18, 25, 2, 9]]).float()
    A = torch.tensor([[1., 2., 3.],
                      [1., 2., 3.],
                      [2., 4., 6.],
                      [1., 1., 1.],
                      [1., 2., 1.],
                      [2., 4., 2.]])
    L, U = LU_decomposition(A)
    print(U)
    print(U[torch.any(U!=0,dim=-1)])