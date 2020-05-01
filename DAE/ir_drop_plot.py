# coding=gbk
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from typing import TypeVar, Iterable, Tuple, List, Optional

def irdrop_out_error_plot(args, size, vnode: np.ndarray, vin: np.ndarray,
                          i_list: List[np.ndarray], i_list_name: List[str]):
    ideal_vdevice = vin.reshape(vin.shape[0], 1)
    ideal_vdevice = np.tile(ideal_vdevice, (1, size))

    # 生成三维数据
    xx = np.arange(size)
    yy = np.arange(size)
    X, Y = np.meshgrid(xx, yy)
    Z1 = (ideal_vdevice - (vnode[0] - vnode[1]))[X, Y]
    Z2 = vnode[0][X, Y]
    Z3 = vnode[1][X, Y]
    # print(Z2)
    # print(Z3)

    # 定义坐标轴
    # fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig = plt.figure(figsize=(30, 20))
    object_3d = [Z1, Z2, Z3]
    for plt_num in range(len(object_3d)):
        Z_obj = object_3d[plt_num]
        ax1 = fig.add_subplot(2, 3, 1 + plt_num, projection='3d')
        surf = ax1.plot_surface(X, Y, Z_obj, rstride=1, cstride=1, alpha=0.5,
                                cmap='winter')  # 生成表面， alpha 用于控制透明度
        Voltage_range = (np.max(Z_obj) - np.min(Z_obj))
        base_plain = np.min(Z_obj)
        offset = base_plain - 0.5 * Voltage_range

        ax1.contourf(X, Y, Z_obj, zdir='z', offset=offset, cmap="rainbow")  # 生成z方向投影，投到x-y平面
        ax1.contour(X, Y, Z_obj, zdir='x', offset=(size - 1), cmap="rainbow")  # 生成x方向投影，投到y-z平面
        ax1.contour(X, Y, Z_obj, zdir='y', offset=0, cmap="rainbow")  # 生成y方向投影，投到x-z平面
        # ax1.contourf(X,Y,Z_obj,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影填充，投到x-z平面，contourf()函数

        # 设定显示范围
        plt.tick_params(labelsize=20)  # 刻度字体大小20
        ax1.set_xlabel('row', fontsize=20)
        ax1.set_ylabel('col', fontsize=20)
        ax1.set_zlabel('voltage(v)', fontsize=20)

        ax1.set_xlim(0, size - 1, auto=True)  # 拉开坐标轴范围显示投影
        ax1.set_ylim(0, size - 1, auto=True)
        ax1.set_zlim(offset, offset + 2 * Voltage_range, auto=True)
        # 颜色条
        cb = fig.colorbar(surf, shrink=0.5, aspect=5)
        cb.set_label('Voltage', fontsize=20)
        # change color bar text size
        cb.ax.tick_params(labelsize=20)

    # plot actual output and ideal output after correction
    ax2 = fig.add_subplot(2, 3, 4)
    for i_ix, i in enumerate(i_list):
        ax2.plot(range(i.shape[0]), i, label=i_list_name[i_ix])

    # Places a legend on the axes
    plt.legend()
    plt.tick_params(labelsize=20)
    plt.show()
