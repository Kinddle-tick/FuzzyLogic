#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DrawPicture.py
# @Time      :2023/3/30 3:24 PM
# @Author    :Kinddle
from Project1.FuzzyMF import *
from matplotlib import pyplot as plt
import os

Filename_front = ""
Save_dir_root = "Simulation"
Save_dir_leaf = "Singleton_default_max-min_min_height"


def genPath(filename):
    return os.path.join(Save_dir_root, Save_dir_leaf, "{}{}".format(Filename_front, filename))

if __name__ == '__main__':
    Z_tmp = np.load(genPath("raw_data.npy"))    # (x2,x3,x1)
    step_x2, step_x3,_ = Z_tmp.shape
    x1 = [3, 7]
    x2 = np.linspace(0, 10, step_x2)
    x3 = np.linspace(0, 10, step_x3)

    Z1 = Z_tmp[:, :, 0]
    Z2 = Z_tmp[:, :, 1]
    # for Mu_A in Mu_A_List:

    plt.figure(figsize=[5,10])
    # plt.subplot(2,1,1)
    ax = plt.axes((0.1, 0.55, 0.8, 0.4),projection="3d")
    # ax.plot_wireframe(X2,X3,Z1,cmap='viridis')
    X2, X3 = np.meshgrid(x2, x3, indexing="ij")
    ax.plot_surface(X3, X2, Z1, cmap='viridis')
    ax.set_xlabel('x3(Movement)')
    ax.set_ylabel('x2(Battery)')
    ax.set_zlabel('Mu')
    ax.set_zlim([10,0])
    ax.view_init(elev=-140, azim=35)
    ax.set_title('(a)', y=-0.13)
    # plt.savefig(f"x1={x1[0]}.png")
    # plt.show()

    # plt.figure(figsize=[5, 5])
    ax = plt.axes((0.1, 0.05, 0.8, 0.4), projection="3d")
    ax.plot_surface(X3, X2, Z2, cmap='viridis')
    ax.set_xlabel('x3(Movement)')
    ax.set_ylabel('x2(Battery)')
    ax.set_zlabel('Mu')
    ax.set_zlim([10, 0])
    ax.view_init(elev=-140, azim=35)
    ax.set_title('(b)', y=-0.13)
    plt.savefig(genPath(f"upper:x1={x1[0]},bottem:x1={x1[1]}.png"),transparent=True)
    plt.show()

    Sim_Data = np.load(genPath("SimData.npy"))
    Score = np.load(genPath("Score.npy"))
    Distance = np.load(genPath("Distance.npy"))

    idx_Score = np.argmax(Score)
    idx_Battery = np.argmax(Sim_Data[:, 2])
    idx_close = np.argmin(Distance)
    plt.figure(figsize=[5, 5])
    plt.scatter(Sim_Data[:, 0], Sim_Data[:, 1], marker=".")
    plt.scatter(*Sim_Data[idx_Score][:2], marker="s", s=30)
    plt.scatter(*Sim_Data[idx_Battery][:2], marker="v", s=30)
    plt.scatter(*Sim_Data[idx_close][:2], marker="*", s=30)
    plt.savefig(genPath(f"Sim.png"))
    plt.show()
    # print(f"The Highest Score: id {np.argmax(Score)}, {Sim_Data[np.argmax(Score)]}")
    # print(f"The Highest Battery: id {np.argmax(Sim_Data[:, 2])}, {Sim_Data[np.argmax(Sim_Data[:, 2])]}")
    # print(f"The Nearest Point: id {np.argmin(Distance)}, {Sim_Data[np.argmin(Distance)]}")

    print(f"The Highest Score: (distance{Distance[idx_Score]}, battery capacity {Sim_Data[idx_Score,2]}, "
          f"mobility degree {Sim_Data[idx_Score,3]})")
    print(f"The Highest Battery: (distance{Distance[idx_Battery]}, battery capacity {Sim_Data[idx_Battery,2]}, "
          f"mobility degree {Sim_Data[idx_Battery,3]})")
    print(f"The Nearest Point: (distance{Distance[idx_close]}, battery capacity {Sim_Data[idx_close,2]}, "
          f"mobility degree {Sim_Data[idx_close,3]})")

    print("The Highest Score: (distance: {:.4f}, battery capacity: {:.4f}, mobility degree: {:.4f})."
          .format(Distance[idx_Score], Sim_Data[idx_Score, 2], Sim_Data[idx_Score, 3]))
    print("The Highest Battery: (distance: {:.4f}, battery capacity: {:.4f}, mobility degree: {:.4f})."
          .format(Distance[idx_Battery], Sim_Data[idx_Battery, 2], Sim_Data[idx_Battery, 3]))
    print("The Nearest Point: (distance: {:.4f}, battery capacity: {:.4f}, mobility degree: {:.4f})."
          .format(Distance[idx_close], Sim_Data[idx_close, 2], Sim_Data[idx_close, 3]))

    print("最高分数: (距质心距离: {:.4f}, 剩余电量: {:.4f}, 运动能力: {:.4f})."
          .format(Distance[idx_Score], Sim_Data[idx_Score, 2], Sim_Data[idx_Score, 3]))
    print("最多电池容量: (距质心距离: {:.4f}, 剩余电量: {:.4f}, 运动能力: {:.4f})."
          .format(Distance[idx_Battery], Sim_Data[idx_Battery, 2], Sim_Data[idx_Battery, 3]))
    print("距质心最近: (距质心距离: {:.4f}, 剩余电量: {:.4f}, 运动能力: {:.4f})."
          .format(Distance[idx_close], Sim_Data[idx_close, 2], Sim_Data[idx_close, 3]))
    # print(f"The Highest Battery: id {np.argmax(Sim_Data[:, 2])}, {Sim_Data[np.argmax(Sim_Data[:, 2])]}")
    # print(f"The Nearest Point: id {np.argmin(Distance)}, {Sim_Data[np.argmin(Distance)]}")







