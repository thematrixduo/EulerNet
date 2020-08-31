# first version coded by Duo Wang (wd263@cam.ac.uk) 2018
# slightly modified by Tiansi Dong (dongt@bit.uni-bonn.de) 2020

import matplotlib.pyplot as plt
import numpy as np


ratio = 0.9

def gen_B_in_A(r_lb,r_ub):
    A_radius=np.random.uniform(r_lb,r_ub)
    B_radius=np.random.uniform(r_lb/4,A_radius*ratio)
    A_center=(0.5,0.5)
    B_center=(0.5+np.random.uniform((B_radius-A_radius)*ratio,(A_radius-B_radius)*ratio),
              0.5+np.random.uniform((B_radius-A_radius)*ratio,(A_radius-B_radius)*ratio))
    return A_center,B_center,A_radius,B_radius


def gen_A_in_B(r_lb,r_ub):
    B_radius=np.random.uniform(r_lb,r_ub)
    A_radius=np.random.uniform(r_lb/2,B_radius*0.8)
    B_center=(0.5,0.5)
    A_center=(0.5+np.random.uniform((A_radius-B_radius)*ratio,(B_radius-A_radius)*ratio),
              0.5+np.random.uniform((A_radius-B_radius)*ratio,(B_radius-A_radius)*ratio))
    return A_center,B_center,A_radius,B_radius


def gen_join(r_lb,r_ub):
    B_radius=np.random.uniform(r_lb,r_ub)
    A_radius=np.random.uniform(r_lb,r_ub)
    B_center=(0.5-np.random.uniform(B_radius*0.2,B_radius*0.8), 0.5)
    # A_center=(0.5+np.random.uniform(A_radius*0.2,A_radius*0.8), 0.5)
    A_center = (B_center[0] + B_radius + np.random.uniform(- A_radius * ratio, A_radius * ratio), 0.5)

    return A_center,B_center,A_radius,B_radius


def gen_disjoin(r_lb,r_ub):
    B_radius=np.random.uniform(r_lb,r_ub)
    A_radius=np.random.uniform(r_lb,r_ub)
    B_center=(0.5-np.random.uniform(B_radius*1.1,B_radius*1.3),0.5)
    # A_center=(0.5+np.random.uniform(A_radius*1.1,A_radius*1.3),0.5)
    A_center = (B_center[0] + B_radius + np.random.uniform(A_radius * 1.1, A_radius * 2.0), 0.5)

    return A_center,B_center,A_radius,B_radius 


def gen_circle(op,filename,color_1,color_2):
    op_num=0
    if op=='>':
        A_center,B_center,A_radius,B_radius=gen_B_in_A(0.2,0.3)
        op_num=1
    elif op=='<':
        A_center,B_center,A_radius,B_radius=gen_A_in_B(0.2,0.3)
        op_num=2
    elif op=='&':
        A_center,B_center,A_radius,B_radius=gen_join(0.2,0.26)
        op_num=3
    elif op=='!':
        A_center,B_center,A_radius,B_radius=gen_disjoin(0.15,0.2)
        op_num=4

    plot_circle(A_center,B_center,A_radius,B_radius,color_1,color_2,filename)
    return op_num
    #return A_center,B_center,A_radius,B_radius


def plot_circle(A_center,B_center,A_radius,B_radius,A_color,B_color,filename):

    circle1 = plt.Circle(A_center, A_radius, color=A_color,fill=False,linewidth=4,linestyle='dashed')
    circle2 = plt.Circle(B_center, B_radius, color=B_color,fill=False,linewidth=4,linestyle='dashed')

    fig, ax = plt.subplots(figsize=(8,8)) # note we must use plt.subplots, not plt.subplot

    ax.add_artist(circle1)
    ax.add_artist(circle2)
 
    ax.axis('off')

    fig.savefig(filename, dpi=8)
    plt.close()

# A_center,B_center,A_radius,B_radius=gen_B_in_A(0.2,0.3)
# A_center,B_center,A_radius,B_radius=gen_disjoin(0.15,0.2)

# plot_circle(A_center,B_center,A_radius,B_radius,'r','b','plotcircles.png')
