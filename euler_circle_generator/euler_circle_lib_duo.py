import matplotlib.pyplot as plt
import numpy as np

def gen_B_in_A(r_lb,r_ub):
    A_radius=np.random.uniform(r_lb,r_ub)
    B_radius=np.random.uniform(r_lb/2,A_radius*0.8)
    A_center=(0.5,0.5)
    B_center=(0.5+np.random.uniform((B_radius-A_radius)*0.5,(A_radius-B_radius)*0.5),
              0.5+np.random.uniform((B_radius-A_radius)*0.5,(A_radius-B_radius)*0.5))
    return A_center,B_center,A_radius,B_radius  

def gen_A_in_B(r_lb,r_ub):
    B_radius=np.random.uniform(r_lb,r_ub)
    A_radius=np.random.uniform(r_lb/2,B_radius*0.8)
    B_center=(0.5,0.5)
    A_center=(0.5+np.random.uniform((A_radius-B_radius)*0.5,(B_radius-A_radius)*0.5),
              0.5+np.random.uniform((A_radius-B_radius)*0.5,(B_radius-A_radius)*0.5))
    return A_center,B_center,A_radius,B_radius  

def gen_join(r_lb,r_ub):
    B_radius=np.random.uniform(r_lb,r_ub)
    A_radius=np.random.uniform(r_lb,r_ub)
    B_center=(0.5-np.random.uniform(B_radius*0.2,B_radius*0.8),0.5)
    A_center=(0.5+np.random.uniform(A_radius*0.2,A_radius*0.8),0.5)

    return A_center,B_center,A_radius,B_radius 

def gen_disjoin(r_lb,r_ub):
    B_radius=np.random.uniform(r_lb,r_ub)
    A_radius=np.random.uniform(r_lb,r_ub)
    B_center=(0.5-np.random.uniform(B_radius*1.1,B_radius*1.3),0.5)
    A_center=(0.5+np.random.uniform(A_radius*1.1,A_radius*1.3),0.5)

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

    circle1 = plt.Circle(A_center, A_radius, color=A_color,fill=False,linewidth=0.7)
    circle2 = plt.Circle(B_center, B_radius, color=B_color,fill=False,linewidth=0.7)


    fig, ax = plt.subplots(figsize=(0.64,0.64)) # note we must use plt.subplots, not plt.subplot


    ax.add_artist(circle1)
    ax.add_artist(circle2)
 
    ax.axis('off')

    fig.savefig(filename)
    plt.close()

#A_center,B_center,A_radius,B_radius=gen_B_in_A(0.2,0.3)
A_center,B_center,A_radius,B_radius=gen_disjoin(0.15,0.2)

plot_circle(A_center,B_center,A_radius,B_radius,'r','b','plotcircles.png')
