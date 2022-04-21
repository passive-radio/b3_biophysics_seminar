import math
import time

from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class mcmc(object):
    def __init__(self, N, T, rho, monte_step):
        self.n = N
        self.T = T * 1.380649 * 10**-23
        self.rho = rho
        self.l = (N/rho)**(1/3)
        self.monte_step = monte_step
    
    def init_pos(self):
        n = self.n
        l = self.l
        rx_array = l * np.random.rand(n)
        ry_array = l * np.random.rand(n)
        rz_array = l * np.random.rand(n)
        
        return rx_array, ry_array, rz_array
        
        
    def metropolis(self):
        p_avg_list = []
        p_list = []
        sum = 0
        l = self.l
        dl = (1/self.rho)**(1/3)
        
        rx_array, ry_array, rz_array = self.init_pos()
        u = self.potential_tot(rx_array, ry_array, rz_array)
        
        for step in tqdm(range(self.monte_step)):
            for i in range(self.n):
                
                dx = dl/2 * (1.0 - 2.0 * np.random.rand())
                dy = dl/2 * (1.0 - 2.0 * np.random.rand())
                dz = dl/2 * (1.0 - 2.0 * np.random.rand())
                
                rx_array[i] += dx
                ry_array[i] += dy
                rz_array[i] += dz
                
                if rx_array[i] > l:
                    rx_array[i] -= l
                elif rx_array[i] < 0:
                    rx_array[i] += l
                if ry_array[i] > l:
                    ry_array[i] -= l
                elif ry_array[i] < 0:
                    ry_array[i] += l
                if rz_array[i] > l:
                    rz_array[i] -= l
                elif rz_array[i] < 0:
                    rz_array[i] += l
                
                u_new = self.potential_tot(rx_array, ry_array, rz_array)
                # print(u, u_new)

                if u < u_new and np.random.rand() > self.p(u, u_new):
                    rx_array[i] -= dx
                    ry_array[i] -= dy
                    rz_array[i] -= dz
                    
                    if rx_array[i] > l:
                        rx_array[i] -= l
                    elif rx_array[i] < 0:
                        rx_array[i] += l
                    if ry_array[i] > l:
                        ry_array[i] -= l
                    elif ry_array[i] < 0:
                        ry_array[i] += l
                    if rz_array[i] > l:
                        rz_array[i] -= l
                    elif rz_array[i] < 0:
                        rz_array[i] += l
                
                else:
                    u = u_new
            u = self.potential_tot(rx_array, ry_array, rz_array)
            p_list.append(u)
            sum += u
            
            # 移動平均のサイズ
            w_size = 100
            v = np.ones(w_size) / w_size # 長さがw_sizeで,値が1/w_sizeの配列
            p_ma_list = np.convolve(p_list, v, mode='valid')
            p_avg_list.append(sum/(step+1))
        
        return p_list, p_avg_list, p_ma_list
    
    
    def p(self, Uold, Unew):
        return math.exp(-1.0*(Unew-Uold)/self.T)
        
        
    def potential_tot(self, x_array, y_array, z_array):
        u = 0
        for i in range(self.n-1):
            for j in np.arange(start=i+1, stop=self.n, step=1):
                u += 4 * self.potential_ij(x_array[i],y_array[i], z_array[i],
                                    x_array[j], y_array[j], z_array[j])
        return u

    def potential_ij(self, r0,r1,r2,s0,s1,s2):
        l = self.l
        l_half = l/2
        
        rx = r0 - s0
        ry = r1 - s1
        rz  =r2 - s2
        
        if rx > l_half:
            rx -= l
        elif rx < -1*l_half:
            rx += l
        if ry > l_half:
            ry -= l
        elif ry < -1*l_half:
            ry += l
        if rz > l_half:
            rz -= l
        elif rz < -1*l_half:
            rz += l
            
        r2 = rx**2 + ry**2 + rz**2
        return (1/r2)**6 - (1/r2)**3

if __name__ == "__main__":
    mc = mcmc(N=16, T=10.0, rho=1, monte_step=21001)
    
    p_list, p_avg_list, p_ma_list = mc.metropolis()
    p_df = pd.DataFrame(p_list)
    avg_df = pd.DataFrame(p_avg_list)
    ma_df = pd.DataFrame(p_ma_list)
    # df = pd.concat([p_df, avg_df, ma_df], axis="columns")
    # df = df.set_axis(["potential", "cumulative_average", "moving_average(n=10)"], axis="columns")
    df = pd.concat([p_df, ma_df], axis="columns")
    df = df.set_axis(["potential", "moving_average(n=100)"], axis="columns")
    
    sns.histplot(df[10001:20000], bins=30)
    plt.show()
    
    fig = plt.figure()
    #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    df1 = df[1001:6000]
    sns.lineplot(data=df1, ax=ax1)
    # plt.xticks( np.arange(1001, 2000, 200))
    ax1.set_xlabel("monte step")
    ax1.set_ylabel("potential")
    ax1.legend(fontsize=10)
    
    df2 = df[6001:11000]
    sns.lineplot(data=df2, ax=ax2)
    # plt.xticks( np.arange(1001, 2000, 200))
    ax2.set_xlabel("monte step")
    ax2.set_ylabel("potential")

    df3 = df[11001:16000]
    sns.lineplot(data=df3, ax=ax3)
    # plt.xticks( np.arange(1001, 2000, 200))
    ax3.set_xlabel("monte step")
    ax3.set_ylabel("potential")
    
    df4 = df[16001:21000]
    sns.lineplot(data=df4, ax=ax4)
    ax4.set_xlabel("monte step")
    ax4.set_ylabel("potential")
    
    plt.show()