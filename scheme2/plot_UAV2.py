import numpy as np


import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from collections import Counter

episode=2000
npzfile = np.load('radioenvir.npz')
OutageMapActual = npzfile['arr_0']
X_vec = npzfile['arr_1']
Y_vec = npzfile['arr_2']
result_DDQN_snr2 = np.load('Dueling_DDQN_MultiStepLeaning_main_Results_1.npz', allow_pickle=True)
return_mov_avg_DDQN_snr2 = result_DDQN_snr2['arr_0']
reward_mov_avg_DDQN_snr2 = result_DDQN_snr2['arr_1']
trajecotry_mov_avg_DDQN_snr2 = result_DDQN_snr2['arr_2']
DESTINATION = np.array([[1400, 1600]], dtype="float32")

fig = plt.figure('方案二无人机轨迹2')
#绘制二维颜色填充图
plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual)
v = np.linspace(0, 1.0, 11, endpoint=True)
cbar = plt.colorbar(ticks=v)
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
# v = np.linspace(0, 1.0, 11, endpoint=True)
# cbar=plt.colorbar(ticks=v)
# cbar.ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1.0'])
cbar.set_label('覆盖概率', labelpad=20, rotation=270, fontsize=14)

for episode_idx in range(episode - 200, episode):
    S_seq = trajecotry_mov_avg_DDQN_snr2[episode_idx]
    S_seq = np.squeeze(np.asarray(S_seq))

    if S_seq.ndim > 1:
        plt.plot(S_seq[0, 0], S_seq[0, 1], 'rx', markersize=5)
        plt.plot(S_seq[:, 0], S_seq[:, 1], '-')
    else:
        plt.plot(S_seq[0], S_seq[1], 'rx', markersize=5)
        plt.plot(S_seq[0], S_seq[1], '-')

plt.plot(DESTINATION[0, 0], DESTINATION[0, 1], 'b^', markersize=25)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.savefig(fname='scheme2uav2.jpg',path='D:\PycharmProjects\SNARM-UAV-Learning\TEST_plot2',dpi=600)
plt.show()