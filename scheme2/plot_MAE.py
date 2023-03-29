import numpy as np
import matplotlib.pyplot as plt

result_no_mapping = np.load('MAE1.npz',allow_pickle= True)
return_mov_avg_no_mapping = result_no_mapping['arr_0']

fig = plt.figure('Loss')
plt.plot(np.arange(len(return_mov_avg_no_mapping)), return_mov_avg_no_mapping, 'b-',linewidth=0.5)
plt.grid()
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('回合数')
plt.ylabel('平均绝对误差')
plt.legend(['MS3DQN算法'],loc='best')
plt.show()