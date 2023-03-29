import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure("vehicular density1")
ax = plt.gca()
# #去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
#移位置 设为原点相交
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim(0,12.1)
plt.ylim(0,1.8e4)
y=np.arange(0, 14, 2)
x = [0, 0.4e4, 0.8e4, 1.2e4, 1.6e4, 2e4, 2.4e4]
plt.yticks(x, labels=["0", "0.5e4", "1e4", "1.5e4", "2e4", "2.5e4", "3e4"])
plt.grid()


plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(y,[0,0.24e4,0.3e4,0.75e4,0.85e4,0.93e4,1e4], 'o--r', linewidth = 2)
plt.plot(y,[0,0.27e4,0.36e4,1.2e4,1.36e4,1.48e4,1.58e4], '*--b', linewidth = 2)
plt.plot(y,[0,0.33e4,0.45e4,1.32e4,1.42e4,1.55e4,1.64e4], '^--g', linewidth = 2)
plt.xlabel('车辆密度 (车辆数/$Km^2$)',fontdict={"size": 14})
plt.ylabel('系统平均信息年龄',fontdict={"size": 14})
plt.legend(['MS3DQN算法','3DQN算法','DDQN算法'],loc='lower right',prop={ "size": 14})
plt.savefig(fname='vehicle density1.jpg',path='D:\PycharmProjects\SNARM-UAV-Learning\TEST_plot2',dpi=600)
plt.show()
