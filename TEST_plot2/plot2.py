import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure("UAV数量")
ax = plt.gca()
# #去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# 移位置 设为原点相交
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
x = [0, 1, 2, 3, 4]
y = [0,0.25e4, 0.5e4, 0.75e4, 1e4,1.25e4,1.5e4,1.75e4,2e4]
plt.ylim(0,2e4)  #设置坐标显示范围
plt.xlim(0,4.1)
plt.xticks(x, labels=["2", "3", "4", "5", "6"])
plt.yticks(y, labels=["0.4e4","0.6e4", "0.8e4", "1e4", "1.2e4","1.4e4","1.6e4","1.8e4","2e4"])
plt.grid()
plt.plot(x, [1e4, 0.78e4, 0.68e4, 0.48e4,0.42e4], 'o--r', linewidth=2)
plt.plot(x, [1.26e4, 0.89e4, 0.78e4, 0.58e4,0.53e4], '*--b', linewidth=2)
plt.plot(x, [1.62e4, 1.24e4, 1.06e4, 0.82e4,0.78e4], '^--c', linewidth=2)
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('无人机数量',fontdict={"size": 14})
plt.ylabel('系统平均信息年龄',fontdict={"size": 14})
plt.legend(['MM3DQN算法', 'M3DQN算法', 'MDDQN算法'],loc='best',prop={ "size": 14})
plt.savefig(fname='scheme2uavs.jpg',path='D:\PycharmProjects\SNARM-UAV-Learning\TEST_plot2',dpi=600)
plt.show()
