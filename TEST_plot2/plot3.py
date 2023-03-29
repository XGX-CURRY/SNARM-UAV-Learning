import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure("IRS")
ax = plt.gca()
# # # #去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# # # 移位置 设为原点相交
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
x = [0, 1, 2, 3, 4,5,6,7,8]
y = [0,0.5e4, 1e4, 1.5e4, 2e4,2.5e4,3e4,3.5e4,4e4]
plt.ylim(0,4e3)  #设置坐标显示范围
plt.xticks(x, labels=["10", "15", "20", "25", "30","35","40","45","50"])
plt.yticks(y, labels=["0.8e4","1.2e4", "1.6e4", "2e4", "2.4e4","2.8e4","3.2e4","3.6e4","4e4"])
plt.grid()
plt.plot([0, 2, 4,6,8], [3.15e4, 2.52e4, 1.60e4, 1.22e4, 1.07e4], 'o--r', linewidth=2)
plt.plot([0, 2, 4,6,8], [3.39e4, 2.85e4, 1.98e4, 1.41e4, 1.21e4], '*--b', linewidth=2)
plt.plot([0, 2, 4,6,8], [3.64e4, 3.29e4, 2.34e4, 1.87e4, 1.53e4], '^--c', linewidth=2)
plt.rcParams['font.sans-serif'] = ['STSong']
# plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('IRS单元数',fontdict={"size": 14})
plt.ylabel('系统平均信息年龄',fontdict={"size": 14})
plt.legend(['MM3DQN算法', 'M3DQN算法', 'MDDQN算法'],loc='best',prop={ "size": 14})
# plt.show()
plt.savefig(fname='IRS.jpg',path='D:\PycharmProjects\SNARM-UAV-Learning\TEST_plot2',dpi=600)

# maxEnergy = 43.42Wh = 155.952kJ
