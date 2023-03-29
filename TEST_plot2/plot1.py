import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure("vehicular density2")
ax = plt.gca()
# #去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
#移位置 设为原点相交
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
# x = [0, 1, 2, 3, 4]
y = [0,0.5e4, 1e4, 1.5e4, 2e4,2.5e4,3e4,3.5e4]
plt.ylim(0,3e4)  #设置坐标显示范围
plt.xlim(0,12.1)
# plt.xticks(x, labels=["2", "3", "4", "5", "6"])
plt.yticks(y, labels=["0","0.5e4", "1e4", "1.5e4", "2e4","2.5e4","3e4","3.5e4"])
plt.grid()
plt.plot(np.arange(0, 15, 3),[0,0.5e4,0.65e4,1.85e4,2e4], 'o--r', linewidth = 2)
plt.plot(np.arange(0, 15, 3),[0,0.64e4,0.96e4,2.64e4,3e4], '*--b', linewidth = 2)
plt.plot(np.arange(0, 15, 3),[0,0.73e4,1.15e4,3.02e4,3.24e4], '^--c', linewidth = 2)
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('车辆密度(车辆数/$Km^2$)',fontdict={"size": 14})
plt.ylabel('系统平均信息年龄',fontdict={"size": 14})
plt.legend(['MM3DQN算法', 'M3DQN算法', 'MDDQN算法'],loc='lower right',prop={ "size": 14})
plt.savefig(fname='vehicle density2.jpg',path='D:\PycharmProjects\SNARM-UAV-Learning\TEST_plot2',dpi=600)
plt.show()
