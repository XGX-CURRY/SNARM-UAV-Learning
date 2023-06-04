import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure("UAV Energy Consumption")
ax = plt.gca()
# # # #去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# # # 移位置 设为原点相交
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
y = [0,0.5e3, 1e3, 1.5e3, 2e3, 2.5e3, 3e3, 3.5e3]
plt.ylim(0,4e3)  #设置坐标显示范围
# plt.xlim(0,12.1)
plt.xticks(x, labels=["0", "20", "40", "60", "80", "100", "120", "140"])
plt.yticks(y, labels=["0","0.5e4", "1e3=4", "1.5e3=4", "2e3=4", "2.5e3=4", "3e4", "3.5e4"])
plt.grid()
plt.plot(x, [3.15e3, 2.9e3, 1.95e3, 1.25e3, 1.25e3, 1.25e3, 1.25e3, 1.25e3], 'o-r', linewidth=2)
plt.plot(x, [3.25e3, 3.15e3, 2.9e3, 2.32e3, 1.96e3, 1.75e3, 1.75e3, 1.75e3], '*--b', linewidth=2)
plt.plot(x, [3.5e3, 3.42e3, 3.20e3, 2.85e3, 2.35e3, 2.1e3, 2e3, 2e3], '^--g', linewidth=2)
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('无人机能耗 (kJ)',fontdict={"size": 14})
plt.ylabel('系统平均信息年龄',fontdict={"size": 14})
plt.legend(['MS3DQN算法','3DQN算法','DDQN算法'],loc='best',prop={ "size": 14})
plt.savefig(fname='uav energy.jpg',path='D:\PycharmProjects\SNARM-UAV-Learning\TEST_plot2',dpi=600)
plt.show()

# maxEnergy = 43.42Wh = 155.952kJ
