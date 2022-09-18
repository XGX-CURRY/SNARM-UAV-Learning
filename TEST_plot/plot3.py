import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure("UAV Energy Consumption")
ax = plt.gca()
# #去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# 移位置 设为原点相交
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
x = [0, 1, 2, 3, 4, 5, 6]
y = [0, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3]
plt.xticks(x, labels=["0", "20", "40", "60", "80", "100", "120"])
plt.yticks(y, labels=["0", "1e3", "2e3", "3e3", "4e3", "5e3", "6e3"])
plt.grid()
plt.plot(x, [0, 0.08e4, 0.12e4, 0.2e4, 0.3e4, 0.3e4, 0.3e4], 'o-r', linewidth=2, )
plt.plot(x, [0, 0.16e4, 0.26e4, 0.42e4, 0.42e4, 0.42e4, 0.42e4], '*--b', linewidth=2)
plt.plot(x, [0, 0.2e4, 0.36e4, 0.5e4, 0.5e4, 0.5e4, 0.5e4], '^--k', linewidth=2)
plt.xlabel('UAV Energy Consumption(kJ)')
plt.ylabel('Expected Sum AoI')
plt.legend(['Dueling_DDQN_MultiStepLearning', 'Dueling_DDQN', 'DDQN'])
plt.show()


#maxEnergy = 43.42Wh = 155.952kJ