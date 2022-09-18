import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure("outage probability")
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
y = [0, 2e4, 4e4, 6e4, 8e4, 10e4, 12e4]
plt.xticks(x, labels=["0", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "0.5"])
plt.yticks(y, labels=["0", "1e4", "2e4", "3e4", "4e4", "5e4", "6e4"])
plt.grid()
plt.plot(x, [0.3e4, 0.4e4, 0.7e4, 1.5e4, 2.7e4, 4.3e4, 7.5e4], 'o-r', linewidth=2, )
plt.plot(x, [0.36e4, 0.48e4, 0.85e4, 1.8e4, 3.5e4, 5.4e4, 8.9e4], '*--b', linewidth=2)
plt.plot(x, [0.45e4, 0.6e4, 1.1e4, 2.1e4, 4.6e4, 6.9e4, 10.6e4], '^--k', linewidth=2)
plt.xlabel('Outage Probability')
plt.ylabel('Expected Sum AoI')
plt.legend(['Dueling_DDQN_MultiStepLearning', 'Dueling_DDQN', 'DDQN'])
plt.show()
