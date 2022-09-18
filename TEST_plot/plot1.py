import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure("vehicular density")
ax = plt.gca()
# #去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
#移位置 设为原点相交
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.grid()
plt.plot(np.arange(0, 14, 2),[0,0.24e4,0.3e4,0.75e4,0.85e4,0.93e4,1e4], 'o--r', linewidth = 2)
plt.plot(np.arange(0, 14, 2),[0,0.27e4,0.36e4,1.2e4,1.36e4,1.48e4,1.58e4], '*--b', linewidth = 2)
plt.plot(np.arange(0, 14, 2),[0,0.33e4,0.45e4,1.32e4,1.42e4,1.55e4,1.64e4], '^--k', linewidth = 2)
plt.xlabel('Vehicular Density in(Veh/Km)')
plt.ylabel('Expected Sum AoI')
plt.legend(['Dueling_DDQN_MultiStepLearning', 'Dueling_DDQN', 'DDQN'])
plt.show()
