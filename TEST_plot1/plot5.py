#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

fig = plt.figure()
x = [1,3,5,7,9]
y=[1e-5,1e-4,1e-3,1e-2,1e-1]
plt.xticks(range(len(x)),x)

#把x轴的刻度间隔设置为24，即x的值，每24个在x轴显示一个
plt.plot(x,y)
x_major_locator=MultipleLocator(24)

#ax为两条坐标轴的实例
ax = plt.gca()

#把x轴的主刻度设置为1的倍数
ax.xaxis.set_major_locator(x_major_locator)
plt.show()
