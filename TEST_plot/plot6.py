import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,10,2)
y=x*x
la= ["f","s","t","j"]
fig=plt.figure(figsize=(10,8))
plt.plot(x,y)
plt.tick_params(pad=15)
plt.xlabel('')        #去掉x轴标签
plt.xticks(ticks=[0,1,2,3],        #设置要显示的x轴刻度，若指定空列表则去掉x轴刻度
# ,
labels=la,#设置x轴刻度显示的文字，要与ticks对应
fontsize=15,        #设置刻度字体大小
rotation=0,        #设置刻度文字旋转角度
ha='center', va='center',        #刻度文字对齐方式，当rotation_mode为’anchor'时，对齐方式决定了文字旋转的中心。ha也可以写成horizontalalignment，va也可以写成verticalalignment。
)
plt.show()