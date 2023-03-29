
import numpy as np
import matplotlib.pyplot as plt

# x=np.arange(2000)
# y=np.arange(0,4000,2)
# fig = plt.figure(100)
# # plt.plot(np.arange(len(return_mov_avg_no_mapping)), return_mov_avg_no_mapping, 'r-',linewidth=0.5)
# plt.plot(y, x, 'c-',linewidth=0.5)
# # print(len(x))
# # plt.plot(y, y, 'r-',linewidth=0.5)
# # plt.xticks(x, y)
# plt.show()

# fig1 = plt.figure(200)
# plt.plot(y, y, 'r-',linewidth=0.5)
# print(len(y))
# plt.show()



# # 生成示例数据
# x = np.arange(0, 10, 0.1)
# y = np.sin(x)
#
# # 绘制图形
# plt.plot(x, y)
#
# # 设置 y 坐标轴上的刻度线位置
# y_ticks = np.arange(-1, 1.1, 0.5)
# plt.yticks(y_ticks * 5, y_ticks)
#
# # 显示图形
# plt.show()



# 生成示例数据
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# 绘制图形
plt.plot(x, y)

# 添加注释
x0 = 2
y0 = np.sin(x0)
y_delta = 0.3
plt.annotate('',
             xy=(x0, y0 + y_delta), xycoords='data',
             xytext=(x0, y0), textcoords='data',
             arrowprops={'arrowstyle': '<->'})
plt.annotate('{:.1f}'.format(y_delta),
             xy=(x0, y0 + y_delta/2), xycoords='data',
             xytext=(-20, 20), textcoords='offset points')

# 显示图形
plt.show()
