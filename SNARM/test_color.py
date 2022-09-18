import numpy as np
import matplotlib.pyplot as plt

npzfile = np.load('radioenvir.npz')
OutageMapActual = npzfile['arr_0']
X_vec = npzfile['arr_1']
Y_vec = npzfile['arr_2']
DESTINATION = np.array([[1400, 1600]], dtype="float32")

# print(OutageMapActual)
# print(X_vec)
# print(Y_vec)


fig = plt.figure(30)  # 绘制二维颜色填充图
plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual)
v = np.linspace(0, 1.0, 11, endpoint=True)
cbar = plt.colorbar(ticks=v)
cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
plt.plot(DESTINATION[0, 0], DESTINATION[0, 1], 'b^', markersize=25)
plt.xlabel('x (meter)', fontsize=14)
plt.ylabel('y (meter)', fontsize=14)
# plt.show()
