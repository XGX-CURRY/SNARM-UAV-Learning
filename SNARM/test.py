# import tensorflow as tf
# import timeit
#
# with tf.device('/cpu:0'):
#     cpu_a = tf.random.normal([10000, 1000])
#     cpu_b = tf.random.normal([1000, 2000])
#     print(cpu_a.device, cpu_b.device)
#
# with tf.device('/gpu:0'):
#     gpu_a = tf.random.normal([10000, 1000])
#     gpu_b = tf.random.normal([1000, 2000])
#     print(gpu_a.device, gpu_b.device)
#
# def cpu_run():
#     with tf.device('/cpu:0'):
#         c = tf.matmul(cpu_a, cpu_b)
#     return c
#
# def gpu_run():
#     with tf.device('/gpu:0'):
#         c = tf.matmul(gpu_a, gpu_b)
#     return c
#
# # warm up
# cpu_time = timeit.timeit(cpu_run, number=10)
# gpu_time = timeit.timeit(gpu_run, number=10)
# print('warmup:', cpu_time, gpu_time)
#
# cpu_time = timeit.timeit(cpu_run, number=10)
# gpu_time = timeit.timeit(gpu_run, number=10)
# print('run time:', cpu_time, gpu_time)
# import numpy as np
# import matplotlib.pyplot as plt
#
# a=[1,0,3,4,5,6,7,8]
# np.savez('xx.npz',a)
# result = np.load('xx.npz')
# print(result['arr_0'])

import numpy as np
import matplotlib.pyplot as plt
X_MAX=2000
Y_MAX=2000
def random_generate_vehicle_position(num_states):
    loc_x = np.random.uniform(50, X_MAX - 50, (num_states, 1))/1000  #均匀分布，左闭右开
    loc_y = np.random.uniform(50, Y_MAX - 50, (num_states, 1))/1000
    loc_z = np.zeros((num_states, 1))
    #对应列拼接，axis=0，则是对应行拼接,默认为0
    loc = np.concatenate((loc_x, loc_y,loc_z), axis=1)
    return loc

vehicle_loc = random_generate_vehicle_position(12)
print(vehicle_loc)
print(len(vehicle_loc))

plt.figure()
plt.plot(vehicle_loc[:, 0], vehicle_loc[:, 1], 'bp', markersize=5)
plt.show()