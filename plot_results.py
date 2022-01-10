# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:06:58 2020

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

result_no_mapping = np.load('Dueling_DDQN_MultiStepLeaning_main_Results.npz')
return_mov_avg_no_mapping = result_no_mapping['arr_0']
# print(result_no_mapping.files)
print(return_mov_avg_no_mapping)

result_SNARM = np.load('SNARM_main_Results.npz')
return_mov_avg_SNARM = result_SNARM['arr_0']

print(return_mov_avg_SNARM)

fig = plt.figure(50)
plt.plot(np.arange(len(return_mov_avg_no_mapping)), return_mov_avg_no_mapping, 'r-',
         linewidth=1)
plt.plot(np.arange(len(return_mov_avg_SNARM)), return_mov_avg_SNARM, 'b-', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Moving average of return per episode')
plt.legend(['Dueling_DDQN_MultiStepLeaning'])
plt.show()
