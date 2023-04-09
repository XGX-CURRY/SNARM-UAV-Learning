# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:06:58 2020

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

result_no_mapping = np.load('Dueling_DDQN_main_Results2.npz',allow_pickle= True)
return_mov_avg_no_mapping = result_no_mapping['arr_0']
reward_mov_avg_no_mapping = result_no_mapping['arr_1']
trajecotry_mov_avg_no_mapping = result_no_mapping['arr_2']

result_no_mapping4 = np.load('Dueling_DDQN_MultiStepLeaning_main_Results_1.npz',allow_pickle= True)
return_mov_avg_no_mapping4 = result_no_mapping4['arr_0']
reward_mov_avg_no_mapping4 = result_no_mapping4['arr_1']
trajecotry_mov_avg_no_mapping4 = result_no_mapping4['arr_2']


result_no_mapping3 = np.load('DDQN_main_Results3.npz',allow_pickle= True)
return_mov_avg_no_mapping3 = result_no_mapping3['arr_0']
return_mov_avg_no_mapping3_1750=return_mov_avg_no_mapping3[0:1801]
reward_mov_avg_no_mapping3 = result_no_mapping3['arr_1']
trajecotry_mov_avg_no_mapping3 = result_no_mapping3['arr_2']

# print(result_no_mapping.files)
# print(trajecotry_mov_avg_no_mapping)
# print(return_mov_avg_no_mapping)

result_DDQN = np.load('Dueling_DDQN_MultiStepLeaning_main_Results_100.npz', allow_pickle=True)
return_mov_avg_DDQN = result_DDQN['arr_0']
reward_mov_avg_DDQN = result_DDQN['arr_1']
trajecotry_mov_avg_DDQN = result_DDQN['arr_2']
# print(return_mov_avg_DDQN)
# print(result_DDQN.files)
# print(trajecotry_mov_avg_DDQN)
result_SNARM = np.load('SNARM_main_Results1.npz', allow_pickle=True)
return_mov_avg_SNARM = result_SNARM['arr_0']
reward_mov_avg_SNARM = result_SNARM['arr_1']
trajecotry_mov_avg_SNARM = result_SNARM['arr_2']

result_SNARM_snr2 = np.load('return1.npz', allow_pickle=True)
return_mov_avg_SNARM_snr2 = result_SNARM_snr2['arr_0']
# reward_mov_avg_DDQN_snr2 = result_DDQN_snr2['arr_1']
# trajecotry_mov_avg_DDQN_snr2 = result_DDQN_snr2['arr_2']

result_DDQN_snr1 = np.load('Dueling_DDQN_main_Results_snr1.npz', allow_pickle=True)
return_mov_avg_DDQN_snr1 = result_DDQN_snr1['arr_0']
reward_mov_avg_DDQN_snr1 = result_DDQN_snr1['arr_1']
trajecotry_mov_avg_DDQN_snr1 = result_DDQN_snr1['arr_2']

result_DDQN_snr2 = np.load('Dueling_DDQN_main_Results_snr2.npz', allow_pickle=True)
return_mov_avg_DDQN_snr2 = result_DDQN_snr2['arr_0']
reward_mov_avg_DDQN_snr2 = result_DDQN_snr2['arr_1']
trajecotry_mov_avg_DDQN_snr2 = result_DDQN_snr2['arr_2']

result_DDQN_5000_1 = np.load('Dueling_DDQN_MultiStepLeaning_main_Results_5000_1.npz', allow_pickle=True)
return_mov_avg_5000_1 = result_DDQN_5000_1['arr_0']
reward_mov_avg_5000_1 = result_DDQN_5000_1['arr_1']
trajecotry_mov_avg_DDQN_5000_1= result_DDQN_5000_1['arr_2']

result_DDQN_5000_2= np.load('Dueling_DDQN_MultiStepLeaning_main_Results_5000_2.npz', allow_pickle=True)
return_mov_avg_5000_2 = result_DDQN_5000_2['arr_0']
reward_mov_avg_5000_2 = result_DDQN_5000_2['arr_1']
trajecotry_mov_avg_DDQN_5000_2= result_DDQN_5000_2['arr_2']

result_DDQN_5000_3 = np.load('Dueling_DDQN_MultiStepLeaning_main_Results_5000_3.npz', allow_pickle=True)
return_mov_avg_5000_3 = result_DDQN_5000_3['arr_0']
reward_mov_avg_5000_3 = result_DDQN_5000_3['arr_1']
trajecotry_mov_avg_DDQN_5000_3= result_DDQN_5000_3['arr_2']

result_DDQN_5000_4= np.load('Dueling_DDQN_MultiStepLeaning_main_Results_5000_4.npz', allow_pickle=True)
return_mov_avg_5000_4 = result_DDQN_5000_4['arr_0']
reward_mov_avg_5000_4 = result_DDQN_5000_4['arr_1']
trajecotry_mov_avg_DDQN_5000_4= result_DDQN_5000_4['arr_2']
# print(result_no_mapping.files)
# print(trajecotry_mov_avg_no_mapping)
# print(return_mov_avg_no_mapping)
y=np.arange(0,5403,3)
return_mov_avg_no_mapping4[1667:1801]=return_mov_avg_no_mapping4[1667:1801]
# return_mov_avg_no_mapping3_1750[1667:1801]=return_mov_avg_no_mapping3_1750[1667:1801]*1
# return_mov_avg_SNARM[0:300]=return_mov_avg_SNARM[0:300]*1.5
fig = plt.figure(100)
# plt.plot(np.arange(len(return_mov_avg_no_mapping)), return_mov_avg_no_mapping, 'r-',linewidth=0.5)
# plt.plot(y, return_mov_avg_SNARM, color='red',linestyle='-',linewidth=0.5)
# plt.plot(y, return_mov_avg_no_mapping4, color='green',linestyle='-',linewidth=0.5)
# plt.plot(y, return_mov_avg_no_mapping3_1750, color='blue',linestyle='-',linewidth=0.5)
# plt.plot(y, return_mov_avg_DDQN, 'k-', linewidth=0.5)
plt.plot(np.arange(len(return_mov_avg_5000_1)), return_mov_avg_5000_1, 'r-', linewidth=0.5)
plt.plot(np.arange(len(return_mov_avg_5000_2)), return_mov_avg_5000_2, 'b-', linewidth=0.5)
plt.plot(np.arange(len(return_mov_avg_5000_3)), return_mov_avg_5000_3, 'g-', linewidth=0.5)
plt.plot(np.arange(len(return_mov_avg_5000_4)), return_mov_avg_5000_4, 'm-', linewidth=0.5)
# plt.plot(y, return_mov_avg_DDQN_snr2, 'g-', linewidth=0.5)
# plt.plot(y, return_mov_avg_SNARM_snr2, 'y-', linewidth=0.5)
# print(len(return_mov_avg_SNARM))
print(len(return_mov_avg_5000_1))
# print(len(return_mov_avg_no_mapping3_1750))
plt.grid()
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
# x_ticks=[0,800,1600,2400,3200,4000,4800]
# x_ticks=np.arange(5000)
# plt.xticks(1750,x_ticks)
# my_y_ticks = np.arange(0, 1750)
# plt.xticks(my_y_ticks)
plt.xlim(0,5000)
plt.xlabel('训练回合数',fontdict={"size": 14})
plt.ylabel('累积奖励值',fontdict={"size": 14})
plt.legend(['MM3DQN算法','M3DQN算法','DDQN算法'],loc='best',prop={ "size": 14})
# plt.savefig(fname='scheme2reward.jpg',path='D:\PycharmProjects\SNARM-UAV-Learning\TEST_plot2',dpi=600)
plt.show()

# fig1 = plt.figure(100)
# # plt.plot(np.arange(len(return_mov_avg_no_mapping)), return_mov_avg_no_mapping, 'r-',linewidth=0.5)
# # plt.plot(np.arange(len(return_mov_avg_DDQN)), return_mov_avg_DDQN, 'b-', linewidth=0.5)
# plt.plot(np.arange(len(return_mov_avg_DDQN_snr)), return_mov_avg_DDQN_snr, 'k-', linewidth=0.05)
# plt.plot(np.arange(len(return_mov_avg_DDQN_snr2)), return_mov_avg_DDQN_snr2, 'y-', linewidth=0.05)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.legend(['Dueling_DDQN_snr','Dueling_DDQN_snr2'])
# plt.show()
#
# fig2 = plt.figure(100)
# plt.plot(np.arange(len(reward_mov_avg_no_mapping)), reward_mov_avg_no_mapping, 'r-',linewidth=0.25)
# plt.plot(np.arange(len(reward_mov_avg_DDQN)), reward_mov_avg_DDQN, 'b-', linewidth=0.25)
# plt.plot(np.arange(len(reward_mov_avg_no_mapping3)), reward_mov_avg_no_mapping3, 'g-',linewidth=0.25)
# plt.plot(np.arange(len(reward_mov_avg_DDQN_snr)), reward_mov_avg_DDQN_snr, 'k-', linewidth=0.005)
# plt.plot(np.arange(len(reward_mov_avg_DDQN_snr2)), reward_mov_avg_DDQN_snr2, 'm-', linewidth=0.005)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.legend(['Dueling_DDQN_MultiStepLeaning','Dueling_DDQN','Dueling_DDQN_snr','Dueling_DDQN_snr2'])
# plt.show()
#
# fig3 = plt.figure(100)
# plt.plot(np.arange(len(reward_mov_avg_no_mapping)), reward_mov_avg_no_mapping, 'r-',linewidth=0.5)
# plt.plot(np.arange(len(reward_mov_avg_DDQN)), reward_mov_avg_DDQN, 'b-', linewidth=0.5)
# plt.plot(np.arange(len(reward_mov_avg_DDQN_snr1)), reward_mov_avg_DDQN_snr1, 'k-', linewidth=0.5)
# plt.plot(np.arange(len(reward_mov_avg_DDQN_snr2)), reward_mov_avg_DDQN_snr2, 'y-', linewidth=0.5)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.legend(['Dueling_DDQN_snr','Dueling_DDQN_snr2'])
# plt.show()

# fig4 = plt.figure(100)
# plt.plot(np.arange(len(trajecotry_mov_avg_no_mapping)), trajecotry_mov_avg_no_mapping, 'r-',linewidth=0.5)
# plt.plot(np.arange(len(trajecotry_mov_avg_DDQN)), trajecotry_mov_avg_DDQN, 'b-', linewidth=0.5)
# plt.plot(np.arange(len(trajecotry_mov_avg_DDQN_snr1)), trajecotry_mov_avg_DDQN_snr1, 'k-', linewidth=0.5)
# plt.plot(np.arange(len(trajecotry_mov_avg_DDQN_snr2)), trajecotry_mov_avg_DDQN_snr2, 'y-', linewidth=0.5)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.legend(['Dueling_DDQN_snr','Dueling_DDQN_snr2'])
# plt.show()