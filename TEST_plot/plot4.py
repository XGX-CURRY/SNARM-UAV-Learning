import numpy as np


import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from collections import Counter


fig, ax = plt.subplots()

ax.set_title("Right Ear")


x = [100,200,300,400,500,600,700,800,900,1000,1100]

ax.set_ylabel("db HL")
ax.set_xlabel("Frequency")
plt.axis([0,9000,130,-10])
ax.set_facecolor("#ffd2d2")
ax.xaxis.set_ticks(x)
ax.xaxis.set_ticklabels(["","","500","","1K","","2K","","4K","","8K"])

ax.yaxis.set_ticks([120,110,100,90,80,70,60,50,40,30,20,10,0,-10])

ax.plot()
plt.grid(color="grey")
plt.show()