import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/plates_10.csv')
fig = plt.figure()

df.diameter[df.goodPlate == 1].plot(kind='density', label='class 1', color='b')
df.diameter[df.goodPlate == 0].plot(kind='density', label='class 0', color='r')

plt.xlabel('diameter')

plt.legend(loc=2)

# plt.show()
plt.savefig("./images/p10-conditional-density-diameter-class.svg")
