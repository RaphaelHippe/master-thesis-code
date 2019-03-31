import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/plates_10.csv')
fig = plt.figure()

df_goodPlate = df[df.goodPlate == 1]
df_badPlate = df[df.goodPlate == 0]
x_g = df_goodPlate.diameter[df_goodPlate.color == 'orange']
x_b = df_badPlate.diameter[df_badPlate.color == 'orange']
x_g.plot(kind='density', label='class 1', color='b')
x_b.plot(kind='density', label='class 0', color='r')

plt.xlabel('diameter')

plt.legend(loc=2)

# plt.show()
plt.savefig("./images/p10-conditional-density-diameter-color-class.svg")
