import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/plates_10.csv')
fig = plt.figure()

df_goodPlate = df[df.goodPlate == 1]
df_badPlate = df[df.goodPlate == 0]

x_g = df_goodPlate.diameter[df_goodPlate.color == 'orange']
y_g = [1 for i in x_g]
x_b = df_badPlate.diameter[df_badPlate.color == 'orange']
y_b = [1 for i in x_b]

plt.scatter(x_g, y_g, color='g', label='good')
plt.scatter(x_b, y_b, color='r', label='bad')

plt.xlabel('diameter')
plt.ylabel('color orange')

plt.legend(loc=1)

# plt.show()
plt.savefig("./images/p10-conditional-scatter-diameter-color-class.svg")
