import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('./../datasets/damage_done.csv')
df = pd.read_csv('./../datasets/plates_10.csv')
fig = plt.figure()

df['class'] = df['goodPlate']

ax = df['class'].hist()

plt.ylabel('occurrence frequency')

# plt.show()
plt.savefig("./images/p10-class-hist.svg")
