import pandas as pd
import matplotlib.pyplot as plt

df_y = pd.read_csv('./../datasets/plates_100.csv')
df = pd.read_csv('./../datasets/dummy_pca/p100/p100_n2.csv')
df['y'] = df_y['goodPlate']
fig = plt.figure()

dy_y_1 = df[df.y == 1]
df_y_0 = df[df.y == 0]

plt.scatter(dy_y_1.pc0, dy_y_1.pc1, color='b', s=2, label='class 1')
plt.scatter(df_y_0.pc0, df_y_0.pc1, color='r', s=2, label='class 0')

plt.xlabel('pc0')
plt.ylabel('pc1')

plt.legend(loc=1)

# plt.show()
plt.savefig("./images/p100-scatter-pca-dummy-2.svg")
