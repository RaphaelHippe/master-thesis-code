import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./../datasets/autoencoded-p100/p100_100c_150_not_scaled_kullback_leibler_divergence_tanh.csv')

fig = plt.figure()

dy_y_1 = df[df.y == 1]
df_y_0 = df[df.y == 0]

plt.scatter(dy_y_1.a1, dy_y_1.a2, color='b', s=2, label='class 1')
plt.scatter(df_y_0.a1, df_y_0.a2, color='r', s=2, label='class 0')

plt.xlabel('a1')
plt.ylabel('a2')

plt.legend(loc=1)

# plt.show()
plt.savefig("./images/p100-scatter-not-scaled-kullback-leibler-divergence-tanh.svg")
