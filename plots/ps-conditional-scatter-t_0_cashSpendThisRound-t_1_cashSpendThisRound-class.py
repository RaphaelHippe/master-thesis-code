import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/player_stats.csv')
fig = plt.figure()

df_ct_wins = df[df.ct_wins == 1]
df_t_wins = df[df.ct_wins == 0]

plt.scatter(df_ct_wins.t_0_cashSpendThisRound, df_ct_wins.t_1_cashSpendThisRound, color='g', s=1.5)
plt.scatter(df_t_wins.t_0_cashSpendThisRound, df_t_wins.t_1_cashSpendThisRound, color='r', s=1.5)

plt.xlabel('Cash spend this round t0')
plt.ylabel('Cash spend this round t1')

plt.legend(loc=1)

# plt.show()
plt.savefig("./images/ps-conditional-scatter-t_0_cashSpendThisRound-t_1_cashSpendThisRound-class.svg")
