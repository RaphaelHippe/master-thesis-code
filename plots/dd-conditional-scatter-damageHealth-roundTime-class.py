import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df_ct_wins = df[df.ct_wins == 1]
df_t_wins = df[df.ct_wins == 0]

plt.scatter(df_ct_wins.damageHealth, df_ct_wins.roundTime, color='g', s=1.5)
plt.scatter(df_t_wins.damageHealth, df_t_wins.roundTime, color='r', s=1.5)

plt.xlabel('damage to health')
plt.ylabel('round time')

plt.legend(loc=1)

# plt.show()
plt.savefig("./images/dd-conditional-scatter-damageHealth-roundTime-class.svg")
