import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df_ct_wins = df[df.ct_wins == 1]
df_t_wins = df[df.ct_wins == 0]

plt.scatter(df_ct_wins.attackerHealth, df_ct_wins.damageHealth, color='g', s=1.5)
plt.scatter(df_t_wins.attackerHealth, df_t_wins.damageHealth, color='r', s=1.5)

plt.xlabel('health of attacker')
plt.ylabel('damage to health')

plt.legend(loc=1)

# plt.show()
plt.savefig("./images/dd-conditional-scatter-attackerHealth-damageHealth-class.svg")
