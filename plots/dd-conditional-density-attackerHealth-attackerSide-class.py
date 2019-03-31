import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df.attackerHealth[
(df.ct_wins == 1)
& (df.attackerSide == 'Terrorist')
].plot(kind='density', label='class 1', color='b')
df.attackerHealth[
(df.ct_wins == 0)
& (df.attackerSide == 'Terrorist')
].plot(kind='density', label='class 0', color='r')

plt.xlabel('health of attacker')

plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-density-attackerHealth-attackerSide-class.svg")
