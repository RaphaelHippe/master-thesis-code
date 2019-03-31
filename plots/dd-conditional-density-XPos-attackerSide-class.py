import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df.attackerXPosition[(df.ct_wins == 1) & (df.attackerSide == 'Terrorist')].plot(kind='density', label='ct_win attacker', color='green')
df.attackerXPosition[(df.ct_wins == 0) & (df.attackerSide == 'Terrorist')].plot(kind='density', label='t_win attacker', color='red')

df.victimXPosition[(df.ct_wins == 1) & (df.attackerSide == 'Terrorist')].plot(kind='density', label='ct_win victim', color='lightgreen')
df.victimXPosition[(df.ct_wins == 0) & (df.attackerSide == 'Terrorist')].plot(kind='density', label='t_win victim', color='lightcoral')

plt.xlabel('player x position')


plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-density-XPosition-attackerSide-T-class.svg")
