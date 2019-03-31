import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df.attackerZPosition[df.ct_wins == 1].plot(kind='density', label='class 1 attacker', color='blue')
df.attackerZPosition[df.ct_wins == 0].plot(kind='density', label='class 0 attacker', color='red')

df.victimZPosition[df.ct_wins == 1].plot(kind='density', label='class 1 victim', color='lightblue')
df.victimZPosition[df.ct_wins == 0].plot(kind='density', label='class 0 victim', color='lightcoral')

plt.xlabel('player z position')


plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-density-ZPosition-class.svg")
