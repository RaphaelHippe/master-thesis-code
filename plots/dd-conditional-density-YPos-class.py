import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df.attackerYPosition[df.ct_wins == 1].plot(kind='density', label='class 1 attacker', color='blue')
df.attackerYPosition[df.ct_wins == 0].plot(kind='density', label='class 0 attacker', color='red')

df.victimYPosition[df.ct_wins == 1].plot(kind='density', label='class 1 victim', color='lightblue')
df.victimYPosition[df.ct_wins == 0].plot(kind='density', label='class 0 victim', color='lightcoral')

plt.xlabel('player y position')


plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-density-YPosition-class.svg")
