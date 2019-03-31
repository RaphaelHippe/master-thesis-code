import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df.attackerXPosition[df.ct_wins == 1].plot(kind='density', label='class 1 attacker', color='blue')
df.attackerXPosition[df.ct_wins == 0].plot(kind='density', label='class 0 attacker', color='red')

df.victimXPosition[df.ct_wins == 1].plot(kind='density', label='class 1 victim', color='lightblue')
df.victimXPosition[df.ct_wins == 0].plot(kind='density', label='class 0 victim', color='lightcoral')

plt.xlabel('player x position')


plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-density-XPosition-class.svg")
