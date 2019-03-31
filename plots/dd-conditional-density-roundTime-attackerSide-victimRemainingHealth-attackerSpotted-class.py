import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df.roundTime[(df.ct_wins == 1)
            & (df.attackerSide == 'Terrorist')
            & (df.victimRemainingHealth == 0)
            & (df.attackerSpotted == False)
            ].plot(kind='density', label='class 1', color='b')
df.roundTime[(df.ct_wins == 0)
            & (df.attackerSide == 'Terrorist')
            & (df.victimRemainingHealth == 0)
            & (df.attackerSpotted == False)
            ].plot(kind='density', label='class 0', color='r')

plt.xlabel('round time')


plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-density-roundTime-attackerSide-victimRemainingHealth-attackerSpotted-class.svg")
