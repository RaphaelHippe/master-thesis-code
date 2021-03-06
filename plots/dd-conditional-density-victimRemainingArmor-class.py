import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

df.victimRemainingArmor[df.ct_wins == 1].plot(kind='density', label='class 1', color='b')
df.victimRemainingArmor[df.ct_wins == 0].plot(kind='density', label='class 0', color='r')

plt.xlabel('remaining armor of vicitim')


plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-density-victimRemainingArmor-class.svg")
