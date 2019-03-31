import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

# print(df.attackerHasHelmet[df.ct_wins == 1].value_counts().sort_index())
# print(df.attackerHasHelmet[df.ct_wins == 0].value_counts().sort_index())



vals = ('attacker has helmet', 'attacker has no helmet')

idx = np.arange(len(vals))
bar_width = 0.25

plt.bar(idx, df.attackerHasHelmet[df.ct_wins == 1].value_counts().sort_index(), bar_width, label="class 1", color="b")
plt.bar(idx + bar_width, df.attackerHasHelmet[df.ct_wins == 0].value_counts().sort_index(), bar_width, label="class 0", color="r")
plt.xticks(idx + bar_width, vals)
plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-hist-attackerHasHelmet-class.svg")
