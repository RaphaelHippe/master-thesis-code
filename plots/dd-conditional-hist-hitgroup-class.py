import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()

# print(df.hitgroup[df.ct_wins == 1].value_counts().sort_index())
# print(df.hitgroup[df.ct_wins == 0].value_counts().sort_index())

vals = ('0', '1', '2', '3', '4', '5', '6', '7')

idx = np.arange(len(vals))
bar_width = 0.25

plt.bar(idx, df.hitgroup[df.ct_wins == 1].value_counts().sort_index(), bar_width, label="class 1", color="b")
plt.bar(idx + bar_width, df.hitgroup[df.ct_wins == 0].value_counts().sort_index(), bar_width, label="class 0", color="r")
plt.xticks(idx + bar_width, vals)
plt.legend(loc=2)

# plt.show()
plt.savefig("./images/dd-conditional-hist-hitgroup-class.svg")
