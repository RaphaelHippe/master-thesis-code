import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/player_stats.csv')
fig = plt.figure()

# print(df.t_1_hasHelmet[df.ct_wins == 1].value_counts().sort_index())
# print(df.t_1_hasHelmet[df.ct_wins == 0].value_counts().sort_index())



vals = ('t1 has helmet', 't1 has no helmet')

idx = np.arange(len(vals))
bar_width = 0.25

plt.bar(idx, df.t_1_hasHelmet[df.ct_wins == 1].value_counts().sort_index(), bar_width, label="class 1", color="b")
plt.bar(idx + bar_width, df.t_1_hasHelmet[df.ct_wins == 0].value_counts().sort_index(), bar_width, label="class 0", color="r")
plt.xticks(idx + bar_width, vals)
plt.legend(loc=9)

# plt.show()
plt.savefig("./images/ps-conditional-hist-t_1_hasHelmet-class.svg")
