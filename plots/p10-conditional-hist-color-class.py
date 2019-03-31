import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/plates_10.csv')
fig = plt.figure()

idx = np.arange(10)
bar_width = 0.25

print(df.color[df.goodPlate == 1].value_counts().sort_index())
print(df.color[df.goodPlate == 0].value_counts().sort_index())

plt.bar(idx, df.color[df.goodPlate == 1].value_counts().sort_index(), bar_width, label="class 1", color="b")
plt.bar(idx + bar_width, df.color[df.goodPlate == 0].value_counts().sort_index(), bar_width, label="class 0", color="r")
plt.xticks(idx + bar_width, ('red', 'brown', 'cyan', 'green', 'orange', 'pink', 'purple', 'red', 'violet', 'yellow'))
plt.legend(loc=2)

# plt.show()
plt.savefig("./images/p10-conditional-hist-color-class.svg")
