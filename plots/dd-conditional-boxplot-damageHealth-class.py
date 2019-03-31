import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')

ax = df.boxplot(column='damageHealth', by='ct_wins')
plt.suptitle("")

# plt.show()
plt.savefig("./images/dd-conditional-boxplot-damageHealth-class.svg")
