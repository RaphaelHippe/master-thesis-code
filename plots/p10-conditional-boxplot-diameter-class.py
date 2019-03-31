import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/plates_10.csv')

ax = df.boxplot(column='diameter', by='goodPlate')
plt.suptitle("")

# plt.show()
plt.savefig("./images/p10-conditional-boxplot-diameter-class.svg")
