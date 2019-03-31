import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./../datasets/plates_10.csv')

df.drop(['Unnamed: 0', 'goodPlate'], inplace=True, axis=1)
df.boxplot(column=['diameter'], by='color')

plt.show()
