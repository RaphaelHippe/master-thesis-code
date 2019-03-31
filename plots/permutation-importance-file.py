import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


DATASET = 'p10'
ENCODER = 'binary'
CLASSIFIER = 'rfc'
FILENAME = '{}-{}-{}-50'.format(DATASET, ENCODER, CLASSIFIER)
df = pd.read_csv('./../eval/results/{}.csv'.format(FILENAME))
df.drop(['Unnamed: 0'], inplace=True, axis=1)

# print()

df.loc[df['feature'] == 'diameter', ['feature']] = 'x1'
df.loc[df['feature'] == df.feature.values[1], ['feature']] = 'x2'


ax = df.boxplot(
column='acc_weight',
by='feature',
figsize=cm2inch(14.69, 10.5),
rot=90,
fontsize=6
)

ax.set_xlabel('')

plt.suptitle("")
plt.subplots_adjust(bottom=0.3)

plt.show()
# plt.savefig("./images/p100-permutation-importance-boxplot-numerical.svg")
