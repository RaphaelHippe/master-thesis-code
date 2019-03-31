import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ca
import nomtonum


df = pd.read_csv('./../datasets/damage_done.csv')
# y = df['goodPlate']
y = df['ct_wins']
# X0 = df.drop(['goodPlate', 'Unnamed: 0'], axis=1)
X0 = df.drop(['ct_wins', 't_wins'], axis=1)

# nom2num = nomtonum.NOMTONUM(['color'], [])

nom2num = nomtonum.NOMTONUM(['weapon', 'attackerSpotted', 'attackerSide',
'attackerIsScoped', 'attackerIsDucked', 'attackerIsDucking',
'attackerHasHelmet', 'victimSide', 'victimIsDucked', 'victimIsDucking',
'victimIsDefusing', 'victimIsScoped', 'victimHasHelmet', 'hitgroup'],
['attackerXPosition', 'attackerYPosition', 'attackerZPosition',
'victimXPosition', 'victimYPosition', 'victimZPosition'])

X = nom2num.fit_transform(X0)

# X['goodPlate'] = df['goodPlate']
X['ct_wins'] = df['ct_wins']

X.to_csv('dd_nom2num.csv')
