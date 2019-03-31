import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/damage_done.csv')
fig = plt.figure()



# print(df.weapon[df.ct_wins == 1].value_counts().sort_index())
# print(df.weapon[df.ct_wins == 0].value_counts().sort_index())

df_win = df[df.ct_wins == 1]
df_loss = df[df.ct_wins == 0]

vals = ['ak47','aug','awp','bizon','cz75a','deagle','decoy_projectile','elite',
'famas','fiveseven','flashbang','g3sg1','galilar','glock','hegrenade','hkp2000',
'inferno','knife','m4a1','m4a1_silencer','mac10','mag7','molotov_projectile','mp7',
'mp9','nova','p250','p90','scar20','sg556','smokegrenade','ssg08','taser','tec9',
'ump45','usp_silencer','xm1014']

sum = 0
win_counts = []
loss_counts = []
for val in vals:
    win_amount = len(df_win[df_win.weapon == val])
    loss_amount = len(df_loss[df_loss.weapon == val])
    sum = sum + win_amount + loss_amount
    win_counts.append(win_amount)
    loss_counts.append(loss_amount)

idx = np.arange(37)
bar_width = 0.25

plt.bar(idx, win_counts, bar_width, label="class 1", color="b")
plt.bar(idx + bar_width, loss_counts, bar_width, label="class 0", color="r")
plt.xticks(idx + bar_width, vals, rotation='vertical')
plt.subplots_adjust(bottom=0.3)

plt.legend(loc=1)

# plt.show()
plt.savefig("./images/dd-conditional-hist-weapon-class.svg")
