import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../datasets/player_stats.csv')
fig = plt.figure()

df.t_0_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p0', color='green')
df.t_0_kills[df.ct_wins == 0].plot(kind='density', label='t_win p0', color='red')
df.t_1_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p1', color='forestgreen')
df.t_1_kills[df.ct_wins == 0].plot(kind='density', label='t_win p1', color='lightcoral')
df.t_2_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p2', color='darkgreen')
df.t_2_kills[df.ct_wins == 0].plot(kind='density', label='t_win p2', color='coral')
df.t_3_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p3', color='seagreen')
df.t_3_kills[df.ct_wins == 0].plot(kind='density', label='t_win p3', color='chocolate')
df.t_4_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p4', color='olivedrab')
df.t_4_kills[df.ct_wins == 0].plot(kind='density', label='t_win p4', color='sandybrown')
df.ct_0_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p5', color='olive')
df.ct_0_kills[df.ct_wins == 0].plot(kind='density', label='t_win p5', color='tomato')
df.ct_1_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p6', color='darkseagreen')
df.ct_1_kills[df.ct_wins == 0].plot(kind='density', label='t_win p6', color='firebrick')
df.ct_2_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p7', color='lightseagreen')
df.ct_2_kills[df.ct_wins == 0].plot(kind='density', label='t_win p7', color='orangered')
df.ct_3_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p8', color='lime')
df.ct_3_kills[df.ct_wins == 0].plot(kind='density', label='t_win p8', color='peru')
df.ct_4_kills[df.ct_wins == 1].plot(kind='density', label='ct_win p9', color='limegreen')
df.ct_4_kills[df.ct_wins == 0].plot(kind='density', label='t_win p9', color='darkorange')

# df.victimXPosition[df.ct_wins == 1].plot(kind='density', label='ct_win victim', color='lightgreen')
# df.victimXPosition[df.ct_wins == 0].plot(kind='density', label='t_win victim', color='lightcoral')

plt.xlabel('player kills')


plt.legend(loc=2)

# plt.show()
plt.savefig("./images/ps-conditional-density-kills-class.svg")
