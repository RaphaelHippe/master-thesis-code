import numpy as np
import pandas as pd
import random
# colors = ['red', 'green', 'blue', 'yellow', 'orange',
          # 'purple', 'brown', 'violet', 'pink', 'cyan']
diameter_sizes = [15., 15.5, 16. ,17., 17.5, 18., 20., 22.5, 24., 25.5, 26., 27., 28., 29., 30.]
colors = [
    '#ff0000','#e50000','#cc0000','#b20000','#990000','#7f0000','#660000','#4c0000','#330000','#190000',
    '#008000','#007300','#006600','#005900','#004c00','#004000','#003300','#002600','#001900','#000c00',
    '#0000ff','#0000e5','#0000cc','#0000b2','#000099','#00007f','#000066','#00004c','#000033','#000019',
    '#ffff00','#e5e500','#cccc00','#b2b200','#999900','#7f7f00','#666600','#4c4c00','#333300','#191900',
    '#ffa500','#e59400','#cc8400','#b27300','#996300','#7f5200','#664200','#4c3100','#332100','#191000',
    '#800080','#730073','#660066','#590059','#4c004c','#400040','#330033','#260026','#190019','#0c000c',
    '#a52a2a','#942525','#842121','#731d1d','#631919','#521515','#421010','#310c0c','#210808','#100404',
    '#ee82ee','#d675d6','#be68be','#a65ba6','#8e4e8e','#774177','#5f345f','#472747','#2f1a2f','#170d17',
    '#ffc0cb','#e5acb6','#cc99a2','#b2868e','#997379','#7f6065','#664c51','#4c393c','#332628','#191314',
    '#00ffff','#00e5e5','#00cccc','#00b2b2','#009999','#007f7f','#006666','#004c4c','#003333','#001919'
]
diameter = []
goodPlate = []
color = []
for i1, c in enumerate(colors):
    for i2, d in enumerate(diameter_sizes):
        # if i2 >= 8 and i1 <= 4 or i2 < 8 and i1 > 4:
        if i2 >= 8 and i1 <= 49 or i2 < 8 and i1 > 49:
            label = True
        else:
            label = False
        diameter.append(d)
        color.append(c)
        goodPlate.append(label)
df = pd.DataFrame({
    'diameter': diameter,
    'goodPlate': goodPlate,
    'color': color
})
# df.to_csv('./plates_10.csv')
df.to_csv('./plates_100.csv')
