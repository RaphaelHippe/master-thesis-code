import pandas as pd
import category_encoders as ce
import os

NAME = 'damage_done'
DATASET = 'dd'
DF_ORIGINAL = pd.read_csv('./../datasets/{}.csv'.format(NAME))
FILENAME = './latex_{}.txt'.format(NAME)

def renameX(X):
    X = X.rename(index=str, columns={
        "attackerHealth": "x1",
        "attackerXPosition": "x2",
        "attackerYPosition": "x3",
        "attackerZPosition": "x4",
        "weapon": "x5",
        "attackerSpotted": "x6",
        "attackerSide": "x7",
        "attackerIsScoped": "x8",
        "attackerIsDucked": "x9",
        "attackerIsDucking": "x10",
        "attackerHasHelmet": "x11",
        "victimRemainingHealth": "x12",
        "victimRemainingArmor": "x13",
        "victimXPosition": "x14",
        "victimYPosition": "x15",
        "victimZPosition": "x16",
        "victimSide": "x17",
        "victimIsDucked": "x18",
        "victimIsDucking": "x19",
        "victimIsDefusing": "x20",
        "victimIsScoped": "x21",
        "victimHasHelmet": "x22",
        "damageArmor": "x23",
        "damageHealth": "x24",
        "hitgroup": "x25",
        "roundTime": "x26",
        "round_number": "x27"
    })
    return X

def describeData(df, encoding):
    begintable = '\\begin{table}\n\\begin{center}'
    caption = '\\caption{{Pandas describe table for encoding {} on {} data set.}}'.format(encoding, DATASET)
    endtable = '{}\\end{{center}}\n\\end{{table}}'.format(caption)
    latex = '{}\n{}{}'.format(begintable, df.describe(include = 'all').round(2).T.to_latex(), endtable)
    return '{} \n'.format(latex)

def writeFile(input):
    if os.path.isfile(FILENAME):
        with open(FILENAME, 'a') as myfile:
            myfile.write(input)
            myfile.close()
    else:
        with open(FILENAME, 'w') as myfile:
            myfile.write(input)
            myfile.close()



nominal_columns = ['x5', 'x6', 'x7',
                   'x8', 'x9', 'x10',
                   'x11', 'x17', 'x18', 'x19',
                   'x20', 'x21', 'x22', 'x25']
# nominal_columns = ['x2']
# y = DF_ORIGINAL['goodPlate']
# X0 = DF_ORIGINAL.drop(['goodPlate', 'Unnamed: 0'], axis=1)
# X0 = X0.rename(index=str, columns={
#     "diameter": "x1",
#     "color": "x2"
# })
y = DF_ORIGINAL['ct_wins']
X1 = DF_ORIGINAL.drop(['ct_wins', 't_wins'], axis=1)
X0 = renameX(X1)

encoder = ce.OneHotEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
writeFile(describeData(X, 'OneHot'))

X = pd.get_dummies(X0)
writeFile(describeData(X, 'Dummy'))

encoder = ce.BackwardDifferenceEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
X = X.drop(['intercept'], axis=1)
writeFile(describeData(X, 'BackwardDifference'))

encoder = ce.BaseNEncoder(base=3, cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
writeFile(describeData(X, 'BaseN'))

encoder = ce.BinaryEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
writeFile(describeData(X, 'Binary'))

encoder = ce.HelmertEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
X.drop(['intercept'], inplace=True, axis=1)
writeFile(describeData(X, 'Helmert'))

encoder = ce.SumEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
X.drop(['intercept'], inplace=True, axis=1)
writeFile(describeData(X, 'Sum'))

# extra stuff because of renaming bug on DD set
nominal_columns = [
'weapon',
'attackerSpotted',
'attackerSide',
'attackerIsScoped',
'attackerIsDucked',
'attackerIsDucking',
'attackerHasHelmet',
'victimSide',
'victimIsDucked',
'victimIsDucking',
'victimIsDefusing',
'victimIsScoped',
'victimHasHelmet',
'hitgroup',
]

encoder = ce.LeaveOneOutEncoder(cols=nominal_columns).fit(X1, y)
X = encoder.transform(X1)
X = renameX(X)
writeFile(describeData(X, 'LeaveOneOut'))

encoder = ce.TargetEncoder(cols=nominal_columns).fit(X1, y)
X = encoder.transform(X1)
X = renameX(X)
writeFile(describeData(X, 'Target'))

encoder = ce.OrdinalEncoder(cols=nominal_columns).fit(X1, y)
X = encoder.transform(X1)
X = renameX(X)
writeFile(describeData(X, 'Ordinal'))

encoder = ce.WOEEncoder(cols=nominal_columns).fit(X1, y)
X = encoder.transform(X1)
X = renameX(X)
writeFile(describeData(X, 'WOE'))
