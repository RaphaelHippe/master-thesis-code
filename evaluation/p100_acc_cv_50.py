import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import PermutationImportance
import category_encoders as ce

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


df = pd.read_csv('./../datasets/plates_100_new.csv')
y = df['goodPlate']
X0 = df.drop(['goodPlate', 'Unnamed: 0'], axis=1)

nominal_columns = ['color']



def doPermutationTests(X, y, features, encoding):
    columns = ['x1_knn','x2_knn','x1_rfc','x2_rfc','x1_gnb','x2_gnb']
    data = []
    for _ in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        # KNN
        clf_knn = KNeighborsClassifier(n_neighbors=3)
        clf_knn.fit(X_train, y_train)
        pi_knn = PermutationImportance.PermutationImportance(clf_knn)
        pi_knn.fit(X_test, y_test, features=features)
        pi_knn_df = pi_knn.get_weights()
        # KNN
        clf_rfc = RandomForestClassifier()
        clf_rfc.fit(X_train, y_train)
        pi_rfc = PermutationImportance.PermutationImportance(clf_rfc)
        pi_rfc.fit(X_test, y_test, features=features)
        pi_rfc_df = pi_rfc.get_weights()
        # GNB
        clf_gnb = GaussianNB()
        clf_gnb.fit(X_train, y_train)
        pi_gnb = PermutationImportance.PermutationImportance(clf_gnb)
        pi_gnb.fit(X_test, y_test, features=features)
        pi_gnb_df = pi_gnb.get_weights()

        pi_knn_df.loc[pi_knn_df['feature'] == 'diameter', 'feature'] = 'x1'
        pi_knn_df.loc[pi_knn_df['feature'] != 'x1', 'feature'] = 'x2'

        pi_rfc_df.loc[pi_rfc_df['feature'] == 'diameter', 'feature'] = 'x1'
        pi_rfc_df.loc[pi_rfc_df['feature'] != 'x1', 'feature'] = 'x2'

        pi_gnb_df.loc[pi_gnb_df['feature'] == 'diameter', 'feature'] = 'x1'
        pi_gnb_df.loc[pi_gnb_df['feature'] != 'x1', 'feature'] = 'x2'

        data.append([
            pi_knn_df['acc_weight'].loc[pi_knn_df['feature'] == 'x1'].iloc[0],
            pi_knn_df['acc_weight'].loc[pi_knn_df['feature'] == 'x2'].iloc[0],
            pi_rfc_df['acc_weight'].loc[pi_rfc_df['feature'] == 'x1'].iloc[0],
            pi_rfc_df['acc_weight'].loc[pi_rfc_df['feature'] == 'x2'].iloc[0],
            pi_gnb_df['acc_weight'].loc[pi_gnb_df['feature'] == 'x1'].iloc[0],
            pi_gnb_df['acc_weight'].loc[pi_gnb_df['feature'] == 'x2'].iloc[0]
        ])

    df = pd.DataFrame(data, columns=columns)
    df.boxplot(rot=90, figsize=cm2inch(8, 8))
    plt.subplots_adjust(bottom=0.2)
    plt.ylim(-1, 1)
    plt.savefig("./images/p100-permutation-importance-boxplot-{}_new.svg".format(encoding))
    plt.clf()




def doAccuracyTests(X,y, encoding):
    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    acc_scores_knn = cross_val_score(clf_knn, X, y, cv=50)
    # RFC
    clf_rfc = RandomForestClassifier()
    acc_scores_rfc = cross_val_score(clf_rfc, X, y, cv=50)
    # GNB
    clf_gnb = GaussianNB()
    acc_scores_gnb = cross_val_score(clf_gnb, X, y, cv=50)
    return [encoding, np.mean(acc_scores_knn), np.mean(acc_scores_rfc), np.mean(acc_scores_gnb)]

results = []
encoder = ce.OneHotEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'onehot'))
features = ['diameter',
[
'color_1','color_2', 'color_3', 'color_4', 'color_5', 'color_6', 'color_7', 'color_8', 'color_9',
'color_10','color_11','color_12', 'color_13', 'color_14', 'color_15', 'color_16', 'color_17', 'color_18', 'color_19',
'color_20','color_21','color_22', 'color_23', 'color_24', 'color_25', 'color_26', 'color_27', 'color_28', 'color_29',
'color_30','color_31','color_32', 'color_33', 'color_34', 'color_35', 'color_36', 'color_37', 'color_38', 'color_39',
'color_40','color_41','color_42', 'color_43', 'color_44', 'color_45', 'color_46', 'color_47', 'color_48', 'color_49',
'color_50','color_51','color_52', 'color_53', 'color_54', 'color_55', 'color_56', 'color_57', 'color_58', 'color_59',
'color_60','color_61','color_62', 'color_63', 'color_64', 'color_65', 'color_66', 'color_67', 'color_68', 'color_69',
'color_70','color_71','color_72', 'color_73', 'color_74', 'color_75', 'color_76', 'color_77', 'color_78', 'color_79',
'color_80','color_81','color_82', 'color_83', 'color_84', 'color_85', 'color_86', 'color_87', 'color_88', 'color_89',
'color_90','color_91','color_92', 'color_93', 'color_94', 'color_95', 'color_96', 'color_97', 'color_98', 'color_99',
'color_100', 'color_-1'
]]
doPermutationTests(X, y, features, 'onehot')

X = pd.get_dummies(X0)
results.append(doAccuracyTests(X, y, 'dummy'))
features = ['diameter', [
'color_#000019','color_#000033','color_#00004c','color_#000066','color_#00007f',
'color_#000099','color_#0000b2','color_#0000cc','color_#0000e5','color_#0000ff',
'color_#000c00','color_#001900','color_#001919','color_#002600','color_#003300',
'color_#003333','color_#004000','color_#004c00','color_#004c4c','color_#005900',
'color_#006600','color_#006666','color_#007300','color_#007f7f','color_#008000',
'color_#009999','color_#00b2b2','color_#00cccc','color_#00e5e5','color_#00ffff',
'color_#0c000c','color_#100404','color_#170d17','color_#190000','color_#190019',
'color_#191000','color_#191314','color_#191900','color_#210808','color_#260026',
'color_#2f1a2f','color_#310c0c','color_#330000','color_#330033','color_#332100',
'color_#332628','color_#333300','color_#400040','color_#421010','color_#472747',
'color_#4c0000','color_#4c004c','color_#4c3100','color_#4c393c','color_#4c4c00',
'color_#521515','color_#590059','color_#5f345f','color_#631919','color_#660000',
'color_#660066','color_#664200','color_#664c51','color_#666600','color_#730073',
'color_#731d1d','color_#774177','color_#7f0000','color_#7f5200','color_#7f6065',
'color_#7f7f00','color_#800080','color_#842121','color_#8e4e8e','color_#942525',
'color_#990000','color_#996300','color_#997379','color_#999900','color_#a52a2a',
'color_#a65ba6','color_#b20000','color_#b27300','color_#b2868e','color_#b2b200',
'color_#be68be','color_#cc0000','color_#cc8400','color_#cc99a2','color_#cccc00',
'color_#d675d6','color_#e50000','color_#e59400','color_#e5acb6','color_#e5e500',
'color_#ee82ee','color_#ff0000','color_#ffa500','color_#ffc0cb','color_#ffff00',
]]
doPermutationTests(X, y, features, 'dummy')

encoder = ce.BackwardDifferenceEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'difference'))
features = ['diameter',
[
'color_0','color_1','color_2', 'color_3', 'color_4', 'color_5', 'color_6', 'color_7', 'color_8', 'color_9',
'color_10','color_11','color_12', 'color_13', 'color_14', 'color_15', 'color_16', 'color_17', 'color_18', 'color_19',
'color_20','color_21','color_22', 'color_23', 'color_24', 'color_25', 'color_26', 'color_27', 'color_28', 'color_29',
'color_30','color_31','color_32', 'color_33', 'color_34', 'color_35', 'color_36', 'color_37', 'color_38', 'color_39',
'color_40','color_41','color_42', 'color_43', 'color_44', 'color_45', 'color_46', 'color_47', 'color_48', 'color_49',
'color_50','color_51','color_52', 'color_53', 'color_54', 'color_55', 'color_56', 'color_57', 'color_58', 'color_59',
'color_60','color_61','color_62', 'color_63', 'color_64', 'color_65', 'color_66', 'color_67', 'color_68', 'color_69',
'color_70','color_71','color_72', 'color_73', 'color_74', 'color_75', 'color_76', 'color_77', 'color_78', 'color_79',
'color_80','color_81','color_82', 'color_83', 'color_84', 'color_85', 'color_86', 'color_87', 'color_88', 'color_89',
'color_90','color_91','color_92', 'color_93', 'color_94', 'color_95', 'color_96', 'color_97', 'color_98'
]]
doPermutationTests(X, y, features, 'difference')

encoder = ce.BaseNEncoder(base=3, cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'basen'))
features = ['diameter', ['color_0','color_1',
'color_2', 'color_3', 'color_4', 'color_5']]
doPermutationTests(X, y, features, 'basen')

encoder = ce.BinaryEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'binary'))
features = ['diameter', ['color_0','color_1',
'color_2', 'color_3', 'color_4', 'color_5', 'color_6', 'color_7']]
doPermutationTests(X, y, features, 'binary')

encoder = ce.HelmertEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'helmert'))
features = ['diameter',
[
'color_0','color_1','color_2', 'color_3', 'color_4', 'color_5', 'color_6', 'color_7', 'color_8', 'color_9',
'color_10','color_11','color_12', 'color_13', 'color_14', 'color_15', 'color_16', 'color_17', 'color_18', 'color_19',
'color_20','color_21','color_22', 'color_23', 'color_24', 'color_25', 'color_26', 'color_27', 'color_28', 'color_29',
'color_30','color_31','color_32', 'color_33', 'color_34', 'color_35', 'color_36', 'color_37', 'color_38', 'color_39',
'color_40','color_41','color_42', 'color_43', 'color_44', 'color_45', 'color_46', 'color_47', 'color_48', 'color_49',
'color_50','color_51','color_52', 'color_53', 'color_54', 'color_55', 'color_56', 'color_57', 'color_58', 'color_59',
'color_60','color_61','color_62', 'color_63', 'color_64', 'color_65', 'color_66', 'color_67', 'color_68', 'color_69',
'color_70','color_71','color_72', 'color_73', 'color_74', 'color_75', 'color_76', 'color_77', 'color_78', 'color_79',
'color_80','color_81','color_82', 'color_83', 'color_84', 'color_85', 'color_86', 'color_87', 'color_88', 'color_89',
'color_90','color_91','color_92', 'color_93', 'color_94', 'color_95', 'color_96', 'color_97', 'color_98'
]]
doPermutationTests(X, y, features, 'helmert')

encoder = ce.SumEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'sum'))
features = ['diameter',
[
'color_0','color_1','color_2', 'color_3', 'color_4', 'color_5', 'color_6', 'color_7', 'color_8', 'color_9',
'color_10','color_11','color_12', 'color_13', 'color_14', 'color_15', 'color_16', 'color_17', 'color_18', 'color_19',
'color_20','color_21','color_22', 'color_23', 'color_24', 'color_25', 'color_26', 'color_27', 'color_28', 'color_29',
'color_30','color_31','color_32', 'color_33', 'color_34', 'color_35', 'color_36', 'color_37', 'color_38', 'color_39',
'color_40','color_41','color_42', 'color_43', 'color_44', 'color_45', 'color_46', 'color_47', 'color_48', 'color_49',
'color_50','color_51','color_52', 'color_53', 'color_54', 'color_55', 'color_56', 'color_57', 'color_58', 'color_59',
'color_60','color_61','color_62', 'color_63', 'color_64', 'color_65', 'color_66', 'color_67', 'color_68', 'color_69',
'color_70','color_71','color_72', 'color_73', 'color_74', 'color_75', 'color_76', 'color_77', 'color_78', 'color_79',
'color_80','color_81','color_82', 'color_83', 'color_84', 'color_85', 'color_86', 'color_87', 'color_88', 'color_89',
'color_90','color_91','color_92', 'color_93', 'color_94', 'color_95', 'color_96', 'color_97', 'color_98'
]]
doPermutationTests(X, y, features, 'sum')

encoder = ce.LeaveOneOutEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'leaveoneout'))
features = ['diameter', 'color']
doPermutationTests(X, y, features, 'leaveoneout')

encoder = ce.TargetEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'target'))
features = ['diameter', 'color']
doPermutationTests(X, y, features, 'target')

encoder = ce.OrdinalEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'ordinal'))
features = ['diameter', 'color']
doPermutationTests(X, y, features, 'ordinal')

encoder = ce.WOEEncoder(cols=nominal_columns).fit(X0, y)
X = encoder.transform(X0)
results.append(doAccuracyTests(X, y, 'woe'))
features = ['diameter', 'color']
doPermutationTests(X, y, features, 'woe')


df = pd.DataFrame(results, columns=['encoding', 'knn', 'rfc', 'gnb'])
df.to_csv('./acc/p100_cv50_new.csv')
