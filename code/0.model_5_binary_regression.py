# coding:utf-8

# 引入需要的包
import warnings

# def ignore_warn(*args ,**kwargs):
#     pass
# warnings.warn = ignore_warn

# 引入模型
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split,cross_val_predict
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



from scipy import stats
from scipy.stats import norm, skew




# 读取数据
train = pd.read_csv('../data/d_train_20180102.csv',encoding='gbk')
test = pd.read_csv('../data/d_test_B_20180128.csv',encoding='gbk')

print('train shape',train.shape)
print('test shape',test.shape)

train_ID = train['id']
test_ID = test['id']

print('train feature shape',train.shape)
print('test feature shape',test.shape)

train = train[train['年龄'] >= 16]
train = train[train['血糖'] <= 18]

# print(train.shape)
import math
sub_array = []

# 9 , 10 , 11 ,12
def xxx(x):
    if x >=9:
        return 1
    else:
        return 0

train['l_血糖'] = train['血糖'].map(xxx)
# train['l_血糖'] = train['l_血糖'].map(int)

y_train = train['l_血糖']
t_train = train['血糖'].values

sex_map = {'男': 0, '女': 1, '??': 0}
train.drop(['血糖','l_血糖'], axis=1, inplace=True)
train['性别'] = train['性别'].map(sex_map)
test['性别'] = test['性别'].map(sex_map)

train.drop(['嗜碱细胞%','单核细胞%','白球比例','白蛋白','*总蛋白','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体','id','体检日期'],axis=1,inplace=True)
test.drop(['嗜碱细胞%','单核细胞%','白球比例','白蛋白','*总蛋白','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体','id','体检日期'],axis=1,inplace=True)

train['霉'] = train['*天门冬氨酸氨基转换酶'] + train['*丙氨酸氨基转换酶'] + train['*碱性磷酸酶'] + train['*r-谷氨酰基转换酶']
test['霉'] = test['*天门冬氨酸氨基转换酶'] + test['*丙氨酸氨基转换酶'] + test['*碱性磷酸酶'] + test['*r-谷氨酰基转换酶']

train['尿酸/肌酐'] = train['尿酸'] / train['肌酐']
test['尿酸/肌酐'] = test['尿酸'] / test['肌酐']

train['肾'] = train['尿酸'] + train['尿素'] + train['肌酐']
test['肾'] = test['尿酸'] + test['尿素'] + test['肌酐']

train['红细胞计数*红细胞平均血红蛋白量'] = train['红细胞计数'] * train['红细胞平均血红蛋白量']
test['红细胞计数*红细胞平均血红蛋白量'] = test['红细胞计数'] * test['红细胞平均血红蛋白量']

train['红细胞计数*红细胞平均血红蛋白浓度'] = train['红细胞计数'] * train['红细胞平均血红蛋白浓度']
test['红细胞计数*红细胞平均血红蛋白浓度'] = test['红细胞计数'] * test['红细胞平均血红蛋白浓度']

train['红细胞计数*红细胞平均体积'] = train['红细胞计数'] * train['红细胞平均体积']
test['红细胞计数*红细胞平均体积'] = test['红细胞计数'] * test['红细胞平均体积']
#
# train['红细胞平均血红蛋白量*红细胞平均血红蛋白浓度'] = train['红细胞平均血红蛋白量'] * train['红细胞平均血红蛋白浓度']
# test['红细胞平均血红蛋白量*红细胞平均血红蛋白浓度'] = test['红细胞平均血红蛋白量'] * test['红细胞平均血红蛋白浓度']

train['嗜酸细胞'] = train['白细胞计数'] * train['嗜酸细胞%']
test['嗜酸细胞'] = test['白细胞计数'] * test['嗜酸细胞%']
#
# 甘油三酯 总胆固醇 低密度脂蛋白胆固醇

# 红细胞平均体积
# train['血红蛋白/红细胞计数*红细胞平均血红蛋白浓度'] = train['血红蛋白'] / train['红细胞计数*红细胞平均血红蛋白浓度']
# test['血红蛋白/红细胞计数*红细胞平均血红蛋白浓度'] = test['血红蛋白'] / test['红细胞计数*红细胞平均血红蛋白浓度']

# train['CRF'] = 1.86 * train['肌酐'] - 1.164 * train['年龄'] * (1-train['性别'] * 0.26)
# test['CRF'] = 1.86 * test['肌酐'] - 1.164 * test['年龄'] * (1-train['性别'] * 0.26)

print('train featurs',train.columns)

print(train.head())


cv = KFold(n_splits=6,shuffle=True,random_state=42)

results = []
feature_import = pd.DataFrame()

feature_import['col'] = list(train.columns)

train = train.fillna(0)
test = test.fillna(0)

train = train.values
test = test.values
y_train = y_train.values
y_ppp = []

# from sklearn import svm
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

for traincv, testcv in cv.split(train,y_train):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'auc'
    param['scale_pos_weight'] = 1
    param['min_child_weight'] = 5

    xg_train = xgb.DMatrix(train[traincv], label=y_train[traincv])
    xg_test = xgb.DMatrix(train[testcv], label=y_train[testcv])

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 1000
    bst = xgb.train(param, xg_train, num_round, watchlist,early_stopping_rounds=25)
    y_tmp = bst.predict(xg_test)
    yuzhi = 0.55
    error = sum(y_train[testcv] != (y_tmp > yuzhi))
    print(sum((y_tmp > yuzhi)==1))
    print(error)


    sub = xgb.DMatrix(test)
    y_sub = bst.predict(sub)
    y_sub = y_sub >yuzhi
    sub_array.append(y_sub)
# print("Results: " + str( np.array(results).mean() ))

# # #
# print(np.array(sub_array))

s = 0
for i in sub_array:
    s = s + i
r = pd.DataFrame()
r['res_1'] = list(s)
print(r[r['res_1']>=1])
r['mp'] = r['res_1'] >=1
r['mp'] = r['mp'].astype(int)
r['mp'].to_csv('../mp_test.csv',index=False)
print(r[r['mp']>=1])
print(r.describe())
# print(r.max(),r.min(),r.mean())
#
#
# # results = []
# # feature_import = pd.DataFrame()
# # sub_array = []
# #
# # for model in [averaged_models]:
# #     for traincv, testcv in cv.split(train,y_train):
# #         # m = model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv],y_train[testcv])])
# #         m = model.fit(train[traincv], y_train[traincv])
# #         y_tmp = m.predict(train[testcv])
# #         y_ppp.append(y_tmp)
# #         # print(m.feature_importances_)
# #         # y_tmp = m.predict(train[testcv])
# #         res = mean_squared_error(y_train[testcv],y_tmp) / 2
# #         # feature_import[res] = list(m.feature_importances_)
# #         results.append(res)
# #         # sub_array.append(m.predict(test))
# #         sub_array.append(m.predict(test))
# #     print("Results: " + str( np.array(results).mean() ))
# #
# # # print(np.array(sub_array))
# # s = 0
# # for i in sub_array:
# #     s = s + i
# #
# # r['res_2'] = list(s/5)
# # # print(r)
#
# # r.loc[r.res_1 > 10,'res_1'] = 20
# # print(r.max(),r.min(),r.mean())
# # r['res_1'].to_csv('../result/result_0122_mut_2.csv', float_format='%.3f', index=None,header=None)
#
# # feature_import.to_csv('../result/ff.csv')

