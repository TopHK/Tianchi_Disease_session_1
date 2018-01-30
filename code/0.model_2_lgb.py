# coding:utf-8

# 引入需要的包

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew


# 读取数据

train = pd.read_csv('../data/d_train_20180102.csv',encoding='gbk')
train_ext = pd.read_csv('../data/d_test_A_20180102.csv',encoding='gbk')
train_ext_ans = pd.read_csv('../data/d_answer_a_20180128.csv',encoding='gbk',header=None)
train_ext = pd.concat([train_ext,train_ext_ans],axis=1)
train_ext.rename(columns={0:'血糖'},inplace=True)
train = pd.concat([train,train_ext],axis=0)

test = pd.read_csv('../data/d_test_B_20180128.csv',encoding='gbk')
# mp_test = pd.read_csv('../mp_test.csv',header=None)
#
print('train shape',train.shape)
print('test shape',test.shape)

train_ID = train['id']
test_ID = test['id']



print('train feature shape',train.shape)
print('test feature shape',test.shape)

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

from scipy.special import boxcox, inv_boxcox
# Validation function
# y_train, lambda_ = stats.boxcox(train['血糖'])
# train = train[train['血糖']<=38]
#
train = train[train['年龄'] >= 16]
train = train[train['血糖'] <= 18]

print(train.shape)

def xxx(x):
    return x
    # if x >= 11:
    #     return 1
    # else:
    #     return 0

# train['mp'] = train['血糖'].map(xxx)
# test = pd.concat([test,mp_test],axis=1)
# test.rename(columns={0:'mp'},inplace=True)
# test['mp'] = test[test.index.isin(list([33,208,247,292,313,403,601,628,928,938,951]))]

print(test)


y_train = train['血糖']

sex_map = {'男': 0, '女': 1, '??': 0}
train.drop(['血糖'], axis=1, inplace=True)
train['性别'] = train['性别'].map(sex_map)
test['性别'] = test['性别'].map(sex_map)


# from dateutil.parser import parse
# train['体检日期_1'] = (pd.to_datetime(train['体检日期']) - parse('2017-10-09')).dt.days
# test['体检日期_1'] = (pd.to_datetime(test['体检日期']) - parse('2017-10-09')).dt.days

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

# train['红细胞平均血红蛋白量*红细胞平均血红蛋白浓度'] = train['红细胞平均血红蛋白量'] * train['红细胞平均血红蛋白浓度']
# test['红细胞平均血红蛋白量*红细胞平均血红蛋白浓度'] = test['红细胞平均血红蛋白量'] * test['红细胞平均血红蛋白浓度']
#
train['嗜酸细胞'] = train['白细胞计数'] * train['嗜酸细胞%']
test['嗜酸细胞'] = test['白细胞计数'] * test['嗜酸细胞%']

# for i in train.columns:
#     train['log%s'%(i)] = pd.np.log1p(train[i])
#     test['log%s'%(i)] = pd.np.log1p(test[i])

from scipy.stats import kendalltau
# scipy,stats.kendalltau(a, b, initial_lexsort=None, nan_policy='omit')

# ff = []
# for i in train.columns:
#     tmp = kendalltau(train[i],y_train, initial_lexsort=None, nan_policy='omit')
#     print(tmp)
#     print(tmp[0])
#     if abs(tmp[0]) > 0.05:
#         ff.append(i)

# train = train[ff]
# test = test[ff]

print(train.columns)

# train['嗜碱细胞/嗜酸细胞'] = train['嗜碱细胞%'] % train['嗜酸细胞%']
# test['嗜碱细胞/嗜酸细胞'] = test['嗜碱细胞%'] % test['嗜酸细胞%']

# 甘油三酯 总胆固醇 低密度脂蛋白胆固醇

# 红细胞平均体积

# train['血红蛋白/红细胞计数*红细胞平均血红蛋白浓度'] = train['血红蛋白'] / train['红细胞计数*红细胞平均血红蛋白浓度']
# test['血红蛋白/红细胞计数*红细胞平均血红蛋白浓度'] = test['血红蛋白'] / test['红细胞计数*红细胞平均血红蛋白浓度']

# train['CRF'] = 1.86 * train['肌酐'] - 1.164 * train['年龄'] * (1-train['性别'] * 0.26)
# test['CRF'] = 1.86 * test['肌酐'] - 1.164 * test['年龄'] * (1-train['性别'] * 0.26)

# train['血小板比积*血小板平均体积'] = train['血小板比积'] / train['血小板平均体积']
# test['血小板比积*血小板平均体积'] = test['血小板比积'] / test['血小板平均体积']


# 血小板体积分布宽度
# train = train.fillna(0)
print('train featurs',train.columns)

# n_folds = 5

# CV model
def rmsle_cv(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
    score = -cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error" ,cv=kf)
    # y_pred_in = inv_boxcox(y_pred,lambda_)
    # y_pred_in = y_pred
    # y_train_in = inv_boxcox(y_train,lambda_)
    # y_train_in = y_train
    return score / 2


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves= 8 ,
                              learning_rate=0.05, n_estimators=1000,
                              max_bin = 20, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)

GBoost = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.05,
                                   max_depth=10, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

# score = rmsle_cv(GBoost)
# print(score.mean(),score.std())
#
# score = rmsle_cv(model_lgb)
# print(score.mean(),score.std())


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

#
averaged_models = AveragingModels(models = (GBoost,model_lgb ))
#
# score = rmsle_cv(averaged_models)
# print(score.mean(),score.std())

# from sklearn import cross_validation
# # GBoost model_lgb averaged_models
cv = KFold(n_splits=6,shuffle=True,random_state=42)
#
results = []
feature_import = pd.DataFrame()
sub_array = []
feature_import['col'] = list(train.columns)
# print(train.columns)
# train = train.fillna(0.1)
# test = test.fillna(0.1)
train = train.values
test = test.values
y_train = y_train.values
y_ppp = []
# model cv



def xx(x):
    max_x = max(x) * 0.95
    qut_x =  max(x) * 0.9
    min_x = max(x) * 0.25

    # for i,t in enumerate(x):
    #     if x[i] >= max_x:
    #         x[i] = x[i] * 1.25
    #     elif (x[i] < max_x) & (x[i] >= qut_x):
    #         x[i] = x[i]
    #     elif x[i] <= min_x:
    #         x[i] = x[i] * 0.75
    #     else:
    #         x[i] = x[i]
    return x

for model in [model_lgb]:
    for traincv, testcv in cv.split(train,y_train):
        # m = model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv],y_train[testcv])])
        m = model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv],y_train[testcv])],eval_metric='mse',early_stopping_rounds=150)
        y_tmp = m.predict(train[testcv])

        print(m.feature_importances_)
        # y_tmp = m.predict(train[testcv])
        y_ppp.append(y_tmp)

        res = mean_squared_error(y_train[testcv],xx(y_tmp) ) / 2
        feature_import[res] = list(m.feature_importances_)
        results.append(res)
        # sub_array.append(m.predict(test))
        sub_array.append(xx(m.predict(test)))
    print("Results: " + str( np.array(results).mean() ))

# print(np.array(sub_array))
s = 0
for i in sub_array:
    s = s + i

r = pd.DataFrame()
r['res_1'] = list(s/6)

print(r.describe())
# print(r.max(),r.min(),r.mean())


# results = []
# feature_import = pd.DataFrame()
# sub_array = []
#
# for model in [averaged_models]:
#     for traincv, testcv in cv.split(train,y_train):
#         # m = model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv],y_train[testcv])])
#         m = model.fit(train[traincv], y_train[traincv])
#         y_tmp = m.predict(train[testcv])
#         y_ppp.append(y_tmp)
#         # print(m.feature_importances_)
#         # y_tmp = m.predict(train[testcv])
#         res = mean_squared_error(y_train[testcv],y_tmp) / 2
#         # feature_import[res] = list(m.feature_importances_)
#         results.append(res)
#         # sub_array.append(m.predict(test))
#         sub_array.append(m.predict(test))
#     print("Results: " + str( np.array(results).mean() ))
#
# # print(np.array(sub_array))
# s = 0
# for i in sub_array:
#     s = s + i
#
# r['res_2'] = list(s/5)
# # print(r)

# r.loc[r.res_1 > 10,'res_1'] = 20
# print(r.max(),r.min(),r.mean())
# r = r.reset_index()

# print(r)

# r.loc[r.index == 938,'res_1'] = 14
# r.loc[r.index == 313,'res_1'] = 12
# r.loc[r.index == 33,'res_1'] = 10
# r.loc[r.index == 928,'res_1'] = 10
# r.loc[r.index == 247,'res_1'] = 8.5
# r.loc[r.index == 972,'res_1'] = 7.5

r['res_1'].to_csv('../result/20180129_1.csv', float_format='%.3f', index=None,header=None)
#
feature_import.to_csv('../result/ff.csv')