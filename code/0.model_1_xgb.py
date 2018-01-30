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

# train['嗜碱细胞/嗜酸细胞'] = train['嗜碱细胞%'] % train['嗜酸细胞%']
# test['嗜碱细胞/嗜酸细胞'] = test['嗜碱细胞%'] % test['嗜酸细胞%']

# 甘油三酯 总胆固醇 低密度脂蛋白胆固醇

# 红细胞平均体积

train['血红蛋白/红细胞计数*红细胞平均血红蛋白浓度'] = train['血红蛋白'] / train['红细胞计数*红细胞平均血红蛋白浓度']
test['血红蛋白/红细胞计数*红细胞平均血红蛋白浓度'] = test['血红蛋白'] / test['红细胞计数*红细胞平均血红蛋白浓度']

# train['CRF'] = 1.86 * train['肌酐'] - 1.164 * train['年龄'] * (1-train['性别'] * 0.26)
# test['CRF'] = 1.86 * test['肌酐'] - 1.164 * test['年龄'] * (1-train['性别'] * 0.26)

print('train featurs',train.columns)

print(train.head())
print(test.head())






cv = KFold(n_splits=6,shuffle=True,random_state=42)
results = []
feature_import = pd.DataFrame()
sub_array = []
feature_import['col'] = list(train.columns)
# print(train.columns)
train = train.fillna(0)
test = test.fillna(0)
train = train.values
test = test.values
y_train = y_train.values

y_mean = np.mean(y_train)

model_xgb = xgb.XGBRegressor(
                            # colsample_bytree=0.4603,
                             gamma=0.0468,
                             learning_rate=0.05,
                             max_depth=5,
                             n_estimators=2200,
                             reg_alpha=0.4640,
                             reg_lambda=0.8571,
                             subsample=0.5213,
                             nthread = -1,
                            # base_score= y_mean
)

for model in [model_xgb]:
    for traincv, testcv in cv.split(train,y_train):
        m = model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv], y_train[testcv])],eval_metric='rmse',early_stopping_rounds=50)

        y_tmp = m.predict(train[testcv],ntree_limit=m.best_ntree_limit)
        res = mean_squared_error(y_train[testcv],(y_tmp) ) / 2
        results.append(res)
        # feature_import[res] = list(m)
        sub_array.append(m.predict(test,ntree_limit=m.best_ntree_limit))
    print("Results: " + str( np.array(results).mean()))

s = 0
for i in sub_array:
    s = s + i

r = pd.DataFrame()
r['res_1'] = list(s/6)

print(r.describe())

# r = r.reset_index()
#
# print(r)

# 分类结果 下限
#
# r.loc[r.index == 938,'res_1'] = 15
# r.loc[r.index == 313,'res_1'] = 12
# r.loc[r.index == 33,'res_1'] = 10
# r.loc[r.index == 928,'res_1'] = 10
# r.loc[r.index == 247,'res_1'] = 8.5
# r.loc[r.index == 972,'res_1'] = 7.9

r['res_1'].to_csv('../result/result_0129_1.csv', float_format='%.3f', index=None,header=None)
# print(r.describe())

# feature_import.to_csv('../result/x_ff.csv')