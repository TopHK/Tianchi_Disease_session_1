# coding:utf-8

# 引入需要的包
# import keras
# from keras.models import Sequential
# from keras.layers import Dense

import numpy as np
import pandas as pd

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

# 读取数据

train = pd.read_csv('../data/d_train_20180102.csv',encoding='gbk')
test = pd.read_csv('../data/d_test_A_20180102.csv',encoding='gbk')

print('train shape',train.shape)
print('test shape',test.shape)

train_ID = train['id']
test_ID = test['id']



print('train feature shape',train.shape)
print('test feature shape',test.shape)

# 引入模型
from sklearn.model_selection import KFold, cross_val_score, train_test_split,cross_val_predict
from sklearn.metrics import mean_squared_error

train = train[train['年龄'] >= 16]
train = train[train['血糖'] <= 18]

print(train.shape)

y_train = train['血糖']

sex_map = {'男': 0, '女': 1, '??': 0}
train.drop(['血糖'], axis=1, inplace=True)
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

# train['红细胞平均血红蛋白量*红细胞平均血红蛋白浓度'] = train['红细胞平均血红蛋白量'] * train['红细胞平均血红蛋白浓度']
# test['红细胞平均血红蛋白量*红细胞平均血红蛋白浓度'] = test['红细胞平均血红蛋白量'] * test['红细胞平均血红蛋白浓度']
#
train['嗜酸细胞'] = train['白细胞计数'] * train['嗜酸细胞%']
test['嗜酸细胞'] = test['白细胞计数'] * test['嗜酸细胞%']

# 甘油三酯 总胆固醇 低密度脂蛋白胆固醇

# 红细胞平均体积
# train['血红蛋白/红细胞计数*红细胞平均血红蛋白浓度'] = train['血红蛋白'] / train['红细胞计数*红细胞平均血红蛋白浓度']
# test['血红蛋白/红细胞计数*红细胞平均血红蛋白浓度'] = test['血红蛋白'] / test['红细胞计数*红细胞平均血红蛋白浓度']

train['CRF'] = 1.86 * train['肌酐'] - 1.164 * train['年龄'] * (1-train['性别'] * 0.26)
test['CRF'] = 1.86 * test['肌酐'] - 1.164 * test['年龄'] * (1-train['性别'] * 0.26)

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

from sklearn.preprocessing import MinMaxScaler

# mm_train = MinMaxScaler()
train = train.values
# train = mm_train.fit_transform(train)
# mm_test = MinMaxScaler()
test = test.values
# test = mm_test.fit_transform(test)

# mm_y = MinMaxScaler()
y_train = y_train.values
# y_train = mm_y.fit_transform(y_train)

y_mean = pd.np.mean(y_train)

# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 6,
    'subsample': 0.90,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1
}
import xgboost as xgb
from catboost import Pool, CatBoostRegressor
model = CatBoostRegressor(rsm=0.9, depth=5, learning_rate=0.05, eval_metric='RMSE')


for model in [model]:
    for traincv, testcv in cv.split(train,y_train):
        model.fit(train[traincv], y_train[traincv],eval_set=[train[testcv],y_train[testcv]])
        y_tmp = model.predict(train[testcv])
        res = mean_squared_error(y_train[testcv],y_tmp ) / 2
        results.append(res)
        # feature_import[res] = list(m)
        sub_array.append(np.array(model.predict(test)))
    print("Results: " + str( np.array(results).mean()))

s = 0
for i in sub_array:
    s = s + i

s = np.array(s).ravel()

r = pd.DataFrame()
r['res_1'] = list(s/6)

print(r.describe())

r['res_1'].to_csv('../result/result_0121_1.csv', float_format='%.3f', index=None,header=None)

# feature_import.to_csv('../result/x_ff.csv')