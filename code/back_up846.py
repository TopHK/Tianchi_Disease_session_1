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
test = pd.read_csv('../data/d_test_A_20180102.csv',encoding='gbk')

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
y_train = train['血糖']

sex_map = {'男': 1, '女': 0, '??': 0}
train.drop(['血糖'], axis=1, inplace=True)
train['性别'] = train['性别'].map(sex_map)
test['性别'] = test['性别'].map(sex_map)

train.drop(['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体','id','体检日期'],axis=1,inplace=True)
test.drop(['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体','id','体检日期'],axis=1,inplace=True)

# add feature 血小板计数
# train['尿酸'],l_tr = stats.boxcox(train['尿酸'])
# test['尿酸'],l_te = stats.boxcox(test['尿酸'])

# train[u'红白细胞和'] = train[u'白细胞计数'] + (train[u'红细胞计数'])
# test[u'红白细胞和'] = test[u'白细胞计数'] + (test[u'红细胞计数'])


# train[u'甘油三酯_高密度脂蛋白胆固醇'] = train[u'高密度脂蛋白胆固醇'] + (train[u'甘油三酯'])
# test[u'甘油三酯_高密度脂蛋白胆固醇'] = test[u'高密度脂蛋白胆固醇'] + (test[u'甘油三酯'])

# Results: 0.993177484089

# train[u'A/G'] = train[u'白蛋白'] / (train[u'*球蛋白'])
# test[u'A/G'] = test[u'白蛋白'] / (test[u'*球蛋白'])

# train[u'白蛋白比'] = train[u'白蛋白'] / (train[u'*总蛋白'] + 1)
# test[u'白蛋白比'] = test[u'白蛋白'] / (test[u'*总蛋白'] + 1)

# train[u'性别年龄组合'] = (train[u'性别'] + 1) * 100 + (train[u'年龄'])
# test[u'性别年龄组合'] = (test[u'性别'] + 1) * 100 + (test[u'年龄'])

# 0.91 甘油三酯 2.75
# train['is_danger_by_高密度'] = 0
# train.loc[train['高密度脂蛋白胆固醇']<0.91,'is_danger_by_高密度'] = 1
# train['is_danger_by_甘油三酯'] = 0
# train.loc[train['甘油三酯']>2.75,'is_danger_by_甘油三酯'] = 1
# train['is_danger'] = train['is_danger_by_甘油三酯'] + train['is_danger_by_高密度']
#
# test['is_danger_by_高密度'] = 0
# test.loc[test['高密度脂蛋白胆固醇']<0.91,'is_danger_by_高密度'] = 1
# test['is_danger_by_甘油三酯'] = 0
# test.loc[test['甘油三酯']>2.75,'is_danger_by_甘油三酯'] = 1
# test['is_danger'] = test['is_danger_by_甘油三酯'] + test['is_danger_by_高密度']






# train = train.fillna(0)
print('train featurs',train.columns)

n_folds = 5

# CV model
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
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

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
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
# averaged_models = AveragingModels(models = (GBoost,model_lgb ))
#
# score = rmsle_cv(averaged_models)
# print(score.mean(),score.std())

# from sklearn import cross_validation
# # GBoost model_lgb averaged_models
cv = KFold(n_splits=5,shuffle=True,random_state=42)
#
results = []
feature_import = pd.DataFrame()
sub_array = []
feature_import['col'] = list(train.columns)
# print(train.columns)
train = train.values
test = test.values
y_train = y_train.values
# model cv
for model in [model_lgb]:
    for traincv, testcv in cv.split(train,y_train):

        # m = model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv],y_train[testcv])])
        m = model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv],y_train[testcv])],eval_metric='mse',early_stopping_rounds=50)
        y_tmp = m.predict(train[testcv])

        print(m.feature_importances_)
        # y_tmp = m.predict(train[testcv])
        res = mean_squared_error(y_train[testcv],y_tmp) / 2
        feature_import[res] = list(m.feature_importances_)
        results.append(res)
        # sub_array.append(m.predict(test))
        sub_array.append(m.predict(test))
    print("Results: " + str( np.array(results).mean() ))

# print(np.array(sub_array))
s = 0
for i in sub_array:
    s = s + i

r = pd.DataFrame()
r['res'] = list(s/5)
# print(r)
print(r.max(),r.min(),r.mean())

r['res'].to_csv('../result/result_0108_3.csv', float_format='%.3f', index=None,header=None)

feature_import.to_csv('../result/ff.csv')