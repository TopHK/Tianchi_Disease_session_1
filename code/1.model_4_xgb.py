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
# 引入模型
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from scipy.special import boxcox, inv_boxcox


# 读取数据
def get_date():
    train = pd.read_csv('../data/d_train_20180102.csv',encoding='gbk')
    test = pd.read_csv('../data/d_test_A_20180102.csv',encoding='gbk')

    print('train shape',train.shape)
    print('test shape',test.shape)

    train_ID = train['id']
    test_ID = test['id']

    print('train feature shape',train.shape)
    print('test feature shape',test.shape)

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

    train.drop(['嗜碱细胞%', '单核细胞%', '白球比例', '白蛋白', '*总蛋白', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', 'id', '体检日期'],
               axis=1, inplace=True)
    test.drop(['嗜碱细胞%', '单核细胞%', '白球比例', '白蛋白', '*总蛋白', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', 'id', '体检日期'],
              axis=1, inplace=True)

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

    # train['血小板比积*血小板平均体积'] = train['血小板比积'] / train['血小板平均体积']
    # test['血小板比积*血小板平均体积'] = test['血小板比积'] / test['血小板平均体积']


    print('train featurs',train.columns)

    train = train.fillna(0)
    test = test.fillna(0)

    # from sklearn.preprocessing import PolynomialFeatures
    # poly=PolynomialFeatures(degree=2,interaction_only=True)   #构造二次多项式特征对象
    # train = poly.fit_transform(train)
    # test = poly.fit_transform(test)

    train = train.values
    test = test.values
    print(train.shape)
    print(test.shape)
    return train,test,y_train,test_ID


if __name__ == '__main__':
    np.random.seed(2017)

    train, test, y_train, test_ID = get_date()
    #
    # function_set = ['add', 'sub', 'mul', 'div',
    #                 'sqrt', 'log', 'abs', 'neg', 'inv',
    #                 'max', 'min']

    from gplearn.genetic import SymbolicTransformer,SymbolicRegressor

    function_set = ['add', 'sub', 'mul', 'div',
                    'sqrt', 'log', 'abs', 'neg', 'inv',
                    'max', 'min']
    gp = SymbolicTransformer(generations=100, population_size=2000,
                             hall_of_fame=100, n_components=10,
                             function_set=function_set,
                             parsimony_coefficient=0.0005,
                             max_samples=0.9, verbose=1,
                             random_state=0, n_jobs=3)

    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=8,
                                  learning_rate=0.05, n_estimators=1000,
                                  max_bin=20, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)


    cv = KFold(n_splits=6,shuffle=True,random_state=42)
    results = []
    feature_import = pd.DataFrame()
    sub_array = []
    # feature_import['name'] = train.columns

    y_train = y_train.values

    y_mean = np.mean(y_train)

    for model in [model_lgb]:
        for traincv, testcv in cv.split(train,y_train):
            gp.fit(train[traincv], y_train[traincv])

            gp_features = gp.transform(train)
            print(gp_features)
            train = np.hstack((train, gp_features))

            m = model.fit(train[traincv], y_train[traincv],eval_set=[(train[testcv], y_train[testcv])],early_stopping_rounds=150)

            y_tmp = m.predict(train[testcv],num_iteration=m.best_iteration)
            res = mean_squared_error(y_train[testcv],(y_tmp) ) / 2
            results.append(res)

            t_gp_features = gp.transform(test)
            print(t_gp_features)
            test = np.hstack((test, t_gp_features))

            # feature_import[res] = list(m.feature_importances_)
            sub_array.append(m.predict(test,num_iteration=m.best_iteration))

        print("Results: " + str( np.array(results).mean()))

    s = 0
    for i in sub_array:
        s = s + i

    r = pd.DataFrame()
    r['res_1'] = list(s/6)

    print(r.describe())
    #
    r['res_1'].to_csv('../result/result_0122_gp_1.csv', float_format='%.3f', index=None,header=None)
    #
    # feature_import.to_csv('../result/x_ff.csv')


