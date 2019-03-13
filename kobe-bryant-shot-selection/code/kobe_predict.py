#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
pd.set_option('display.width',110) #console显示宽度
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, cross_val_score

if __name__=="__main__":
    # 读取原始数据
    train_data = pd.read_csv('../dataset/data.csv',na_values='N/A',header = 0)

    print(train_data.dtypes)

    # 绘制各因素下shot_made_flag的count
    
    # 根据loc_x、loc_y和lan、lat的同等，可以将一方删除
    # loc_x、loc_y、lan、lat -> 改用极坐标表示dist、angle
    train_data['dist'] = np.sqrt(train_data['loc_x']**2 + train_data['loc_y']**2)
    loc_x_zero = train_data['loc_x'] == 0
    train_data['angle'] = np.array([0]*len(train_data))
    train_data['angle'][~loc_x_zero] = np.arctan(train_data['loc_y'][~loc_x_zero] / train_data['loc_x'][~loc_x_zero])
    train_data['angle'][loc_x_zero] = np.pi / 2 

    # 根据shot_zone_area, shot_zone_basic, shot_zone_range画loc\
    # 发现都可以用dist和angle代替，故可以删除
    # minutes_remaining、seconds_remaining
    train_data['remaining_time'] = train_data['minutes_remaining'] * 60 + train_data['seconds_remaining']

    # season,取出其年份
    print(train_data['season'].unique())
    train_data['season'] = train_data['season'].apply(lambda x: int(x.split('-')[1]))
    print(train_data['season'].unique())

    # opponent、matchup   Only opponent is needed.


    # 删除不需要的列
    drops = ['team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
         'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date']
    for drop in drops:
        train_data = train_data.drop(drop, 1)

    print(train_data.info())
    # 哑编码(dummy encoding)
    categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
    for var in categorical_vars:
        train_data = pd.concat([train_data, pd.get_dummies(train_data[var], prefix=var)], 1)
        train_data = train_data.drop(var, 1)
    print(train_data.info())

    # 划分train和submission
    all_train = train_data[pd.notnull(train_data['shot_made_flag'])]
    all_train = all_train.drop(['shot_id'],1)
    all_submission = train_data[pd.isnull(train_data['shot_made_flag'])]
    all_submission = all_submission.drop('shot_made_flag', 1)
    submission = all_submission.drop(['shot_id'],1)
    print('len of train:',len(all_train))
    print('len of submission:',len(submission))
    print(submission.info())
    

    X = all_train.drop(['shot_made_flag'],1)
    y = all_train['shot_made_flag']


    
    # 调参（params tuning）
    seed = 7
    processors = 1
    num_folds = 3
    num_instances = len(X)
    scoring = 'log_loss'
    kfold = KFold(n=num_instances, n_folds = num_folds, random_state=seed)
    rf_grid = GridSearchCV(
        estimator = RandomForestClassifier(warm_start=True, random_state=seed),
        param_grid = {
            'n_estimators' : [100,200],
            'criterion' : ['gini', 'entropy'],
            'max_features' : [18,20],
            'max_depth' : [8,10],
            'bootstrap' : [True]
         },
         cv = kfold,
         scoring = scoring,
         n_jobs =processors)
    rf_grid.fit(X,y)
    print(rf_grid.best_score_)
    print(rf_grid.best_estimator_.best_params_)
    

    '''
    print('submission预测')
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf = clf.fit(X, y)
    y_predicted = clf.predict_proba(submission)
    print(y_predicted)
    '''

    '''
    print('交叉验证......')
    # 划分测试集与训练集，抽取10条数据作为测试集
    Xd_train, Xd_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=14)
    # 创建随机森林分类器
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    # 训练模型
    clf = clf.fit(Xd_train, y_train)
    # 对测试集进行预测
    y_predicted = clf.predict(Xd_test)
    print('len of test:',len(y_test))
    print(y_test.values)
    print(y_predicted)
    # 计算准确率
    accuracy = np.mean(y_test == y_predicted) * 100
    print ("accuracy:",accuracy,"%")
    '''

    '''
    # 对submission预测结果写入文件
    res = pd.DataFrame({
        "shot_id":all_submission.shot_id,
        "shot_made_flag":y_predicted[:,1]
        })
    res.to_csv('./mysubmission2.csv',index = False)
    '''
