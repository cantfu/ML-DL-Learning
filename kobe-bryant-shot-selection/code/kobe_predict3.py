#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
pd.set_option('display.width',110) #console显示宽度


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
    # 


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

    
    print('交叉验证......')
    Xd_train, Xd_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=14)
    clf = RandomForestClassifier(max_depth=10, n_estimators=100,max_features = 80, random_state=7)
    clf = clf.fit(Xd_train, y_train)
    score = clf.score(Xd_test, y_test) #准确率
    print('模型准确率：', score)
