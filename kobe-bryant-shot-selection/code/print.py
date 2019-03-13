#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
pd.set_option('display.width',110) #console显示宽度


if __name__=="__main__":
    # 读取原始数据
    data = pd.read_csv('../dataset/data.csv',na_values='N/A',header = 0)
    #print(data['shot_zone_area'].unique())
    #print(data['shot_zone_basic'].unique())
    #print(data['shot_zone_range'].unique())
    data = data.dropna()
    print('无空行的数据基本信息.....')
    print(data.head(2))
    print(data.info())
    shot_zone_area_mapping = {
        'Right Side(R)' : 0,
        'Left Side(L)' : 1,
        'Left Side Center(LC)' : 2,
        'Right Side Center(RC)' :3,
        'Center(C)' : 4,
        'Back Court(BC)' : 5
    }
    shot_zone_basic_mapping = {
        'Mid-Range' : 0,
        'Restricted Area': 1,
        'In The Paint (Non-RA)' : 2,
        'Above the Break 3' : 3,
        'Right Corner 3' : 4,
        'Backcourt' : 5,
        'Left Corner 3' : 6
    }
    shot_zone_range_mapping = {
        '16-24 ft.' :0,
        '8-16 ft.' : 1,
        'Less Than 8 ft.' : 2,
        '24+ ft.' : 3,
        'Back Court Shot' :4
    }
    