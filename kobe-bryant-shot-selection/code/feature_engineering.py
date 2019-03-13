#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc
import pandas as pd
import numpy as np
pd.set_option('display.width',110) #console显示宽度
from sklearn.preprocessing import Imputer

plt.rcParams['font.sans-serif'] = ['KaiTi'] #显示中文
plt.rcParams['axes.unicode_minus'] = False #显示正常负号


def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages):
    
    fig, h = plt.subplots();
    for i, (mean, covarianceMatrix) in enumerate(zip(gaussianMixtureModel.means_, gaussianMixtureModel.covariances_)):
        # get the eigen vectors and eigen values of the covariance matrix
        v, w = np.linalg.eigh(covarianceMatrix)
        v = 2.5*np.sqrt(v) # go to units of standard deviation instead of variance
        
        # calculate the ellipse angle and two axis length and draw it
        u = w[0] / np.linalg.norm(w[0])    
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        currEllipse = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=ellipseColors[i])
        currEllipse.set_alpha(0.5)
        h.add_artist(currEllipse)
        h.text(mean[0]+7, mean[1]-1, ellipseTextMessages[i], fontsize=13, color='blue')

# combined_shot_type、season、period对shot_made_flag计数
def count_1(data):
    f, axarr = plt.subplots(3)
    sns.countplot(x="combined_shot_type", hue="shot_made_flag", data=data, ax=axarr[0])
    sns.countplot(x="season", hue="shot_made_flag", data=data, ax=axarr[1])
    sns.countplot(x="period", hue="shot_made_flag", data=data, ax=axarr[2])
    axarr[0].set_title('Combined shot type')
    axarr[1].set_title('Season')
    axarr[2].set_title('Period')
    plt.savefig('./feature_imgs/count-of-combined_shot_type、season、period.png')

# playoffs、shot_type对shot_made_flag计数
def count_2(data):
    f, axarr = plt.subplots(1, 2)
    sns.countplot(x="playoffs", hue="shot_made_flag", data=data, ax=axarr[0])
    sns.countplot(x="shot_type", hue="shot_made_flag", data=data, ax=axarr[1])
    axarr[0].set_title('Playoffs')
    axarr[1].set_title('Shot Type')
    plt.savefig('./feature_imgs/count-of-playoffs、shot_type.png')

# shot_zone_area、shot_zone_basic、shot_zone_range特征对shot_made_flag计数
def count_3(data):
    f, axarr = plt.subplots(3)
    sns.countplot(x="shot_zone_area", hue="shot_made_flag", data=data, ax=axarr[0])
    sns.countplot(x="shot_zone_basic", hue="shot_made_flag", data=data, ax=axarr[1])
    sns.countplot(x="shot_zone_range", hue="shot_made_flag", data=data, ax=axarr[2])
    axarr[0].set_title('Shot Zone Area')
    axarr[1].set_title('Shot Zone Basic')
    axarr[2].set_title('Shot Zone Range')
    plt.savefig('./feature_imgs/投篮区域计数.png')

# 绘制投篮点位 loc_x、loc_y、lat、lon
def draw_loc(train_data):
    plt.figure()
    plt.subplot(121)
    plt.scatter(train_data.loc_x, train_data.loc_y, s=2, alpha=0.02, c=(train_data['shot_made_flag'] ))#== 0))#'yellow' if (train_data['shot_made_flag'] == 0) else 'red')
    plt.xlabel('loc_x')
    plt.ylabel('loc_y')
    plt.title('kobe投篮点位')
    #plt.show()
    plt.subplot(122)
    plt.scatter(train_data.lon, train_data.lat, alpha=0.02,c=(train_data['shot_made_flag']))
    plt.xlabel('lon(纬度)')
    plt.ylabel('lat(经度)')
    plt.title('kobe投篮点位')
    plt.savefig('./feature_imgs/投篮点位.png')

# 投篮方式及命中率
def draw_shot_types(train_data):
    # 投篮方式
    plt.figure();
    shot_type = train_data['combined_shot_type'].value_counts() #投篮方式次数
    a = np.array([1,2,3,4,5,6])
    #sns.set_style('whitegrid')
    plt.bar(a,shot_type,align='center')
    plt.xlabel('进攻方式')
    plt.ylabel('进攻次数')
    plt.title('kobe的进攻')
    plt.grid(linestyle = '--', linewidth=1)
    plt.ylim(0,20000)
    plt.xticks(a,('跳投','带球上篮','扣篮','补篮','勾手补篮','擦板'))
    plt.savefig('./feature_imgs/投篮方式.png')
    #plt.show()
    # 计算命中率
    plt.figure()
    hits = train_data[train_data['shot_made_flag'] == 1]['combined_shot_type'].value_counts() #命中的行
    hits = hits/shot_type
    hits_plot = hits.plot.bar(rot=-20)
    hits_plot.set_ylabel('命中率')
    hits_plot.set_title('kobe投篮方式命中率')
    plt.savefig('./feature_imgs/投篮方式的命中率.png')

# 根据shot_zone_area, shot_zone_basic, shot_zone_range画location
def draw_shot_loc_by_area(train_data):
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
    plt.figure()
    plt.subplot(131)
    plt.scatter(train_data.loc_x, train_data.loc_y, s=2, alpha=0.2, c=(train_data['shot_zone_area'].map(shot_zone_area_mapping) ))#== 0))#'yellow' if (train_data['shot_made_flag'] == 0) else 'red')
    plt.title('shot_zone_area')

    plt.subplot(132)
    plt.scatter(train_data.loc_x, train_data.loc_y, s=2, alpha=0.2, c=(train_data['shot_zone_basic'].map(shot_zone_basic_mapping) ))#== 0))#'yellow' if (train_data['shot_made_flag'] == 0) else 'red')
    plt.title('shot_zone_basic')

    plt.subplot(133)
    plt.scatter(train_data.loc_x, train_data.loc_y, s=2, alpha=0.2, c=(train_data['shot_zone_range'].map(shot_zone_range_mapping) ))#== 0))#'yellow' if (train_data['shot_made_flag'] == 0) else 'red')
    plt.title('shot_zone_range')
    plt.savefig('./feature_imgs/根据区域画投篮点位.png')
if __name__=="__main__":
    # 读取原始数据
    data = pd.read_csv('../dataset/data.csv',na_values='N/A',header = 0)
    #print(data.head(5))
    #print(data.describe())
    #print(data.info())
    #print(train_data.info())
    #print(train_data['shot_zone_area'].unique())
    #print(train_data['shot_zone_basic'].unique())


    # shot_zone_basic、shot_zone_range特征对shot_made_flag计数
    #count_1(data)
    #count_2(data)
    #count_3(data)
    #plt.show()

    # 画出科比投篮点 loc_x、loc_y、lat、lon
    #draw_loc(data)
    #draw_shot_loc_by_area(data)

    # 投篮方式
    #draw_shot_types(data)
    #
    
    #dist和shot_distant呈正相关
    plt.figure(figsize=(5,5))
    plt.scatter(data.dist, data.shot_distance, color='blue')
    plt.title('dist and shot_distance')


    # 画球场
    plt.figure();
    plt.rcParams['figure.figsize'] = (13, 10)
    plt.rcParams['font.size'] = 15
    draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('shot attempts')
    plt.savefig('./feature_imgs/basketball_court.png')
    plt.show()

    '''
    # 画球场
    plt.figure();
    #plt.rcParams['figure.figsize'] = (13, 10)
    plt.rcParams['font.size'] = 15

    #ellipseTextMessages = [str(100*gaussianMixtureModel.weights_[x])[:4]+'%' for x in range(numGaussians)]
    #ellipseColors = ['red','green','purple','cyan','magenta','yellow','blue','orange','silver','maroon','lime','olive','brown','darkblue']
    #Draw2DGaussians(gaussianMixtureModel, ellipseColors, ellipseTextMessages)
    draw_court(outer_lines=True); plt.ylim(-60,440); plt.xlim(270,-270); plt.title('shot attempts')
    plt.show()
    '''
