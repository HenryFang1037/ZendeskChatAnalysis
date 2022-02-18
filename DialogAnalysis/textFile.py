# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

import os
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

numPattern = re.compile('\d+')


def info_filter(question_pattern, string, only_user_dialog):
    """
    从ZENDESK客服系统中导出的副本信息中提取所需信息
    :param question_pattern: 所需信息样式
    :only_user_dialog: 是否只提取用户对话
    :param string:
    :return:
    """
    if 'Timestamp:' in string:
        return {'Timestamp': string.split('Timestamp: ')[-1].strip()}
    elif 'Unread:' in string:
        return {'Unread': string.split('Unread:')[-1].strip()}
    elif 'Visitor Name:' in string:
        number = numPattern.findall(string)
        name = number[0] if len(number) else string.split('Visitor Name:')[-1].strip()
        return {"Visitor Name": name}
    elif 'Visitor Email:' in string:
        return {'Visitor Email': string.split('Visitor Email: ')[-1].strip()}
    elif 'Visitor ID:' in string:
        return {'Visitor Id': string.split('Visitor ID: ')[-1].strip()}
    elif 'Visitor Notes:' in string:
        lstring = string.split('Visitor Notes: ')[-1].strip()
        return {'Visitor Notes': lstring}
    elif 'Country Name:' in string:
        return {'country Name': string.split('Country Name: ')[-1].strip()}
    elif 'Region:' in string:
        return {'Region': string.split('Region: ')[-1].strip()}
    elif 'City' in string:
        return {'City': string.split('City: ')[-1].strip()}
    elif 'Platform:' in string:
        return {'Platform': string.split('Platform: ')[-1].strip()}
    elif 'Browser:' in string:
        return {'Browser': string.split('Browser: ')[-1].strip()}
    else:
        content = re.findall(question_pattern, string)
        if len(content) and not content[0].strip().startswith('火币牛牛') and not content[0].strip().startswith('在线客服'):
            if only_user_dialog:
                Question = content[0]
                if Question.strip().startswith('火币用户') or Question.strip().startswith('Visitor'):
                    return {'Question': Question.split(':')[-1]}
            else:
                return {'Question': content[0]}


def content_parse(filepath, only_user_dialog):
    """
    文件内容解析，获取用户对话记录
    :param filepath: text文件地址
    :only_user_dialog: 是否只提取用户对话
    :return:
    """
    contents = []
    # 只记录用户的对话
    question_pattern = re.compile("\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\)\s{1,2}(.*)")
    with open(filepath) as f:
        line = f.readline()
        news = []
        while line:
            if not line.startswith('=='):
                res = info_filter(question_pattern, line, only_user_dialog=only_user_dialog)
                if res is not None:
                    news.append(res)
            else:
                contents.append(news)
                news = []
            line = f.readline()
    return contents


def content_format(contents):
    """文件内容格式化,输出为格式化的Pandas DataFrame"""
    res = []
    for part in contents:
        dic = {'Question': []}
        for item in part:
            for key, val in item.items():
                if key not in dic:
                    dic[key] = val
                elif key == 'Question':
                    dic[key].append(val)
                else:
                    pass
        res.append(dic)
    return pd.DataFrame(res)


def question_clean(file):
    """用户问题清洗, 将空问题替换为numpy.nan"""
    question = file.apply(lambda row: np.nan if len(row['Question']) == 0 else row['Question'], axis=1)
    file['Question'] = question
    return file


def txt2csv(dirs, only_user_dialog):
    """text文件转csv"""
    files = os.listdir(dirs)
    num = len(list(filter(lambda x: x.endswith('.text'), files)))
    print("{}: Start collecting information from text file .".format(time.asctime()))
    for file in tqdm(files):
        if file.endswith('.text'):
            filepath = os.path.join(dirs, file)
            # txt文件解析
            contents = content_parse(filepath, only_user_dialog=only_user_dialog)
            # 生成csv文件
            file = content_format(contents)
            # 用户反馈问题清洗
            file = question_clean(file)
            # csv文件保存
            file.to_csv(filepath.replace('.text', '.csv'))
        else:
            continue
    print("{}: Total {} files have been completed !".format(time.asctime(), num))


def csvfile_collect(dirs):
    """
    收集csv文件
    :param dirs:
    :return:
    """
    cols = ['Question', 'Timestamp', 'Visitor Email', 'Visitor Notes', 'Visitor Id', 'Visitor Name',
            'country Name', 'Region', 'City', 'Platform', 'Browser', 'Unread']
    files = os.listdir(dirs)
    num = len(list(filter(lambda x: x.endswith('.text'), files)))
    df = pd.DataFrame()
    print("{}: Start collecting csv files .".format(time.asctime()))
    for file in tqdm(files):
        filepath = os.path.join(dirs, file)
        if filepath.endswith('.csv'):
            new_df = pd.read_csv(filepath, usecols=cols)
            df = df.append(new_df)
    df = df.drop_duplicates()
    print("{}: Total {} csv files have been collected !".format(time.asctime(), num))
    return df


if __name__ == '__main__':
    dirs = '/Users/henry/downloads/classificationData/data_06'
    # dirs = '/Users/fangzubing/downloads/customservice/no_3'
    txt2csv(dirs, only_user_dialog=True)
    df = csvfile_collect(dirs)
    # df['Date'] = df['Timestamp'].apply(lambda x: x.split('T')[0])
    # df = df.groupby(['Date', 'country Name'])['Region', 'City'].count().reset_index().rename(columns={'Region': 'Count'})[['Date', 'country Name', 'Count']]
    # df.to_csv('/Users/fangzubing/downloads/chat/{}.csv'.format(datetime.now().strftime('%Y-%m-%d')))
    df.to_csv(dirs + 'df.csv')
