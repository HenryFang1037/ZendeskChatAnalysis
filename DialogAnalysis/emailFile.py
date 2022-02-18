# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

import pandas as pd


def emailFileReader(path):
    """
    CEO 邮件内容读取
    :param path: 文件路径
    :return: dataFrame
    """
    columns = ['反馈来源', '提交时间', 'UID', '联系方式', '问题分类', '详细描述', '图片', '标签']
    df = pd.read_csv(path, usecols=columns)
    return df
