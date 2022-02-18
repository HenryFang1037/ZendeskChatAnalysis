# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

from datetime import datetime

import pandas as pd

from DialogAnalysis.statisticAnalysis import subject_extract, section_analize


def complaint_conversation(df, subject='[投起]诉|公安|法院|警察|检察|举报', colname='Question'):
    """
    提取包含投诉、起诉、公安、法院、警察、检察、举报的会话
    :param df: 清洗后的会话（pandas.DataFrame格式）
    :param subject: 指定投诉相关主题词
    :param colname: 会话列名
    :return: 投诉相关会话
    """
    complaints = subject_extract(df, subject, colname=colname)
    return complaints


def complaints_dialog_extract(df, dirs):
    """
    保存Complaint 对话
    :param df: Complaint 对话文本
    :param dirs: 保存地址
    :return:
    """
    df = df[['Timestamp', 'Visitor Email', 'Question']]
    df['Dialog'] = df.apply(lambda row: ','.join(filter(lambda item: len(item.strip()) > 1, eval(row['Question']))),
                            axis=1)
    df[['Timestamp', 'Visitor Email', 'Dialog']].to_excel(
        dirs + '/投诉/{}.xls'.format(datetime.strftime(datetime.today(), '%Y-%m-%d')))


def calc_complain_ration(df, subject='[投起]诉|公安|法院|警察|检察|举报', colname='Question'):
    """
    计算投诉/咨询比例
    :param df: 清洗后的会话（pandas.DataFrame格式）
    :param subject: 指定投诉相关主题词
    :param colname: 会话列名
    :return: 投诉/咨询比例
    """
    complaints = complaint_conversation(df, subject=subject, colname=colname)
    complaint_ratio = pd.DataFrame({'投诉': [complaints.shape[0]], '咨询': [df.shape[0]]})
    return complaint_ratio


def complaint_theme_analysis(complaints, theme, colname='Question', stats=False):
    """
    投诉主题归类
    :param complaints: 投诉相关对话
    :param theme: 主题类别
    :param colname: 会话列名
    :param stats: 是否统计主题相关回话（默认：False)）
    :return:
    """
    complaints_theme_df = section_analize(complaints, sections=theme, colname=colname, stats=stats)
    return complaints_theme_df


def stats(df, colname='Question'):
    """
    按交易前中后、业务模块、次级业务统计问题数量
    :param df:
    :param colname:
    :return:
    """
    df = df[[colname, 'Flow', 'Section', 'Step']]
    df.rename(columns={colname: 'Count'}, inplace=True)
    res = df.groupby(['Flow', 'Section', 'Step'])['Count'].count().reset_index().sort_values(by=['Flow', 'Count'])
    return res


def parse_date(string):
    return datetime.strptime(string, '%Y-%m-%dT%H:%M:%SZ').date()
