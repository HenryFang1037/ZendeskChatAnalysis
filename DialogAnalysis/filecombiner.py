# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

def fileCombiner(emailFile, textFile):
    """
    将ceo邮箱导出的文本和客服zendesk中导出的文本合并
    :param emailFile: ceo邮件文本
    :param textFile: zendesk对话文本
    :return: 合并后的文本
    """
    emailFile.rename(columns={'提交时间': 'Timestamp', '联系方式': 'Visitor Email', '详细描述': 'Question'}, inplace=True)
    emailFile.dropna(subset=['Question'], inplace=True)
    emailFile = emailFile[['Timestamp', 'Visitor Email', 'Question']]
    emailFile['Question'] = emailFile['Question'].apply(lambda x: str([x]))
    return textFile.append(emailFile)
