# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

import functools
import itertools
import re

import numpy as np
import pkuseg


def tokenizer(list_string, pattern, pku, stopwords, isCeoEmail=False):
    """
    分词
    :param list_string: 输入
    :param pattern: 正则表达
    :param pku: 分词器北大分词
    :param stopwords: 停用词
    :return:
    """
    # 同义词替换
    dic = {
        '人工客服': '客服',
        '人工服务': '客服',
        '在线客服': '客服',
        '在线人工': '客服',
        '人工': '客服',
        '账号': '账户',
        '未到账': '没到账',
        '卖家': '商家',
        '购买': '买币',
        '充值': '充币',
        '认证': '实名认证',
        '身份认证': '实名认证',
        '身份': '实名认证',
        '实名': '实名认证',
        '号码': '手机',
    }
    if list_string is np.nan:
        return np.nan
    if isCeoEmail is True:
        list_string = str([list_string])
    dialogs = itertools.chain.from_iterable(pattern.findall(dialog) for dialog in eval(list_string))
    tokens = [
        [dic.get(word.lower(), word.lower()) for word in pku.cut(dialog) if len(word) > 1 and word not in stopwords] for
        dialog in
        dialogs]
    tokens = list(filter(lambda token: True if len(token) else False, tokens))
    tokens = [[re.sub('人工客服|人工服务|人工', '客服', tok) for tok in token] for token in tokens]
    return tokens


def build_ngram(words, n=2):
    """
    构建ngram
    :param words: 词
    :param n: ngram词数，默认为2
    :return: ngram
    """
    res = []
    # ngram 的 n 大于词组长度时返回所有词
    if n >= len(words):
        return [''.join(words)]
    for i in range(len(words) - n + 1):
        res.append(''.join(words[i:i + n]))
    return res


def get_token(dataframe, stopwords, userwords, colname='Question', isCeoEmail=False):
    """
    对话分词
    :param dataframe: 输入的DataFrame
    :param stopwords: 停用词
    :param userwords: 自定义词
    :param colname: 需要做分词的列名
    :param isCeoEmail: 是否为Ceo邮箱导出的文件（Ceo邮箱文件表头与Zendesk导出的表头不一样)
    :return: 增加了Tokens列的DataFrame
    """
    # 只要中文和英文
    pattern = re.compile(u"([\u4e00-\u9fffa-zA-Z]+)")
    pku = pkuseg.pkuseg(user_dict=userwords)
    func = functools.partial(tokenizer, pattern=pattern, pku=pku,
                             stopwords=stopwords, isCeoEmail=isCeoEmail)
    dataframe['Tokens'] = dataframe[colname].apply(func)
    print('Tokenizer completed !')
    return dataframe


def get_ngram(dataframe, colname='Tokens'):
    """
    对分词做Ngram
    :param dataframe: 输入的DataFrame
    :param colname: Tokens列名，默认为'Tokens'
    :return: 增加了Ngram列的DataFrame
    """
    dataframe['Ngram'] = dataframe[colname].apply(lambda row: [build_ngram(words) for words in row]
    if row is not np.nan else np.nan)
    print('Ngram completed !')
    return dataframe
