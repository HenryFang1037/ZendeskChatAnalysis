# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021
# This is a sample Python script.

from datetime import datetime

from DialogAnalysis.stopwords import stop_words, user_words
from DialogAnalysis.textFile import csvfile_collect
from DialogAnalysis.tokenizer import get_token, get_ngram

if __name__ == '__main__':
    txtdirs = '/Users/fangzubing/Downloads/dialog'
    stopwordspath = '/Users/fangzubing/Downloads/文本处理/stopwords'
    userwordspath = '/Users/fangzubing/Downloads/文本处理/specialterms'
    # 加载停用词和用户自定义词
    stopwords = stop_words(stopwordspath)
    userwords = user_words(userwordspath)
    # # 将txt文件整理保存为csv文件
    # txt2csv(txtdirs)
    # # 收集csv文件
    df = csvfile_collect(txtdirs)
    # 将对话分词
    df = get_token(df, stopwords, userwords, colname='Question')
    # 对分词做Ngram
    df = get_ngram(df, colname='Tokens')
    df.to_csv(txtdirs + '/all_{}.csv'.format(datetime.now().strftime('%Y_%m_%d')))
