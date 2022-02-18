# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

def stop_words(stopwordspath):
    """
    停用词加载
    :param stopwordspath: 停用次文件地址
    :return:
    """
    with open(stopwordspath, 'r') as f:
        words = f.readlines()
    stopwords = {word.strip() for word in words}
    return stopwords


def user_words(userwordspath):
    """
    用户自定义词汇加载
    :param userwordspath: 自定义词汇文件地址
    :return:
    """
    with open(userwordspath, 'r') as f:
        words = f.readlines()
    userwords = {word.strip() for word in words}
    return userwords


if __name__ == '__main__':
    stopwordspath = '/Users/fangzubing/Downloads/文本处理/stopwords'
    userwordspath = '/Users/fangzubing/Downloads/文本处理/specialterms'

    stopwords = stop_words(stopwordspath)
    userwords = user_words(userwordspath)
