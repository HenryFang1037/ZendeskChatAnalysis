# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

import re
import time
from functools import partial

import numpy as np
import pandas as pd

numPattern = "\d{3,}"
blockChainAddressPattern = "[a-z\d]{20,}"
emailPattern = "[a-z\d]+@[a-z]+\.[a-z]{,6}"
linkPattern = "((http[s]?|www)([:./]+)([a-z\d\-]+[\.\/\?\&\=]{0,2})+)"
dirPatten = "[/]{1,2}[a-z\-\.]+"
subPattern = re.compile(
    "{}|{}|{}|{}|{}".format(emailPattern, blockChainAddressPattern, numPattern, linkPattern, dirPatten), re.IGNORECASE)
splitPattern = re.compile("[\-\=\[\]\(\)\{\}\.\t\n\s*~?!#\/\/|$@\_\%]")
enPattern = re.compile('[a-z0-9\'\-]+')
englishPattern = re.compile('[a-z0-9\s\'\-’$£€¢¥]+')
negative_words_pattern = re.compile(
    r'(?:^|\b)((?<!or) not|don\'t|didn\'t|haven\'t|hasn\'t|couldn\'t|can\'t|failed|error|rejected)(?:$|\b)')
question_words_pattern = re.compile(r'(?:^|\b)(how|what|why|what\'s)(?:$|\b)')
stop_words_file = ''
stop_words_dict = {}


def subString(string):
    """
    Remove number, block chain address, user id, user email, file path from conversation
    :param string: original string
    :return: sub string
    """
    return subPattern.sub('', string)


def checkPicture(string):
    """
    Check chat string contains picture or not
    :param string:
    :return:
    """
    if string.endswith('.jpeg') or string.endswith('.jpg') \
            or string.endswith('.png') or string.endswith('.pdf'):
        return True
    return False


def loadStopWords(file_dir='/Users/henry/PycharmProjects/ZendeskChat/Classification/stopwords.txt'):
    """
    Load stop words
    :param file_dir: file path
    :return:
    """
    stop_words_dict = []
    with open(file_dir, 'r') as f:
        line = f.readline()
        while line:
            stop_words_dict.append(line.strip())
            line = f.readline()
    return set(stop_words_dict)


def chatClean(array):
    """
    Remove all number，user id, picture, block chain address, file address and email address
    :param array: chat dialog stored in array
    :param numPattern: Regex pattern for number
    :return: numpy.NaN or cleaned string
    """
    if array is np.NaN or len(array) == 0:
        return np.NaN

    sentence = [subString(i.strip()) for i in eval(array) if len(i.strip()) > 10
                and checkPicture(i.strip().lower()) is False]
    sentence = [i.strip() for i in sentence if len(i.strip()) != 0]
    if len(sentence) == 0:
        return np.NaN

    res = '. '.join(sentence)
    return res


def information_score(sentence, negative_words_weighted):
    """
    Sentence information score
    :param sentence: chat sentence
    :param negative_words_weighted:
    :return: score
    """

    def sentence_score(sentence_len):
        return np.log(sentence_len) / np.sqrt(sentence_len + 1)

    def emotion_score(sentence_len, num, point):
        return num * point * np.log(sentence_len) / np.sqrt(sentence_len)

    num = len(negative_words_pattern.findall(sentence)) if negative_words_weighted else len(
        question_words_pattern.findall(sentence))
    sentence_len = len(sentence.split())
    negative_words_point = 1.5
    question_words_point = 1.
    point = negative_words_point if negative_words_weighted is True else question_words_point
    return sentence_score(sentence_len) + emotion_score(sentence_len, num, point)


def find_top_n(scores, n):
    """
    Find top n sentence which has higher information score
    :param scores: all sentence scores
    :param n: num
    :return: top n scored sentence index
    """
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return sorted(idx)


def extractEnlishDialog(string, englishPattern, minimum_num=2, top_n_sentence=3, negative_words_weighted=True):
    """
    Only focus on english conversation
    :param string:
    :param englishPattern: Regex pattern for english sentence
    :param minimum_num: minimum words in one sentence
    :param first_n_sentences: choose the first n sentences
    :return:
    """

    def sentence_position_score(sentence_len):
        return 1 / np.log(np.arange(1, sentence_len + 1) + 1)

    if string is np.NaN or len(string.strip()) == 0:
        return np.NaN
    sentences = englishPattern.findall(string)
    sentences = list(filter(lambda sentence: len(sentence.split()) > minimum_num, sentences))
    if len(sentences) == 0:
        return np.NaN
    position_score = sentence_position_score(len(sentences))
    scores = list(map(partial(information_score, negative_words_weighted=negative_words_weighted), sentences))
    scores = np.array(scores) + position_score
    top_n = find_top_n(scores, n=top_n_sentence)

    return '.'.join([sentences[i] for i in top_n])


def extractEnglishWords(string, stop_words=False, stop_words_dict=None):
    """
    Only processing english chat
    :param string: original chat
    :param stop_words:
    :return:
    """
    if string is np.NaN or len(string) == 0:
        return np.NaN

    words = enPattern.findall(string)
    words = [word for word in words if not word.isdigit()]
    if len(words) <= 3:
        return np.NaN
    if stop_words is True:
        if stop_words_dict is None:
            raise ValueError('stop words dict is not supplied !')
        stop_words_removed = [word for word in words if word not in stop_words_dict]
        if len(stop_words_removed) == 0:
            return np.NaN
        return stop_words_removed
    return words


def tokenizer(string, stop_words=True):
    """
    Custom
    :param string:
    :param stop_words:
    :return:
    """
    words = splitPattern.split(string)
    if stop_words is True:
        res = [word for word in words if word != '' and word not in stop_words]
        return res
    return [word for word in words if word != '']


def proprecessing(df, stop_words=True,
                  stop_word_path='/Users/henry/PycharmProjects/ZendeskChat/Classification/stopwords.txt'):
    print("{}: English dialog extracting...".format(time.asctime()))
    data = df.dropna(subset=['Question'])
    data['Dialog'] = data['Question'].apply(lambda array: chatClean(array.lower()))
    data['Dialog_Extracted'] = data['Dialog'].apply(lambda string: extractEnlishDialog(string, englishPattern))
    if stop_words is True:
        print("{}: Words tokenizing".format(time.asctime()))
        stop_words_dict = loadStopWords(stop_word_path)
        data['Words'] = data['Dialog_Extracted'].apply(
            lambda sentence: extractEnglishWords(sentence, stop_words=stop_words,
                                                 stop_words_dict=stop_words_dict))
    print("{}: English dialog extraction has been finished !".format(time.asctime()))
    return data


if __name__ == '__main__':
    df = pd.read_csv('/Users/henry/downloads/df.csv')
    data = df.dropna(subset=['Question'])
    d = proprecessing(data)
    d[['Dialog', 'Dialog']].to_excel('/Users/henry/test.xls')
