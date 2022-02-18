import re
import jieba
import pandas as pd


def stop_words(stopwordspath):
    """
    Stop words
    :param stopwordspath: directory of stop words files
    :return:
    """
    with open(stopwordspath, 'r') as f:
        words = f.readlines()
    stopwords = {word.strip() for word in words}
    return stopwords


def user_words(userwordspath):
    """
    User defined words for special terms
    :param userwordspath: directory of udw file
    :return:
    """
    with open(userwordspath, 'r') as f:
        words = f.readlines()
    userwords = {word.strip() for word in words}
    return userwords


def fillMissingValue(df):
    """Fill missing values with 'Unknown' """
    df = df.applymap(lambda x: 'Unknown' if x is None or x == '' else x)
    return df


def remove_num_picture(string):
    """
    Remove numerical number and picture urls
    :param string:
    :return:
    """
    list_dialog = re.split('\n|。', string)
    return '。'.join(filter(lambda item: item.strip().isnumeric() is False and
                                        item.startswith('Visitor uploaded:') is False and
                                        item.startswith('URL:') is False and
                                        item.startswith('Type:') is False and
                                        item.startswith('Size:') is False, list_dialog))


def section_decomposition(sections):
    """
    Business Block decomposition
    :param blocks:
    :return:
    """
    res = []
    for step, step_values in sections.items():
        for block in step_values:
            for part, vals in block.items():
                if vals.__class__ is list:
                    for val in vals:
                        for sec_part, pattern in val.items():
                            res.append([step, part, sec_part, pattern])
                else:
                    res.append([step, part, part, vals])
    return res


def section_analize(data, sections, colname='dialogText'):
    """
    Classify dialog into business section problems
    :param data: pandas DataFrame filesc
    :param sections: classified problems
    :param colname: column name of dialog text
    :return:
    """
    d = pd.DataFrame()
    decompositions = section_decomposition(sections)

    for dec in decompositions:
        idx = data[colname].apply(
            lambda string: True if type(string) is str and re.search(dec[-1], string.lower()) else False)
        s = data[idx]
        s['Flow'], s['Module'], s['Topic'] = dec[0], dec[1], dec[2]
        d = d.append(s)
    d = d.reset_index()
    return d


def subject_extract(dataframe, subject, colname='dialogText'):
    """
    Extract the specific subject of dialog
    :param dataframe: pandas DataFrame file
    :param subject: specific subject(supporting Regx and terms)
    :return: dataframe
    """
    idx = dataframe[colname].str.contains(subject)
    idx = idx.fillna(False)
    question = dataframe[idx].reset_index()
    return question
