import pandas as pd
import datetime
import numpy as np
from TextPreprocess import remove_num_picture


class NotSupportedType(Exception):
    pass


def timeSpan(row):
    """Calculate the session time span"""
    delta = datetime.datetime.strptime(row['end_date'], '%Y-%m-%dT%H:%M:%SZ') - datetime.datetime.strptime(
        row['start_date'], '%Y-%m-%dT%H:%M:%SZ')
    return delta.total_seconds()


def clockTime(row, by='hour'):
    time = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
    if by is 'hour':
        return time.hour
    elif by is 'day':
        return time.day
    elif by is 'month':
        return time.month
    else:
        raise NotSupportedType("Only support hour, day, month")


def filePrepocess(df):
    """
    1.Prepocess the dialog text for removing the numeric number and picture ulr
    2.Calc the time span of each conversation
    """
    df['timeSpan'] = df.apply(timeSpan, axis=1)
    df['cleanedDialog'] = df.apply(lambda row: remove_num_picture(row['dialogText']), axis=1)
    return df


def topicRatio(question, section):
    """Calculate the topic and business section consult times"""
    question['index'] = question['Flow']
    section['index'] = section['Module']
    ratio = question.merge(section, on='index', how='left')
    ratio = ratio[['Flow_y', 'Module_y', 'Module_x', 'Topic', 'Visitor_x', 'Visitor_y']]
    ratio.rename(columns={'Flow_y': 'Flow', 'Module_y': 'Module', 'Module_x': 'Submodule',
                          'Visitor_x': 'TopicCount', 'Visitor_y': 'ModuleCount'}, inplace=True)
    return ratio


def timeSpanStats(timeSpan_df):
    """Statistics of conversation time span excluding extreme values"""
    q25, q75 = np.percentile(timeSpan_df['timespan'], 25), np.percentile(timeSpan_df['timespan'], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    outliers_removed = [x for x in timeSpan_df['timespan'] if x >= lower and x <= upper]
    return pd.DataFrame(columns=['timeSpan'], data=outliers_removed)


def consulatTimeDestribution(df, by='hour'):
    """Calculate the consult time distribution"""
    df[by] = df.apply(clockTime, axis=1)
    return df.groupby([by])[by].count().reset_index()


def deviceDestribution(df):
    """Device destribution"""
    return df.groupby('platform')['Visitor'].count().reset_index().rename(columns={'Visitor': 'Count'})


def countryDestribution(df):
    """Country destribution"""
    return df.groupby('country_name')['Visitor'].count().reset_index().rename(columns={'Visitor': 'Count'})


def ratingStats(df):
    """Rating statistics"""
    return df.groupby('rating')['Visitor'].count().reset_index().rename(columns={'Visitor': 'Count'})


def unreadStats(df):
    "Unread statistics"
    return df.groupby('unread')['Visitor'].count().reset_index().rename(columns={'Visitor': 'Count'})


def departmentStats(df):
    """Department statistics"""
    df = df.groupby('department_name')['Visitor'].count().reset_index().rename(
        columns={'department_name': 'department_topic', 'Visitor': 'Count'})
    df['department_topic'] = df['department_topic'].apply(lambda x: x.split('、')[-1].strip() if x != 'Unknown' else x)
    return df
