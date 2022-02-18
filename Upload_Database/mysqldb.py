# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

import time, os
import warnings
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import pymysql

warnings.filterwarnings('ignore')


def parse_datetime(string):
    return datetime.strptime(string, "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d %H:%M:%S')


@contextmanager
def mysql(host='localhost', user='root', password='2008', database='zendeskchat'):
    connection = pymysql.Connect(host=host, user=user, password=password, database=database)
    try:
        yield connection
    finally:
        connection.close()


def file_upload(df, database='zendeskchat', table='details'):
    with mysql(host='localhost', user='root', password='2008', database=database) as connection:
        cursor = connection.cursor()
        cols = "`,`".join([str(i) for i in df.columns.tolist()])
        for i, row in df.iterrows():
            sql = "INSERT INTO `{}` (`".format(table) + cols + "`) VALUES (" + "%s," * (len(row) - 1) + "%s)"
            cursor.execute(sql, tuple(row))
            connection.commit()
    print("{}: File has been uploaded in table {} .".format(time.asctime(), table))


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------
    # Labeled Dialog
    cols = ['Timestamp', 'Visitor Email', 'Visitor Name', 'country Name', 'Region', 'City', 'Dialog',
            'Dialog_Extracted', 'Section', 'Subsection', 'Reason']
    # file_dir = '2021-12-08'
    # -----------------------------------------------------------------------------------
    files = os.listdir('/Users/fangzubing/downloads/zendeskChat/')
    files = list(filter(lambda x: '.' not in x, files))
    df = pd.DataFrame()
    for file_dir in files:
        path = "/Users/fangzubing/downloads/zendeskChat/{}/classified/labeledDialog.xlsx".format(file_dir)
        d = pd.read_excel(path, usecols=cols)
        df = df.append(d)
    df['Timestamp'] = df['Timestamp'].apply(parse_datetime)
    df.fillna('Null', inplace=True)
    df['Visitor Email'] = df['Visitor Email'].apply(lambda x: x if len(x)<50 else x[-50:])
    # df.fillna('Null', inplace=True)
    file_upload(df)

    # -----------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------
    # Oversea growth data
    upload_num = 1
    # -----------------------------------------------------------------------------------
    # growth_file_path = "/Users/fangzubing/downloads/Growth/growth.xlsx"
    # growth = pd.read_excel(growth_file_path, usecols=['Date', 'Register Num.', 'Trading User Num.'])
    # growth['Date'] = growth['Date'].apply(lambda date: date.strftime('%Y-%m-%d'))
    # file_upload(growth.head(upload_num), database='growth', table='details')

