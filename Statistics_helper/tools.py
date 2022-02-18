import numpy as np
import pandas as pd
import os
import warnings


def count_by(data, by='topCountry', top='all', ascending=False):
    if by == 'topCountry':
        res = data.groupby(['country Name'])['Dialog_Extracted'].reset_index().rename(
        columns={'Extracted_Dialog': 'Count'}).sort_values(by='Count', ascending=ascending)
    elif by == 'topSection':
        res = data.groupby(['Section'])['Dialog_Extracted'].reset_index().rename(
            columns={'Extracted_Dialog': 'Count'}).sort_values(by='Count', ascending=ascending)
    elif by == 'topQuestion':
        res = data.groupby(['Reason'])['Dialog_Extracted'].reset_index().rename(
            columns={'Extracted_Dialog': 'Count'}).sort_values(by='Count', ascending=ascending)
    elif by == 'topSection&Question':
        res = data.groupby(['Section', 'Reason'])['Dialog_Extracted'].reset_index().rename(
            columns={'Extracted_Dialog': 'Count'}).sort_values(by='Count', ascending=ascending)
    else:
        raise Exception('')
    if isinstance(top, int):
        return res.head(top)
    else:
        res

