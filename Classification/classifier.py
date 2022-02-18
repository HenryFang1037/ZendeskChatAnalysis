# -*- coding:utf-8 -*-
# Wrote by HenryFang, UEC Department, November/2021

import os
import time
import warnings
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import SelfTrainingClassifier

from Classification.tools import proprecessing
from DialogAnalysis.textFile import txt2csv, csvfile_collect
from DialogAnalysis.tools import parse_date

warnings.filterwarnings('ignore')


class SemiSupervisedClassifier():
    """
    Semi-supervised classifier help us to augment training dataset by increasing labeled data, which predicted by
    classifier model in each iteration; It could save a lot labeling work for us;

    """

    def __init__(self):
        self.sdg_params = dict(alpha=1e-5, penalty="l2", loss="log", class_weight='balanced')
        self.vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
        self.labelEncoder = LabelEncoder()
        self.selfTrainingPipelineModel = Pipeline(
            [
                ("vecter", CountVectorizer(**self.vectorizer_params)),
                ("tfidf", TfidfTransformer()),
                ("clf", SelfTrainingClassifier(SGDClassifier(**self.sdg_params), verbose=True)),
            ]
        )

    def fit(self, X, string_label):
        y = self.labelEncoder.fit_transform(string_label)
        idx = string_label.isna().to_numpy()
        y[idx] = -1
        self.selfTrainingPipelineModel.fit(X, y)

    def predict(self, X):
        return self.selfTrainingPipelineModel.predict(X)

    def predict_label(self, X):
        y = self.selfTrainingPipelineModel.predict(X)
        return self.labelEncoder.inverse_transform(y)

    def predict_prob(self, X):
        return self.selfTrainingPipelineModel.predict_proba(X)

    def save(self):
        joblib.dump(self.selfTrainingPipelineModel, 'SemiSupervisedClassifier.pkl')


class SupervisedClassifier():

    def __init__(self):
        self.sdg_params = dict(alpha=1e-5, penalty="l2", loss="log", class_weight='balanced')
        self.vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
        self.label_encoder = LabelEncoder()
        self.PipelineModel = Pipeline(
            [
                ("vecter", CountVectorizer(**self.vectorizer_params)),
                ("tfidf", TfidfTransformer()),
                ("clf", SGDClassifier(**self.sdg_params))
            ]
        )

    def fit(self, X, string_label):
        y = self.label_encoder.fit_transform(string_label)
        self.PipelineModel.fit(X, y)

    def predict(self, X):
        return self.PipelineModel.predict(X)

    def predict_label(self, X):
        y = self.PipelineModel.predict(X)
        return self.label_encoder.inverse_transform(y)

    def predict_prob(self, X):
        return self.PipelineModel.predict_proba(X)


def model_saver(model, modelType='supervised', colName='Section', sectionName=None, subsectionName=None):
    """
    Save model
    :param model: model
    :param modelType: type of the model, only support `supervised` and `semi-supervised` parameter
    :param colName: label column name
    :param sectionName: If colName is 'Subsection' or 'Reason', this parameter must be supplied correctly!
    :param subsectionName: If colName is 'Reason', this parameter must be supplied correctly!
    :return:
    """
    modelType = 'supervised' if modelType == 'supervised' else 'semi-supervised'
    date = datetime.now().strftime('%Y-%m-%d')
    path = "../Models/{modelType}/{date}".format(modelType=modelType, date=date)
    if not os.path.exists(path):
        os.makedirs(path)
    print(os.path.abspath(path))
    if colName == 'Section':
        joblib.dump(model,
                    path + '/{colName}_{modelType}Classifier.pkl'.format(colName=colName, modelType=modelType))
        print('{date} : {colName} classification model training has been completed !'.format(
            date=time.asctime(), colName=colName))
    elif colName == 'Subsection':
        joblib.dump(model,
                    path + '/{sectionName}_{modelType}Classifier.pkl'.format(sectionName=sectionName,
                                                                             modelType=modelType))
        print('{date} : {sectionName} classification model training has been completed !'.format(
            date=time.asctime(), sectionName=sectionName))
    elif colName == 'Reason':
        joblib.dump(model,
                    path + '/{sectionName}_{subsectionName}_{modelType}Classifier.pkl'.format(
                        sectionName=sectionName, subsectionName=subsectionName, modelType=modelType))
        print(
            '{date} : {sectionName}_{subsectionName} classification model training has been completed !'.format(
                date=time.asctime(), sectionName=sectionName, subsectionName=subsectionName))


def model_loader(modelPath):
    """
    Load model
    :param modelPath: model path
    :return:
    """
    model = joblib.load(modelPath)
    return model


def time_selector(df, start, end=None):
    """
    Time range selector
    :param df: Input data (Pandas DataFrame)
    :param start: Start date (string format)
    :param end: End date (string format)
    :return: df with certain time range
    """

    def parse_date(string):
        return datetime.strptime(string, '%Y-%m-%d').date()

    if end is None:
        try:
            start = parse_date(start)
        except:
            raise Exception('Start time format failed, please check start time')
        df = df[df['Date'] == start]
        return df
    else:
        try:
            start = parse_date(start)
            end = parse_date(end)
        except:
            raise Exception('Start/End time format failed, please check start/end time')
        df = df[(df['Date'] >= start) & (df['Date'] <= end)]
        return df


def train_model(trainFile, colName='Section', modelType='supervised', sectionName=None, subsectionName=None):
    """
    Single model training
    :param trainFile: Input file (Pandas DataFrame)
    :param colName: Label column name
    :param modelType: If only having small sample of training dataset, `modelType` could set up to `semi-supervised`,
        otherwise use `supervised`
    :param sectionName:  If colName is 'Subsection' or 'Reason', this parameter must be supplied correctly!
    :param subsectionName: If colName is 'Reason', this parameter must be supplied correctly!
    :return:
    """
    model = SupervisedClassifier() if modelType == 'supervised' else SemiSupervisedClassifier()

    if colName == 'Section':
        data = trainFile.dropna(
            subset=['Dialog_Extracted', colName]) if modelType == 'supervised' else trainFile.dropna(
            subset=['Dialog_Extracted'])
        X = data['Dialog_Extracted']
        label = data[colName]
    else:
        data = trainFile.dropna(
            subset=['Dialog_Extracted', 'Subsection', 'Reason']) if modelType == 'supervised' else trainFile.dropna(
            subset=['Dialog_Extracted', 'Subsection'])
        X = data['Dialog_Extracted']
        label = data.apply(
            lambda row: row['Subsection'] + '@' + row['Reason'] if row['Reason'] is not np.NaN else np.NaN, axis=1)

    model.fit(X, label)
    print('{date} : Saving model...'.format(date=time.asctime()))
    model_saver(model, colName=colName, modelType=modelType, sectionName=sectionName, subsectionName=subsectionName)


def train_predict(df_file, model_path=None, train_file=None, predict=True):
    if predict is False:
        print('{date} : Training model...'.format(date=time.asctime()))
        train_model(train_file)
    else:
        model = model_loader(model_path)
        classified_file = df_file.dropna(subset=['Dialog_Extracted'])
        X = classified_file['Dialog_Extracted']
        labels = model.predict_label(X)
        classified_file['Section'] = labels
        return classified_file


def train(df, colName='Section', sectionName=None, modelType='supervised'):
    """
    Classification model
    :param df: Input file (Pandas DataFrame)
    :param colName: Label column name
    :param sectionName:  If colName is 'Subsection' or 'Reason', this parameter must be supplied correctly!
    :param subsectionName: If colName is 'Reason', this parameter must be supplied correctly!
    :param modelType: If only having small sample of training dataset, `modelType` could set up to `semi-supervised`,
        otherwise use `supervised`
    :return:
    """
    if colName == 'Section':
        train_model(df, colName=colName, modelType=modelType, sectionName=None, subsectionName=None)
    elif colName == 'Subsection':
        if sectionName is None:
            raise ValueError("Section name doesn't provided")
        # data = df[df['Section'] == sectionName]
        train_model(df, colName=colName, modelType=modelType, sectionName=sectionName, subsectionName=None)
    # elif colName == 'Reason':
    #     if sectionName is None:
    #         raise ValueError("Section name doesn't provided")
    #     if subsectionName == None:
    #         raise ValueError("Subsection name doesn't provided")
    #     # data = df[df['Section'] == sectionName]
    #     # data = data[data['Subsection'] == subsectionName]
    #     train_model(df, colName=colName, modelType=modelType, sectionName=sectionName, subsectionName=subsectionName)
    else:
        raise ValueError("colName isn't provided correctly")


def train_model_with_opts(df, colName='Section', modelType='supervised',
                          train_opts='all'):
    """
    Classification model training func with training options
    :param df: Input data (Pandas DataFrame)
    :param colName: Label column name
    :param sectionName: If colName is 'Subsection' or 'Reason', this parameter must be supplied correctly!
    :param subsectionName: If colName is 'Reason', this parameter must be supplied correctly!
    :param modelType: If only having small sample of training dataset, `modelType` could set up to `semi-supervised`,
        otherwise use `supervised`
    :param train_opts: `all` means training all classification models; `section&subsection` means only training to
        classify column `Section` and column `Subsection`; `section` means only training to classify column `Section`;
        `subsection` means only training the given `Section` column , which means df[df['Section']==sectionName];
        `Reason` means training the given `Section` column and `Subsection` column, which means
        df[(df['Section']==sectionName)&(df['Subsection']==subsectionName)];

    :return:
    """
    if train_opts == 'all':
        train(df, colName=colName, modelType=modelType)
        for section in df['Section'].unique():
            if section == 'Other':
                continue
            data = df[df['Section'] == section]
            # print('Data size {}'.format(data.shape))
            train(data, colName='Subsection', sectionName=section, modelType=modelType)
            # for subsection in data['Subsection'].unique():
            #     sub_data = data[data['Subsection'] == subsection]
            #     train(sub_data, colName='Reason', sectionName=section, subsectionName=subsection, modelType=modelType)

    # elif train_opts == 'section&subsection':
    #     train(df, colName=colName, modelType=modelType)
    #     for section in df['Section'].unique():
    #         data = df[df['Section'] == section]
    #         train(data, colName='Subsection', sectionName=section, modelType=modelType)

    elif train_opts == 'section':
        train(df, modelType=modelType)

    # elif train_opts == 'subsection':
    #     train(df, colName='Subsection', sectionName=sectionName)
    #
    # elif train_opts == 'reason':
    #     train(df, colName='Reason', sectionName=sectionName, subsectionName=subsectionName)

    else:
        raise ValueError("Parameter train_opts error")


def predict(df, model_dict, model_name):
    X = df['Dialog_Extracted']
    model = model_dict[model_name]
    label = model.predict_label(X)
    return label


def predict_with_opts(row, model_mapper, predict_opts='all'):
    X = row['Dialog_Extracted']
    if predict_opts == 'all':
        sectionModel = model_mapper['Section']
        sectionLabel = sectionModel.predict_label([X])
        sectionLabel = sectionLabel[0].replace(' ', '_').strip()
        if sectionLabel == 'Other':
            return 'Other', 'Other', 'Other'
        subsectionModel = model_mapper[sectionLabel]
        subsectionLabel = subsectionModel.predict_label([X])
        subsectionLabel, reasonLabel = subsectionLabel[0].strip().split('@')
        return sectionLabel, subsectionLabel, reasonLabel
    # elif predict_opts == 'section&subsection':
    #     sectionModel = model_mapper['Section']
    #     sectionLabel = sectionModel.predict_label([X])
    #     sectionLabel = sectionLabel[0].replace(' ', '_').strip()
    #     subsectionModel = model_mapper[sectionLabel]
    #     subsectionLabel = subsectionModel.predict_label([X])
    #     return sectionLabel, subsectionLabel[0]
    elif predict_opts == 'section':
        sectionModel = model_mapper['Section']
        sectionLabel = sectionModel.predict_label([X])
        return sectionLabel[0]
    # elif predict_opts == 'subsection':
    #     sectionName = sectionName.replace(' ', '_').strip()
    #     subsectionModel = model_mapper[sectionName]
    #     subsectionLabel = subsectionModel.predict_label([X])
    #     return subsectionLabel[0]
    # elif predict_opts == 'reason':
    #     subsectionName = subsectionName.replace(' ', '_')
    #     reasonModel = model_mapper[subsectionName]
    #     reasonLabel = reasonModel.predict_label([X])
    #     return reasonLabel[0]


def modelMapper(dirs='/Users/fangzubing/ZendeskChat/Models/', modelType='supervised', default='latest'):
    if default == 'latest':
        path = os.path.join(dirs, modelType)
        file_dirs = os.listdir(path)
        file_dirs = list(filter(lambda x: os.path.isdir(os.path.join(path,x)), file_dirs))
        date = sorted(file_dirs, key=lambda x: datetime.strptime(x, '%Y-%m-%d').date(), reverse=True)[0]
        path = dirs + modelType + '/' + str(date)
    else:
        path = dirs + modelType + '/' + default
    models = os.listdir(path)
    model_path_mapper = {model.split('_')[0].replace(' ', '_').strip(): os.path.join(path, model) for model in models}
    model_mapper = {modelName: model_loader(modelPath) for modelName, modelPath in model_path_mapper.items()}
    return model_mapper


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # # Training dataset
    # df = pd.read_excel("/Users/fangzubing/downloads/TrainingData/trainingset.xlsx")
    # # Train model
    # # train_model_with_opts(df, train_opts='all', modelType='semi-supervised')
    # # Model loading
    # model_mapper = modelMapper(modelType='semi-supervised')
    # file = df.copy()
    # file['Label'] = file.apply(
    #     lambda row: predict_with_opts(row, model_mapper, predict_opts='all') if row[
    #                                                                                 'Dialog_Extracted'] is not np.NaN else (
    #         'Null', 'Null', 'Null'), axis=1)
    # labels = np.array([list(value) for value in file['Label'].values])
    # file['Section_1'], file['Subsection_1'], file['Reason_1'] = labels[:, 0], labels[:, 1], labels[:, 2]
    # file['Section_1'] = file['Section_1'].apply(lambda x: x.replace('_', ' '))
    # file.drop(columns=['Label'], inplace=True)
    # file.to_excel("/Users/fangzubing/downloads/TrainingData/classified_预测.xlsx")
    ## Prediction
    # for i, row in df.iterrows():
    #     labels = predict_with_opts(row, model_mapper, predict_opts='all')
    #     print(labels)

    # # ----------------------------------------------------------------------
    # # model_path = '/Users/henry/PycharmProjects/ZendeskChat/models/supervisedClassifier_2021-11-24.pkl'
    # # model_path = "/Users/fangzubing/ZendeskChat/Models/supervised/2021-12-07/Section_supervisedClassifier.pkl"
    dirs = '/Users/fangzubing/downloads/zendeskChat/2022-02-17/'
    # dirs = '/Users/fangzubing/downloads/dataset/2022-01-27~2022-02-02/negativeView'
    # # dirs = '/Users/fangzubing/downloads/dataset/2022-1-10_mala'
    start, end = '2022-02-17', None
    # 
    # # ----------------------------------------------------------------------
    # # Text file to CSV file
    txt2csv(dirs, only_user_dialog=True)
    df = csvfile_collect(dirs)
    df.dropna(subset=['Timestamp'], inplace=True)
    df['Date'] = df['Timestamp'].apply(parse_date)
    df = time_selector(df, start=start, end=end)
    path = os.path.join(dirs, 'df.csv')
    df.to_csv(path)
    # ----------------------------------------------------------------------
    # Dialog extracting
    classified_path = os.path.join(dirs, 'classified')
    if not os.path.exists(classified_path):
        os.mkdir(classified_path)
    # file = pd.read_csv(path)
    file = proprecessing(df, stop_words=False)
    # file.drop(columns=['Unnamed: 0'], inplace=True)

    # ----------------------------------------------------------------------
    # Classification
    # file = pd.read_excel('/Users/fangzubing/desktop/labeled_data_12-2~12-8.xlsx')
    print("{}: Labeling started...".format(time.asctime()))
    model_mapper = modelMapper(modelType='semi-supervised')
    file['Label'] = file.apply(
        lambda row: predict_with_opts(row, model_mapper, predict_opts='all') if row[
                                                                                    'Dialog_Extracted'] is not np.NaN else (
            'Null', 'Null', 'Null'), axis=1)
    labels = np.array([list(value) for value in file['Label'].values])
    file['Section'], file['Subsection'], file['Reason'] = labels[:, 0], labels[:, 1], labels[:, 2]
    file['Section'] = file['Section'].apply(lambda x: x.replace('_', ' '))
    file.drop(columns=['Label'], inplace=True)
    # file.to_excel('/Users/fangzubing/desktop/labeled_data_12-2~12-8.xlsx')
    file.to_excel(classified_path + '/labeledDialog.xlsx')
    file.groupby(['Date', 'country Name'])['Question'].count().reset_index().to_excel(
        classified_path + '/countryStat.xlsx')

    file.groupby(['Date', 'Section', 'Subsection', 'Reason'])['Question'].count().reset_index().to_excel(
        classified_path +
        '/countryQuestionStat.xlsx')
    print("{}: Labeling and statistic accomplished".format(time.asctime()))
  

    # ----------------------------------------------------------------------
    # classify
    ## file = pd.read_excel("/Users/henry/downloads/classificationData/trainset_dialog.xlsx")
    # classified_file = train_predict(file, model_path=model_path)
    ## classified_file.to_excel("/Users/henry/downloads/classificationData/trainset_dialog.xlsx")
    # ----------------------------------------------------------------------
    # train model
    # file = pd.read_excel("/Users/henry/desktop/classified.xlsx")
    # train_predict(df_file=None, train_file=file, predict=False)
    # ----------------------------------------------------------------------
    # classified_file.to_excel(classified_path + '/labeledDialog.xlsx')
    # file.groupby(['Date', 'country Name'])['Question'].count().reset_index().to_excel(
    #     classified_path + '/countryStat.xlsx')
    #
    # classified_file.groupby(['Date', 'Section'])['Question'].count().reset_index().to_excel(
    #     classified_path +
    #     '/countryQuestionStat.xlsx')
    # print("{}: Labeling and statistic accomplished".format(time.asctime()))
