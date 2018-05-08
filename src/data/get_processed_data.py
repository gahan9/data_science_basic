import numpy as np
import pandas as pd
import os
import logging

__author__ = "Gahan Saraiya"

from src.data.get_raw_data import TitanicDisaster


class GenerateDataSet(TitanicDisaster):
    def __init__(self):
        super(GenerateDataSet, self).__init__()
        self.df = self.read_data()
        self.logger.info("data frame constructed")
        # set the path of the processed data
        self.processed_data_path = os.path.join(self.project_dir, 'data', 'processed')
        self.write_train_path = os.path.join(self.processed_data_path, 'train.csv')
        self.write_test_path = os.path.join(self.processed_data_path, 'test.csv')

    def read_data(self):
        """
        read raw data and return pandas data frame
        :return: data frame
        """
        self.logger.info("reading raw data")
        train_df = pd.read_csv(self.train_data_path, index_col='PassengerId')
        test_df = pd.read_csv(self.test_data_path, index_col='PassengerId')
        test_df['Survived'] = -888  # set non existing column in test data frame
        self.logger.info("constructing data frame")
        df = pd.concat((train_df, test_df), axis=0)  # concat data frame
        return df

    @staticmethod
    def get_title(name):
        title_group = {
            'mr'          : 'Mr',
            'mrs'         : 'Mrs',
            'miss'        : 'Miss',
            'master'      : 'Master',
            'don'         : 'Sir',
            'rev'         : 'Sir',
            'mlle'        : 'Miss',
            'mme'         : 'Mrs',
            'major'       : 'Officer',
            'lady'        : 'Lady',
            'capt'        : 'Officer',
            'col'         : 'Officer',
            'dona'        : 'Lady',
            'dr'          : 'Officer',
            'jonkheer'    : 'Sir',
            'ms'          : 'Ms',
            'sir'         : 'Sir',
            'the countess': 'Lady'
        }
        f_name = name.split(',')[1]
        title = f_name.split('.')[0].strip().lower()
        return title_group[title]

    def fill_missing_values(self, df):
        self.logger.info("filling missing values")
        # embarked
        df.Embarked.fillna('C', inplace=True)
        # fare
        median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
        df.Fare.fillna(median_fare, inplace=True)
        # age
        title_age_median = df.groupby('Title').Age.transform('median')
        df.Age.fillna(title_age_median, inplace=True)
        self.logger.info("Missing value for embarked, fare and age are filled")
        return df

    @staticmethod
    def get_deck(cabin):
        return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

    def reorder_columns(self, df):
        self.logger.info("reordering column")
        columns = ['Survived'] + [col for col in df.columns if col != 'Survived']
        return df[columns]

    def process_data(self):
        """
        process data frame
        :return:
        """
        self.logger.info("Processing data")
        # using method chaining
        return (self.df
                # create title attribute - then add this
                .assign(Title=lambda x: x.Name.map(self.get_title))
                # working missing values - start with this
                .pipe(self.fill_missing_values)
                # create fare bin feature
                .assign(Fare_Bin=lambda x: pd.qcut(x.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']))
                # create age state
                .assign(AgeState=lambda x: np.where(x.Age >= 18, 'Adult', 'Child'))
                .assign(FamilySize=lambda x: x.Parch + x.SibSp + 1)
                .assign(IsMother=lambda x: np.where(((x.Sex == 'female') & (x.Parch > 0) & (x.Age > 18) & (x.Title != 'Miss')), 1, 0))
                # create deck feature
                .assign(Cabin=lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin))
                .assign(Deck=lambda x: x.Cabin.map(self.get_deck))
                # feature encoding
                .assign(IsMale=lambda x: np.where(x.Sex == 'male', 1, 0))
                .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
                # add code to drop unnecessary columns
                .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1)
                # reorder columns
                .pipe(self.reorder_columns)
                )

    def write_data(self, df):
        self.logger.info("writing data to csv")
        # train data
        df[df.Survived != -888].to_csv(self.write_train_path)
        self.logger.info("train data saved at : {}".format(self.write_train_path))
        # test data
        columns = [col for col in df.columns if col != "Survived"]
        df[df.Survived == -888][columns].to_csv(self.write_test_path)
        self.logger.info("test data saved at : {}".format(self.write_test_path))


if __name__ == "__main__":
    data_set_obj = GenerateDataSet()
    df = data_set_obj.process_data()
    data_set_obj.write_data(df)
