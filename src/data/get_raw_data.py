import os
from dotenv import load_dotenv, find_dotenv
from requests import session
import logging

__author__ = "Gahan Saraiya"
__all__ = ['TitanicDisaster']


class TitanicDisaster(object):
    def __init__(self):
        self.kaggle_login_url = 'https://www.kaggle.com/account/login'
        self.payload = {
            'action': 'login',
            'username': os.environ.get("KAGGLE_USERNAME"),  # kaggle username
            'password': os.environ.get("KAGGLE_PASSWORD"),  # kaggle password
        }
        self.project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        # set the path of the raw data
        self.raw_data_path = os.path.join(self.project_dir, 'data', 'raw')
        self.train_data_path = os.path.join(self.raw_data_path, 'train.csv')
        self.test_data_path = os.path.join(self.raw_data_path, 'test.csv')
        self.log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=self.log_fmt)
        self.logger = logging.getLogger(__name__)

    def extract_data(self, url, file_path):
        with session() as c:  # start session
            c.post(self.kaggle_login_url, data=self.payload)  # login to kaggle with credential
            with open(file_path, 'wb') as handle:  # open file in write-bytes mode
                response = c.get(url, stream=True)  # get content as stream
                for block in response.iter_content(1024):  # iterate over content
                    handle.write(block)  # write bytes to file

    def get_raw_data(self):
        self.logger.info("getting raw data")
        # urls to get data
        train_url = "https://www.kaggle.com/c/3136/download/train.csv"
        test_url = "https://www.kaggle.com/c/3136/download/test.csv"

        # store data from url
        self.extract_data(train_url, self.train_data_path)
        self.extract_data(test_url, self.test_data_path)
        self.logger.info("downloaded raw training and test data")
        return self.train_data_path, self.test_data_path  # return data paths


if __name__ == "__main__":
    dotenv_path = find_dotenv()  # find '.env'
    load_dotenv(dotenv_path)  # load '.env'

    titanic_obj = TitanicDisaster()
    train_data, test_data = titanic_obj.get_raw_data()
    titanic_obj
