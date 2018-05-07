import os
from dotenv import load_dotenv, find_dotenv
from requests import session
import logging


class TitanicDisaster(object):
    def __init__(self):
        self.kaggle_login_url = 'https://www.kaggle.com/account/login'
        self.payload = {
            'action': 'login',
            'username': os.environ.get("KAGGLE_USERNAME"),  # kaggle username
            'password': os.environ.get("KAGGLE_PASSWORD"),  # kaggle password
        }
        self.project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        self.log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=self.log_fmt)
        self.logger = logging.getLogger(__name__)

    def extract_data(self, url, file_path):
        with session() as c:
            c.post(self.kaggle_login_url, data=self.payload)
            with open(file_path, 'wb') as handle:
                response = c.get(url, stream=True)
                for block in response.iter_content(1024):
                    handle.write(block)

    def main(self):
        self.logger.info("getting raw data")
        train_url = "https://www.kaggle.com/c/3136/download/train.csv"
        test_url = "https://www.kaggle.com/c/3136/download/test.csv"

        raw_data_path = os.path.join(self.project_dir, 'data', 'raw')
        train_data_path = os.path.join(raw_data_path, 'train.csv')
        test_data_path = os.path.join(raw_data_path, 'test.csv')

        self.extract_data(train_url, train_data_path)
        self.extract_data(test_url, test_data_path)
        self.logger.info("downloaded raw training and test data")


if __name__ == "__main__":
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    titanic_obj = TitanicDisaster()
    titanic_obj.main()
