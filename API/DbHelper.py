import pymongo
import pymysql
import pandas as pd


class MongodbHelper:
    def __init__(self, host, port, database, username=None, password=None):
        self.host = host
        self.port = port
        self.__database = database
        self.__username = username
        self.__password = password
        self.db = pymongo.MongoClient(host=self.host, port=self.port)[self.__database]

    def save(self, collection_name, data):
        col = self.db[collection_name]
        col.update_one({'id': data['id']}, {'$set': data}, upsert=True)

    def load(self, collection_name, starttime, endtime):
        col = self.db[collection_name]
        collection = pd.DataFrame(list(col.find({'timestamp': 1}, {'$gte': starttime, '$lt': endtime})))
        return collection

    def exist(self, collection_name, id):
        col = self.db[collection_name]
        res = col.find_one({'id': id})
        if res:
            return True
        return False


class MySQLdbHelper:
    def __init__(self, host, port, database_name, username, password):
        self.host = host
        self.port = port
        self.database_name = database_name
        self.__username = username
        self.__password = password
        self.cursor = pymysql.Connection(host=self.host, port=self.port,
                                         user=self.__username, password=self.__password)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


if __name__ == '__main__':
    mongo = MongodbHelper(host='localhost', port=27017, database='Zendesk')
    print(mongo.exist(collection_name='ChatHistory', id='2112.2147184.SssXcjh2qlaEc'))