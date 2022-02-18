import requests
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from DbHelper import MongodbHelper
import concurrent.futures

mongo = MongodbHelper(host='localhost', port=27017, database='Zendesk')
save = partial(mongo.save, collection_name='ChatHistory')


class ZendeskChatHistory():
    """
    Download Zendesk chat history data to local database
    """

    def __init__(self, token):
        """
        :param token:  secret token
        """
        self.__token = token
        self.__headers = {'Authorization': "Bearer {}".format(self.__token)}
        self.base_url = "https://www.zopim.com"
        self.chat_route = '/api/v2/chats/search'
        # self.increment_route = '/api/v2/incremental/chats?start'
        self.department_route = '/api/v2/departments'
        self.timeout = 10
        self.session = requests.Session()

    def getALlDepartment(self):
        """
        Get all departments names
        :return: department names
        """
        results = requests.get(self.base_url + self.department_route, headers=self.__headers)
        if results.status_code != 200:
            return results.status_code
        return results.json()


    def getChatHistory(self, start_time, end_time):
        """
        Download and save chat detail
        :param start_time: Start time and end time for download chat history
        :param end_time:
        :return:
        """
        func = partial(self.session.get, headers=self.__headers, timeout=self.timeout)

        query = urlencode({'q': {'timestamp': '[{} To {}]'.format(start_time, end_time)}})
        results = func(self.base_url + self.chat_route, data=query)
        if results.status_code != 200:
            return results.status_code
        json_data = results.json()
        # Next page url
        next_url = json_data.get('next_url', False)
        # Get detail chat url
        chat_detail_urls = [chat['url'] for chat in json_data['results']]
        while next_url and len(chat_detail_urls):
            print(next_url)
            # Load detail
            with ThreadPoolExecutor(max_workers=2) as executor:
                # chat_details = executor.map(func, chat_detail_urls)
                future_url = {executor.submit(func, url): url for url in chat_detail_urls}
                for future in concurrent.futures.as_completed(future_url):
                    try:
                        mongo.save(future.result().json())
                    except Exception as e:
                        print(e)



            # # Save detail
            # for chat in chat_details:
            #     print(chat.json())
            #     mongo.save('ChatHistory', chat.json())

            if next_url is not False:
                # Sleep 15 second for not meet the Zendesk API request limit (200/minute)
                time.sleep(15)
                results = func(next_url)
                if results.status_code != 200:
                    return results.status_code
                json_data = results.json()
                next_url = json_data.get('next_url', False)
                chat_detail_urls = [chat['url'] for chat in json_data['results']]


if __name__ == '__main__':
    token = "4sAYNNsERmBkcb3YafMTHz0tKGtGsA4Y85v6yDEhJCTCGjUJV4DD3lnJyCJaSpBk"
    chat = ZendeskChatHistory(token=token)
    chat.getChatHistory('2021-12-28', '2021-12-28')
    # res = chat.getALlDepartment()
    # print(res)
