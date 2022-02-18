def ChatTextInfo(history, only_user_dialog=False):
    """Dialog text"""
    if only_user_dialog is True:
        text = '。'.join(item['msg'].strip() for item in history if item.get('msg', False) and
                        (item['name'].startswith('Visitor') or item['name'].startswith('火币用户')))
        return text
    else:
        text = '。'.join(item['msg'].strip() for item in history if item.get('msg', False) and
                        (not item['name'].startswith('火币牛牛') or not item['name'].startswith('在线客服')))
        # print(text)
        return text


def SessionInfo(session):
    """Session data"""
    data = {'country_code': session['country_code'],
            'city': session['city'],
            'browser': session['browser'],
            'country_name': session['country_name'],
            'region': session['region'],
            'platform': session['platform'],
            'start_date': session['start_date'],
            'end_date': session['end_date']}
    return data


def WebPathInfo(webpath):
    """"Webpath"""
    from_ = [path['from'] for path in webpath]
    to_ = [path['to'] for path in webpath]
    path = '->'.join(map(str, zip(from_, to_)))
    return path


def OtherInfo(chatHistory):
    """Other information"""
    data = {
        'Visitor': chatHistory['visitor']['name'],
        'Agent': '->'.join(chatHistory['agent_names']),
        'phone': chatHistory['visitor']['phone'],
        'email': chatHistory['visitor']['email'],
        'timestamp': chatHistory['timestamp'],
        'unread': chatHistory['unread'],
        'rating': chatHistory['rating'],
        'department_name': chatHistory['department_name'],
    }
    return data


def InformationCollect(chatHistory, only_user_dialog=False):
    """Get all necessary information"""
    dialogText = ChatTextInfo(chatHistory['history'], only_user_dialog=only_user_dialog)
    sessionData = SessionInfo(chatHistory['session'])
    webpath = WebPathInfo(chatHistory['webpath'])
    otherData = OtherInfo(chatHistory)
    otherData.update(sessionData)
    otherData.update({'dialogText': dialogText, 'webpath': webpath})
    return otherData


if __name__ == '__main__':
    import pymongo
    import pandas as pd

    results = []
    col = pymongo.MongoClient('localhost', 27017)['Zendesk']['ChatHistory']
    for item in col.find():
        res = InformationCollect(item)
    #     results.append(res)
    # df = pd.DataFrame(results)
    # print(df)
