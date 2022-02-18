from datetime import datetime


def parse_string_timestamp(string):
    return datetime.strptime(string, '%Y-%m-%dT%H:%M:%SZ')
def parse_date(string):
    return datetime.strptime(string, '%Y-%m-%d')

def parse_string_date(string):
    return parse_date(parse_string_timestamp(string).date().strftime('%Y-%m-%d'))