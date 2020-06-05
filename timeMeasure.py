import time
from datetime import datetime


def my_timer(f):
    def tmp(*args, **kwargs):
        start_time = datetime.now()
        result = f(*args, **kwargs)
        dt = datetime.now() - start_time
        dt_string = '{} секунд'.format(dt.total_seconds())
        print('Время выполнения функции {}: {}' .format(f.__name__, dt_string))
        return result

    return tmp
