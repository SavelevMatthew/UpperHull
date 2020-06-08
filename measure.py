from datetime import datetime
import inspect


class Measurer:
    time_measures = []

    @staticmethod
    def timer(f):
        def tmp(*args, **kwargs):
            start_time = datetime.now()
            result = f(*args, **kwargs)
            dt = datetime.now() - start_time
            dt_string = '{} секунд'.format(dt.total_seconds())
            f_name = inspect.stack()[-1].filename + ':' + f.__name__
            Measurer.time_measures.append(dt)
            print('Время выполнения функции {}: {}' .format(f_name,
                                                            dt_string))
            return result

        return tmp
