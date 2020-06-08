from datetime import datetime


class Measurer:
    time_measures = []
    result_measures = []

    @staticmethod
    def timer(f):
        def tmp(*args, **kwargs):
            start_time = datetime.now()
            result = f(*args, **kwargs)
            dt = datetime.now() - start_time
            dt_string = '{} секунд'.format(dt.total_seconds())
            print('Время выполнения функции {}: {}' .format(f.__name__,
                                                            dt_string))
            return result

        return tmp

    @staticmethod
    def clear():
        Measurer.time_measures.clear()
        Measurer.result_measures.clear()

    @staticmethod
    def store_time(f):
        def tmp(*args, **kwargs):
            start_time = datetime.now()
            result = f(*args, kwargs)
            Measurer.time_measures.append(datetime.now() - start_time)
            return result

        return tmp

    @staticmethod
    def store_results(f):
        def tmp(*args, **kwargs):
            result = f(*args, kwargs)
            Measurer.result_measures.append(result)
            return result

        return tmp
