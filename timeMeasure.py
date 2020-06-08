from datetime import datetime


class Measurer:
    measures = []

    @staticmethod
    def my_timer(f):
        def tmp(*args, **kwargs):
            start_time = datetime.now()
            result = f(*args, **kwargs)
            dt = datetime.now() - start_time
            dt_string = '{} секунд'.format(dt.total_seconds())
            Measurer.measures.append(dt)
            print('Время выполнения функции {}: {}' .format(f.__name__,
                                                            dt_string))
            return result

        return tmp

    @staticmethod
    def clear():
        Measurer.measures.clear()