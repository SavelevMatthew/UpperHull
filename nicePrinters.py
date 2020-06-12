from measure import Measurer
from multiprocessing import cpu_count


def info(msg):
    print('[INFO]: {}'.format(msg))


def make_report(counter, phi_np, phi_gd):
    print('=' * 64)
    info('Функция #{} (Потоков: {})'.format(counter, cpu_count()))
    info('Время полного перебора: {} секунд'.format(Measurer
                                                    .time_measures[0]
                                                    .total_seconds()))
    info('Время градиентного спуска: {} секунд'.format(Measurer
                                                       .time_measures[1]
                                                       .total_seconds()))
    max_phi = round(max(phi_np), 3)
    min_phi = round(min(phi_np), 3)
    size = max_phi - min_phi
    max_error = round(max([abs(phi_gd[j] - phi_np[j])
                           for j in range(len(phi_gd))]), 5)
    avg_error = round(sum([abs(phi_gd[j] - phi_np[j])
                           for j in range(len(phi_gd))])
                      / len(phi_gd), 5)
    max_error_percent = max_error / size * 100
    avg_error_percent = avg_error / size * 100
    info('Максимальная ошибка: {}'.format(max_error))
    info('Средняя ошибка: {}'.format(avg_error))
    info('Максимальная ошибка в %: {}'.format(round(max_error_percent, 2)))
    info('Средняя ошибка в %: {}'.format(round(avg_error_percent, 2)))
    counter += 1
    Measurer.time_measures.clear()
    print('=' * 64)
