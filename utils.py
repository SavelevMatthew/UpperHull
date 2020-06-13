from measure import Measurer
from multiprocessing import cpu_count
from datetime import datetime
import pandas as pd
import os


def info(msg):
    print('[INFO]: {}'.format(msg))


def make_report(name, counter, builder, phi_np, phi_gd):
    name = name.replace('\n', ' ').strip()
    print('=' * 64)
    info('Функция {} #{} (Потоков: {}, размерность: {}, '
         'точек: {})'.format(name, counter, cpu_count(), builder.dim + 1,
                            len(phi_np)))
    full_time = Measurer.time_measures[0].total_seconds()
    gd_time = Measurer.time_measures[1].total_seconds()
    info('Время полного перебора: {} секунд'.format(full_time))
    info('Время градиентного спуска: {} секунд'.format(gd_time))
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
    return ['\n{}\n'.format(name), builder.dim + 1, cpu_count(), len(phi_np),
            full_time, gd_time, max_error_percent, avg_error_percent]


def get_available_name():
    sheets = 1
    now = datetime.now()
    name = now.strftime('%Y-%m-%d')
    path = os.path.join(os.getcwd(), 'Report', '{}.xlsx'.format(name))
    if os.path.exists(path):
        xl = pd.ExcelFile(path)
        sheets = len(xl.sheet_names) + 1

    return path, 'Report{}'.format(sheets)

