from measure import Measurer
from multiprocessing import cpu_count
from datetime import datetime
from openpyxl import load_workbook
import pandas as pd
import os


def info(msg):
    print('[INFO]: {}'.format(msg))


def make_report(name, counter, builder, phi_np, phi_gd):
    name = name.replace('\n', ' ').strip()
    print('=' * 64)
    info('Функция {} #{} (Потоков: {}, размерность: {}, '
         'точек: {}, точность: {})'.format(name, counter, cpu_count(),
                                           builder.dim + 1, len(phi_np),
                                           builder.acc))
    full_time = round(Measurer.time_measures[0].total_seconds(), 2)
    gd_time = round(Measurer.time_measures[1].total_seconds(), 2)
    info('Время полного перебора: {} секунд'.format(full_time))
    info('Время градиентного спуска: {} секунд'.format(gd_time))
    max_error = round(max([abs(phi_gd[j] - phi_np[j])
                           for j in range(len(phi_gd))]), 5)
    avg_error = round(sum([abs(phi_gd[j] - phi_np[j])
                           for j in range(len(phi_gd))])
                      / len(phi_gd), 5)
    info('Максимальная ошибка: {}'.format(max_error))
    info('Средняя ошибка: {}'.format(avg_error))
    counter += 1
    Measurer.time_measures.clear()
    print('=' * 64)
    return ['\n{}\n'.format(name), builder.dim + 1, cpu_count(), len(phi_np),
            builder.acc,
            full_time, gd_time, max_error, avg_error]


def get_available_name():
    sheets = 1
    now = datetime.now()
    name = now.strftime('%Y-%m-%d')
    path = os.path.join(os.getcwd(), 'Report', '{}.xlsx'.format(name))
    if os.path.exists(path):
        xl = pd.ExcelFile(path)
        sheets = len(xl.sheet_names) + 1

    return path, 'Report{}'.format(sheets)


def write_statistics(reports):
    df = pd.DataFrame(reports, columns=['Function Name', 'Dimensions',
                                        'Threads', 'Dots amount',
                                        'Accuracy (% of dots used)', 'Alpha',
                                        'Full NP time (sec)', 'GD time (sec)',
                                        'Max error', 'Average Error'])
    path, sheet = get_available_name()
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        if os.path.exists(path):
            writer.book = load_workbook(path)
        df.to_excel(writer, index=True, sheet_name=sheet)
        worksheet = writer.sheets[sheet]
        worksheet.sheet_view.zoomScale = 50
        worksheet.column_dimensions['A'].width = 4
        worksheet.column_dimensions['B'].width = 32
        for d in ['C', 'D', 'E', 'F']:
            worksheet.column_dimensions[d].width = 16
        for d in ['G', 'H', 'I', 'J', 'K']:
            worksheet.column_dimensions[d].width = 24
        writer.save()
