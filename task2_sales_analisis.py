#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas==2.1.4 seaborn==0.13.2 matplotlib==3.7.1')


# In[1]:


import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None


# In[3]:


PATH_BRAND = r"C:\Users\stepa\Downloads\test_brand_data.csv"


# In[5]:


if os.path.exists(PATH_BRAND):
    brand_df = pd.read_csv(PATH_BRAND)
else:
    print('Ошибка. Файл не найден')    


# In[8]:


brand_df['date'] = pd.to_datetime(brand_df['date'])

# Добавляем колонку с годом и месяцем
brand_df['year_month'] = brand_df['date'].dt.to_period('M')


# In[3]:


def get_your_brand(df):
    '''
    Функция просит ввести название интересующего
    бренда. Проверяет есть ли такой бренд в датафрейме
    и если есть то находит в каких регионах компания есть
    и за какие периоды данные.
    args:
        df (DataFrame): исходный датафрейм
    returns:
        your_brand (str): название бренда
        total_your_region (lst): список регионов
        total_period (lst): список периодов
    '''
    your_brand = str(input('Введите ваш бренд: '))
    while your_brand not in list(df['brand'].value_counts().index):
        your_brand = str(
            input('Такого бренда в данных нет. Введите ваш бренд: '))
    total_your_region = list(
        df[df['brand'] == your_brand]['region'].value_counts().index)
    total_period = list(df[df['brand'] == your_brand]
                        ['year_month'].astype('str').value_counts().index)
    return your_brand, total_your_region, total_period


# In[4]:


def get_sales_competitors(df, your_brand):
    '''
    Функция считает общие продажи по брендам и
    выводи ближайших 10 конкуретов
    args:
        df (DataFrame): исходный датафрейм
        your_brand (str): название бренда
    returns:
        nearnest_competitors (DataFrame): датафрейм с 
            ближайшими конкуретнами
    '''
    sales_competitors = (
        df.groupby('brand')
        .agg(total_sales=('total_sales', 'sum'),
             avg_itmes=('total_items_count', 'mean'),
             avg_receiptr=('total_receipts_count', 'mean'))
        .sort_values(by='total_sales', ascending=False)
        .reset_index()
    )

    index_your_brand = list(sales_competitors[sales_competitors['brand']
                                              == your_brand].index)
    print('Ваши ближайшие конкуренты')
    print(sep='')
    start = max(0, index_your_brand[0] - 5)
    end = min(len(sales_competitors), index_your_brand[0]+5)
    nearnest_competitors = sales_competitors.iloc[start:end]
    print(nearnest_competitors)
    return nearnest_competitors


# In[5]:


def get_plot_competitors(nearnest_competitors, your_brand):
    '''
    Функция строит график по общим продажам ближайших 
    конкурентов. Также на графике красным цветом выделяется
    интересующий бренд.
    args:
        nearnest_competitors (DataFrame): датафрейм с 
            ближайшими конкуретнами
        your_brand (str): название бренда
    returns:
        None
    '''

    brands = list(nearnest_competitors['brand'].values)
    sales = nearnest_competitors['total_sales'].values

    # Построение lineplot
    plt.figure(figsize=(10, 6))
    plt.plot(brands, sales, marker='o', linestyle='-', label='Продажи')

    target_index = brands.index(your_brand)  # Индекс бренда
    # Добавление подписей к маркерам
    for i, brand in enumerate(brands):
        plt.text(
            x=brands[i],  # Координата x
            y=sales[i],   # Координата y
            s=brand,      # Текст подписи
            fontsize=10,  # Размер текста
            ha='center',  # Горизонтальное выравнивание
            va='bottom'   # Вертикальное выравнивание
        )
        if i == target_index:
            plt.text(x=brands[i],
                     y=sales[i],
                     s=brand,
                     fontsize=10,
                     ha='center',
                     va='bottom',
                     color='red')

    # Настройка графика
    plt.title(f'Общие продажи {your_brand} и ближайших конкурентов')
    plt.xlabel('Бренд')
    plt.ylabel('Продажи')
    plt.xticks(rotation=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# In[6]:


def get_comparison_period(df, your_brand):
    '''
    Функция создает с общими продажами и с 
    разницей долей рынка между периодами
    args:
        df (DataFrame): исходный датафрейм
        your_brand (str): название бренда
    returns:
        comparison (DataFrame): итоговый датафрейм
    '''

    # Группировка данных по региону, бренду и периоду
    regional_sales = (
        df.groupby(['region', 'brand', 'year_month'])
        .agg(total_sales=('total_sales', 'sum'))
        .reset_index()
    )

    # Разделяем данные конкурентов и интересующего бренда
    your_sales = regional_sales[regional_sales['brand'] == your_brand]
    competitor_sales = regional_sales[regional_sales['brand'] != your_brand]

    # Суммируем продажи конкурентов по регионам и периоду
    competitor_sales = (
        competitor_sales.groupby(['region', 'year_month'])
        .agg(competitor_total_sales=('total_sales', 'sum'))
        .reset_index()
    )

    # Объединяем продажи интересующего бренда и конкурентов
    comparison = pd.merge(
        your_sales,
        competitor_sales,
        on=['region', 'year_month'],
        how='inner'
    )

    # Добавляем расчет доли рынка
    comparison['your_market_share'] = (
        comparison['total_sales'] /
        (comparison['total_sales'] + comparison['competitor_total_sales'])) * 100

    comparison['competitor_market_share'] = 100 - \
        comparison['your_market_share']

    # Считаем разницу в доле рынка между периодами
    comparison['market_share_diff'] = comparison.groupby(
        'region')['your_market_share'].diff()

    return comparison


# In[7]:


def get_plot_comparison(comparison):
    '''
    Функция строит график разницы по регионам 
    по конкретному бренду относитльно всех конкурентов
    в этом регионе
    args:
        comparison (DataFrame): датафрейм с данными
            о разнице в доле рынка
    returns:
        None
    '''

    # Готовим данные
    comparison_brand = comparison.groupby(['region', 'brand'])[
        'market_share_diff'].sum().reset_index()

    comparison_brand['colors'] = [
        'red' if x < 0 else 'green' for x in comparison_brand['market_share_diff']]
    comparison_brand.sort_values('market_share_diff', inplace=True)
    comparison_brand.reset_index(inplace=True)

    # Строим график
    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=comparison_brand.index, xmin=0,
               xmax=comparison_brand.market_share_diff)
    for x, y, tex in zip(comparison_brand.market_share_diff, comparison_brand.index, comparison_brand.market_share_diff):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'green', 'size': 14})

    # Добавляем описание
    plt.yticks(comparison_brand.index, comparison_brand.region, fontsize=12)
    plt.title('Разница по регионам относительно конкурентов', fontdict={
              'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()


# In[8]:


def get_imporatant_changes(df):
    '''
    Функция вычисляет процентное изменение продаж 
    для каждого бренда и региона. Затем определяет
    значительные изменения при пороге 50%
    args:
        df (DataFrame): исходный датафрейм
    returns:
        df (DataFrame): исходный датафрейм с
            новым столбцом
        lst_import (lst): список брендов с 
            важными изменениями 
    '''
    # Вычисляем процентное изменение продаж для каждого бренда и региона
    df = df.sort_values(by=['brand', 'date'])
    df['sales_change'] = df.groupby(
        ['brand'])['total_sales'].pct_change() * 100

    # Определяем значительные изменения (порог изменения: >50%)
    important_changes = df[abs(df['sales_change']) > 50]
    lst_import = list(
        important_changes['brand'].value_counts().index)

    return df, lst_import


# In[9]:


def get_important_plot(df, competitors_brand, your_brand):
    '''
    Функция строит график важных измений по динамике продаж
    отмечаем на графике места, в которых произошли эти важные
    изменения
    args:
        df (DataFrame): исходный датафрейм с
            новым столбцом
        competitors_brand (str): название бредна конкурента
        your_brand (str): название интересующего бренда
    returns:
        None
    '''

    brand_change = df.groupby(['date', 'brand']).agg(sales_change=(
        'sales_change', 'sum'), total_sales=('total_sales', 'sum')).reset_index()
    brand_data_comp = brand_change[brand_change['brand'] == competitors_brand]
    brand_data_your = brand_change[brand_change['brand'] == your_brand]

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(brand_data_comp['date'], brand_data_comp['total_sales'],
             marker='o', label=f'{competitors_brand}')
    plt.plot(brand_data_your['date'], brand_data_your['total_sales'],
             marker='o', label=f'{your_brand}')
    plt.title(
        f'Сравнение динамики продаж бренда {your_brand} с важными изменениями {competitors_brand}')
    plt.xlabel('Дата')
    plt.ylabel('Продажи')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Добавление аннотаций для значительных изменений
    for i, row in brand_data_comp.iterrows():
        if abs(row['sales_change']) > 50:  # Порог для аннотации
            plt.annotate(
                f"{row['sales_change']:.1f}%",
                xy=(row['date'], row['total_sales']),
                xytext=(row['date'], row['total_sales'] + 50),
                fontsize=9,
                color='red'
            )

    for i, row in brand_data_your.iterrows():
        if abs(row['sales_change']) > 50:  # Порог для аннотации
            plt.annotate(
                f"{row['sales_change']:.1f}%",
                xy=(row['date'], row['total_sales']),
                xytext=(row['date'], row['total_sales'] + 50),
                fontsize=9,
                color='red'
            )

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[10]:


def get_your_region(df, your_brand, total_your_region):
    '''
    Функция просит ввести название интересующего
    региона. Проверяет есть ли такой регион и если есть 
    то возвращает его.
    args:
        df (DataFrame): исходный датафрейм с
            новым столбцом
        your_brand (str): название интересующего бренда
        total_your_region (lst): список регионов, в которых
            прдеставлен бренд
    returns:
        your_region (str): название региона
    '''
    your_region = str(input('Введите интересующий регион: '))
    while your_region not in total_your_region:
        your_region = str(
            input('Данных за такой регион не обнаружено. Введите регион: '))
    return your_region


# In[11]:


def get_region_competitors(df, your_region, your_brand, total_your_region):
    '''
    Функция считает общие продажи по брендам и регоинам
    выводит результат для итересующего региона
    args:
        df (DataFrame): исходный датафрейм с
            новым столбцом
        your_region (str): название региона
        your_brand (str): название бренда
        total_your_region (lst): список регионов, в которых
            прдеставлен бренд
    returns:
        region_competitors (DataFrame): датафрейм с
            конкуретнами в конкретном регионе
    '''

    region_competitors = (
        df[df['region'].isin(total_your_region) == True]
        .groupby(['region', 'brand'])
        .agg(total_sales=('total_sales', 'sum'),
             avg_itmes=('total_items_count', 'mean'),
             avg_receiptr=('total_receipts_count', 'mean'))
        .sort_values(by='total_sales', ascending=False)
        .reset_index())

    print(f'Ваши конкуренты в регионе {your_region}')
    print(sep='')
    print(region_competitors[region_competitors['region'] == your_region])
    return region_competitors


# In[12]:


def get_plot_region(your_region, region_competitors, your_brand):
    '''
    Функция строит график общих продаж в конкретном регионе
    args:
        your_region (str): название региона
        region_competitors (DataFrame): датафрейм с
            конкуретнами в конкретном регионе
        your_brand (str): название бренда
    returns:
        None
    '''
    brands = list(
        region_competitors[region_competitors['region'] == your_region]['brand'].values)
    sales = region_competitors[region_competitors['region']
                               == your_region]['total_sales'].values

    # Построение lineplot
    plt.figure(figsize=(10, 6))
    plt.plot(brands, sales, marker='o', linestyle='-', label='Продажи')

    target_index = brands.index(your_brand)  # Индекс нашего бренда
    # Добавление подписей к маркерам
    for i, brand in enumerate(brands):
        plt.text(
            x=brands[i],  # Координата x
            y=sales[i],   # Координата y
            s=brand,      # Текст подписи
            fontsize=10,  # Размер текста
            ha='center',  # Горизонтальное выравнивание
            va='bottom'   # Вертикальное выравнивание
        )
        if i == target_index:
            plt.text(x=brands[i],
                     y=sales[i],
                     s=brand,
                     fontsize=10,
                     ha='center',
                     va='bottom',
                     color='red')

    # Настройка графика
    plt.title(f'Общие продажи в регионе {your_region}')
    plt.xlabel('Бренд')
    plt.ylabel('Продажи')
    plt.xticks(rotation=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# In[13]:


def get_imporatant_changes_region(df, your_region):
    '''
    Функция вычисляет процентное изменение продаж 
    для каждого бренда и региона. Затем определяет
    значительные изменения при пороге 50%
    args:
        df (DataFrame): исходный датафрейм с
            новым столбцом
        your_region (str): название региона
    returns:
        df (DataFrame): исходный датафрейм с
            обновленым новым столбцом
        lst_import (lst): список брендов с 
            важными изменениями в регионе
    '''
    # Вычисляем процентное изменение продаж для каждого бренда и региона
    df = df.sort_values(by=['brand', 'region', 'date'])
    df['sales_change'] = df.groupby(['brand', 'region'])[
        'total_sales'].pct_change() * 100

    # Определяем значительные изменения (порог изменения: >50%)
    important_changes = df[abs(df['sales_change']) > 50]
    lst_import = list(
        important_changes[important_changes['region'] == your_region]['brand'].value_counts().index)

    return df, lst_import


# In[14]:


def get_important_plot_region(df, competitors_brand, your_brand, your_region):
    '''
    Функция строит график важных изменений в динамике продаж
    в сравнение между двумя брендами в конкретном регионе
    args:
        df (DataFrame): исходный датафрейм с
            обновленым новым столбцом
        competitors_brand (str): название бредна конкурента
        your_brand (str): название интересующего бренда
        your_region (str): название региона
    returns:
        None
    '''

    brand_data_comp = df[(df['brand'] == competitors_brand)
                         & (df['region'] == your_region)]
    brand_data_your = df[(df['brand'] == your_brand) &
                         (df['region'] == your_region)]

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(brand_data_comp['date'], brand_data_comp['total_sales'],
             marker='o', label=f'{competitors_brand} - {your_region}')
    plt.plot(brand_data_your['date'], brand_data_your['total_sales'],
             marker='o', label=f'{your_brand} - {your_region}')
    plt.title(
        f'Сравнение динамики продаж бренда {your_brand} с важными изменениями {competitors_brand}')
    plt.xlabel('Дата')
    plt.ylabel('Продажи')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Добавление аннотаций для значительных изменений
    for i, row in brand_data_comp.iterrows():
        if abs(row['sales_change']) > 50:  # Порог для аннотации
            plt.annotate(
                f"{row['sales_change']:.1f}%",
                xy=(row['date'], row['total_sales']),
                xytext=(row['date'], row['total_sales'] + 50),
                fontsize=9,
                color='red'
            )

    for i, row in brand_data_your.iterrows():
        if abs(row['sales_change']) > 50:  # Порог для аннотации
            plt.annotate(
                f"{row['sales_change']:.1f}%",
                xy=(row['date'], row['total_sales']),
                xytext=(row['date'], row['total_sales'] + 50),
                fontsize=9,
                color='red'
            )

    plt.legend()
    plt.tight_layout()
    plt.show()


# In[15]:


def get_period(df, your_brand, total_period):
    '''
    Функция просит ввести интересующий период. Проверяет 
    есть ли такой период в датафрейме и выводит результат
    args:
        df (DataFrame): исходный датафрейм с
            обновленым новым столбцом
        your_brand (str): название интересующего бренда
        total_period (lst): список периодов
    returns:
        your_period (str): интересующий период
    '''
    your_period = str(
        input('Введите интересующий вас период в формате yyyy-mm: '))
    while your_period not in total_period:
        your_period = str(
            input('Данных за этот период нет. Введите другой период: '))
    return your_period


# In[16]:


def get_plot_comparison_period(comparison, period):
    '''
    Функция строит график разницы по регионам относительно
    конкурентов в этом регионе за определенный период
    args:
        comparison (DataFrame): датафрейм с данными
            о разнице в доле рынка
        period (str): интересующий период
    returns:
        None
    '''

    # Готовим данные
    comparison_period = comparison[comparison['year_month'] == period]

    comparison_period['colors'] = [
        'red' if x < 0 else 'green' for x in comparison_period['market_share_diff']]
    comparison_period.sort_values('market_share_diff', inplace=True)
    comparison_period.reset_index(inplace=True)

    # Строим график
    plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=comparison_period.index, xmin=0,
               xmax=comparison_period.market_share_diff)
    for x, y, tex in zip(comparison_period.market_share_diff, comparison_period.index, comparison_period.market_share_diff):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'green', 'size': 14})

    # Добавляем описание
    plt.yticks(comparison_period.index, comparison_period.region, fontsize=12)
    plt.title(f'Разница по регионам относительно конкурентов за {period} период', fontdict={
              'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()


# In[17]:


def main():
    '''
    Функция строит отчет по конкретному бренду, вызывая
    другие функция для построения таблиц и графиков. 
    args:
        None
    returns:
        None
    '''
    # Путь к csv-файлу
    PATH_BRAND = r"C:\Users\stepa\Downloads\test_brand_data.csv"
    
    # Читаем csv-файл
    if os.path.exists(PATH_BRAND):
        df = pd.read_csv(PATH_BRAND)
    else:
        print('Ошибка. Файл не найден')
        
    # Меняем тип данных столбца date
    df['date'] = pd.to_datetime(df['date'])
    # Добавляем колонку с годом и месяцем
    df['year_month'] = df['date'].dt.to_period('M')
    
    your_brand, total_your_region, total_period = get_your_brand(df)
    nearnest_competitors = get_sales_competitors(df, your_brand)

    print(
        f'Компания представлена в {len(set(total_your_region))} регионах')
    print(total_your_region)
    print(
        f'Данные о компании начинаются с {min(total_period)} по {max(total_period)}')

    get_plot_competitors(nearnest_competitors, your_brand)
    comparison = get_comparison_period(df, your_brand)
    get_plot_comparison(comparison)
    df_sales_change, lst_import = get_imporatant_changes(df)
    print(
        f'Наиболее важные изменения произошли у {len(lst_import)} брендах за все время')
    stop_comp = 'y'
    while stop_comp != 'n':
        competitors_brand = str(
            input('Введите название конкурента для сравнения: '))
        get_important_plot(df_sales_change, competitors_brand, your_brand)
        stop_comp = str(
            input('Хотите сравнение с другим конкурентом? y or n '))

    stop_region = str(input('Сделать отчет по конкретному региону? y or n: '))
    while stop_region != 'n':
        your_region = get_your_region(df, your_brand, total_your_region)
        region_competitors = get_region_competitors(
            df, your_region, your_brand, total_your_region)
        get_plot_region(your_region, region_competitors, your_brand)

        df_sales_change, lst_import = get_imporatant_changes_region(
            df, your_region)
        print(
            f'Наиболее важные изменения в регионе {your_region} произошли в {len(lst_import)} брендах')
        print(f'Наименование всех брендов {lst_import}')
        stop_comp = 'y'
        while stop_comp != 'n':
            competitors_brand = str(
                input('Введите название конкурента для сравнения: '))
            get_important_plot_region(
                df_sales_change, competitors_brand, your_brand, your_region)
            stop_comp = str(
                input('Хотите сравнение с другим конкурентом? y or n '))
        stop_region = str(input('Хотите отчет по-другому региону? y or n '))

    stop_period = str(input('Нужен отчет по регионам за период? y or n'))
    while stop_period != 'n':
        your_period = get_period(df, your_brand, total_period)
        get_plot_comparison_period(comparison, your_period)
        stop_period = str(input('Хотите отчет за другой период: y or n '))


# In[18]:


if __name__ == '__main__':
    main()

