#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas==2.1.4 numpy==1.24.3 spacy==3.5.3')


# In[1]:


import warnings
import pandas as pd
import numpy as np
import json
import spacy


# In[2]:


warnings.simplefilter('ignore', UserWarning)


# In[3]:


# Загрузка модели spaCy
nlp = spacy.load("ru_core_news_md")


# In[4]:


def load_entities(file_path):
    '''
    Загружает сущности из CSV-файла.
    args: 
        file_path (str): Путь к файлу CSV с сущностями.
    returns: 
        Список сущностей в виде словарей.
    '''
    entities_df = pd.read_csv(file_path, names=['id', 'name', 'description'])
    return entities_df.to_dict(orient="records")


# In[5]:


def load_texts(file_path):
    '''
    Загружает тексты из CSV-файла.
    args: 
        file_path (str): Путь к файлу CSV с текстами.
    returns: 
        DataFrame с текстами.
    '''
    texts_df = pd.read_csv(file_path, header=None, names=['text'])
    texts_df["id"] = range(1, len(texts_df) + 1)  # Автоматически добавляем ID
    return texts_df


# In[6]:


def classify_text(text, entities):
    '''
    Классифицирует сущности по тексту.
    args: 
        text (str): Текст для анализа.
        entities (lst): Список сущностей.
    returns: 
        Словарь с id основных и упомянутых сущностей.
    '''
    doc = nlp(text.lower())
    primary = set()
    mentions = set()

    for entity in entities:
        entity_id = entity["id"]
        name = entity["name"].lower()
        description = entity["description"].lower()

        # Основные сущности: точное совпадение имени в NER
        for ent in doc.ents:
            if ent.text.lower() == name:
                primary.add(entity_id)

        # Упоминания: совпадение с описанием через токены NER
        for ent in doc.ents:
            if ent.text.lower() in description:
                mentions.add(entity_id)
    
    return {
        "primary": list(primary),
        "mentions": list(mentions - primary)
    }


# In[7]:


def analyze_texts(texts_df, entities):
    '''
    Анализирует все тексты и находит упомянутые сущности.
    args: 
        texts_df (DataFrame): DataFrame с текстами.
        entities (lst): Список сущностей.
    returns: 
        results (lst): Список результатов анализа.
    '''
    results = []
    for _, row in texts_df.iterrows():
        text_id = row["id"]
        text = row["text"]
        classification = classify_text(text, entities)
        results.append({
            "text_id": text_id,
            "result": classification
        })
    return results


# In[8]:


def save_results(results, output_file):
    '''
    Сохраняет результаты в файл JSON.
    args: 
        results (lst): Список результатов анализа.
        output_file (str): Имя выходного файла.
    '''
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


# In[9]:


def main():
    # Пути к csv-файлам
    entities_file = r"C:\Users\stepa\Downloads\test_items.csv"  # Сущности
    texts_file = r"C:\Users\stepa\Downloads\test_texts.csv"       # Тексты
    output_file = "results.json"   # Результат

    # Загрузка данных
    entities = load_entities(entities_file)
    texts_df = load_texts(texts_file)

    # Анализ текстов
    results = analyze_texts(texts_df, entities)

    # Сохранение результатов
    save_results(results, output_file)
    print(f"Результаты сохранены в {output_file}")


# In[10]:


if __name__ == "__main__":
    main()


# In[11]:


with open("results.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Просмотреть содержимое
print(json.dumps(data, ensure_ascii=False, indent=2))


# In[ ]:




