"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRanker, Pool
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


# Предобработка текста
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Объединяем текстовые поля товара
def combine_product_text(row):
    fields = [
        row['product_title'],
        row['product_description'],
        row['product_bullet_point'],
        row['product_brand'],
        row['product_color']
    ]
    combined = ' '.join([preprocess_text(f) for f in fields if pd.notna(f)])
    return combined



# Функция для вычисления косинусного сходства
def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Добавляем признаки
def extract_features(df, query_emb_dict, product_emb_dict):
    features = []
    for _, row in df.iterrows():
        query_emb = query_emb_dict.get(row['query_processed'], np.zeros(384))
        product_emb = product_emb_dict.get(row['product_text'], np.zeros(384))
        sim = cosine_similarity(query_emb, product_emb)

        # Дополнительные признаки
        query_len = len(row['query_processed'].split())
        product_len = len(row['product_text'].split())
        keyword_match = sum(1 for word in row['query_processed'].split() if word in row['product_text'])

        features.append([sim, query_len, product_len, keyword_match])
    return np.array(features)







def create_submission(submission):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """

    # Загрузка данных
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    train_df['product_text'] = train_df.apply(combine_product_text, axis=1)
    test_df['product_text'] = test_df.apply(combine_product_text, axis=1) 

    train_df['query_processed'] = train_df['query'].apply(preprocess_text)
    test_df['query_processed'] = test_df['query'].apply(preprocess_text)

    # возьмем подвыборку если данных много
    if len(train_df) > 50000:
        print(f"Large dataset ({len(train_df)} rows), taking sample...")
        train_df = train_df.sample(50000, random_state=993)

    # Извлечение эмбеддингов
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Получаем уникальные тексты для экономии вычислений
    unique_queries_train = train_df['query_processed'].unique()
    unique_products_train = train_df['product_text'].unique()

    print(f"Unique queries: {len(unique_queries_train)}")
    print(f"Unique products: {len(unique_products_train)}")

    # Берем только первые 1000 для демонстрации, чтобы ускорить
    if len(unique_queries_train) > 1000:
        unique_queries_train = unique_queries_train[:1000]
    if len(unique_products_train) > 1000:
        unique_products_train = unique_products_train[:1000]

    query_embeddings_train = model.encode(unique_queries_train, show_progress_bar=True)
    product_embeddings_train = model.encode(unique_products_train, show_progress_bar=True)

    # Создаем словари для быстрого доступа
    query_emb_dict = dict(zip(unique_queries_train, query_embeddings_train))
    product_emb_dict = dict(zip(unique_products_train, product_embeddings_train))

    X_train = extract_features(train_df, query_emb_dict, product_emb_dict)
    X_test = extract_features(test_df, query_emb_dict, product_emb_dict)

    y_train = train_df['relevance'].values
    groups = train_df['query_id'].values

    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Обучение CatBoostRanker
    model = CatBoostRanker(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='YetiRank',
        verbose=100,
        random_seed=993
    )

    # GroupKFold валидация
    gkf = GroupKFold(n_splits=5)
    scores = []

    for train_idx, val_idx in gkf.split(X_train, y_train, groups):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        group_tr = groups[train_idx]
        group_val = groups[val_idx]

        train_pool = Pool(X_tr, y_tr, group_id=group_tr)
        val_pool = Pool(X_val, y_val, group_id=group_val)

        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        # Предсказание на валидации
        preds = model.predict(X_val)
        val_df = pd.DataFrame({'query_id': group_val, 'pred': preds, 'true': y_val})

        # Рассчитываем NDCG@10 для каждого запроса
        ndcg_scores = []
        for qid in val_df['query_id'].unique():
            df_q = val_df[val_df['query_id'] == qid].sort_values('pred', ascending=False).head(10)
            dcg = 0
            for i, (_, row) in enumerate(df_q.iterrows(), 1):
                gain = (2 ** row['true']) - 1
                discount = np.log2(i + 1)
                dcg += gain / discount

            # Идеальный DCG
            ideal = val_df[val_df['query_id'] == qid].sort_values('true', ascending=False).head(10)
            idcg = 0
            for i, (_, row) in enumerate(ideal.iterrows(), 1):
                gain = (2 ** row['true']) - 1
                discount = np.log2(i + 1)
                idcg += gain / discount

            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

        scores.append(np.mean(ndcg_scores))
        print(f'Fold NDCG@10: {np.mean(ndcg_scores):.4f}')

    print(f'Average CV NDCG@10: {np.mean(scores):.4f}')

    # Обучение на всех данных
    full_pool = Pool(X_train, y_train, group_id=groups)
    model.fit(full_pool)

    # Предсказание на тесте
    test_preds = model.predict(X_test)

    # Сохранение submission.csv
    submission = pd.DataFrame({'id': test_df['id'], 'prediction': test_preds})
    #submission.to_csv('submission.csv', index=False)
    print('Submission saved to submission.csv')

    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()