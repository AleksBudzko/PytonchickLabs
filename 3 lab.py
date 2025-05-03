import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Загрузка наборов данных
df_train = pd.read_csv('train.csv')
df_eval = pd.read_csv('test.csv')

# Вывод списка столбцов исходного обучающего набора
print("Доступные столбцы в обучающем наборе:", df_train.columns)

# Удаляем столбцы, не влияющие на модель
df_train = df_train.drop(columns=['PassengerId', 'Name'])
df_eval = df_eval.drop(columns=['PassengerId', 'Name'])

# Замена пропущенных числовых значений на медиану по столбцу
df_train.fillna(df_train.median(numeric_only=True), inplace=True)
df_eval.fillna(df_eval.median(numeric_only=True), inplace=True)

# Обработка пропущенных категориальных данных – заполняем значением "None"
for column in df_train.select_dtypes(include=['object']).columns:
    df_train[column] = df_train[column].astype(str).fillna("None")
for column in df_eval.select_dtypes(include=['object']).columns:
    df_eval[column] = df_eval[column].astype(str).fillna("None")

# Объединяем обучающий и тестовый набор для корректного кодирования категориальных признаков
combined_df = pd.concat([df_train, df_eval], axis=0)
combined_df = pd.get_dummies(combined_df)

# Выводим перечень столбцов после применения one-hot encoding
print("Столбцы после one hot encoding:", combined_df.columns)

# Проверка наличия столбца целевой переменной 'Transported_True'
if 'Transported_True' not in combined_df.columns:
    raise KeyError(f"Отсутствует колонка 'Transported_True'. Доступные столбцы: {combined_df.columns}")

# Делим объединённый набор обратно на обучающую и тестовую выборки
train_rows = len(df_train)
df_train_processed = combined_df.iloc[:train_rows, :]
df_eval_processed = combined_df.iloc[train_rows:, :]

# Формирование признакового пространства и целевого в обучающем наборе
features_train = df_train_processed.drop(columns=['Transported_False', 'Transported_True'])
target_train = df_train_processed['Transported_True']

# Обработка тестовой выборки (если имеются столбцы, связанные с целевой переменной, они исключаются)
features_eval = df_eval_processed.drop(columns=['Transported_False', 'Transported_True'], errors='ignore')

# Создание и обучение модели решающего дерева
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=2)
decision_tree.fit(features_train, target_train)

# Предсказание на обучающем наборе (так как для тестовой выборки истинные метки отсутствуют)
predictions_train = decision_tree.predict(features_train)

# Оценка точности модели по метрике Precision
model_precision = precision_score(target_train, predictions_train)
print(f'Precision (точность): {model_precision:.3f}')

# Вычисление матрицы ошибок для обучающего набора
error_matrix = confusion_matrix(target_train, predictions_train)
print('Матрица ошибок:\n', error_matrix)

# Визуализация структуры дерева решений
plt.figure(figsize=(10, 8))
plot_tree(decision_tree, filled=True, feature_names=features_train.columns, class_names=['False', 'True'])
plt.show()

"""
Условие разделения:
    CryoSleep_True <= 0.5
означает, что если значение признака CryoSleep_True меньше или равно 0.5, 
то объект попадает в левое поддерево, иначе – в правое.

Gini index (джини) характеризует степень неоднородности узла:
    samples = 8693: общее количество объектов в узле.
    value = [4315, 4378]: распределение объектов по классам (например, для классов True и False).
    class = True: класс, который будет предсказан, если условие узла выполняется.
"""
