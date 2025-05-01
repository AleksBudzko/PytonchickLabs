import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Загрузка наборов данных
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Первые строки обучающего набора:")
print(train_df.head())
print("Информация о наборе данных (пропуски, типы):")
print(train_df.info())

def preprocess_dataframe(df):
    """
    Функция предобработки данных:
    - Удаляются столбцы, не влияющие на модель.
    - Логические переменные преобразуются в числовой формат.
    - Применяется one-hot кодирование для категориальных признаков.
    - Заполняются пропуски в числовых признаках средним значением.
    """
    # Исключаем ненужные признаки
    df = df.drop(columns=['PassengerId', 'Name', 'Cabin'])
    # Преобразование булевых признаков
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0})
    df['VIP'] = df['VIP'].map({True: 1, False: 0})
    # Кодирование категориальных переменных
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination'])
    # Заполнение пропусков для числовых столбцов
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
    return df

# Применяем предобработку к обучающему и тестовому набору
train_df = preprocess_dataframe(train_df)
test_df = preprocess_dataframe(test_df)

# Разделение признаков и целевой переменной
features = train_df.drop(columns=['Transported'])
target = train_df['Transported'].astype(int)

# Разбиваем данные на обучающую и валидационную выборки
feat_train, feat_val, target_train, target_val = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Масштабирование признаков
data_scaler = StandardScaler()
feat_train = data_scaler.fit_transform(feat_train)
feat_val = data_scaler.transform(feat_val)

# Обучение модели логистической регрессии
log_regressor = LogisticRegression(max_iter=500)
log_regressor.fit(feat_train, target_train)
pred_logistic = log_regressor.predict(feat_val)

# Обучение модели линейной регрессии
lin_regressor = LinearRegression()
lin_regressor.fit(feat_train, target_train)
# Преобразуем предсказания в бинарный формат
pred_linear = (lin_regressor.predict(feat_val) > 0.5).astype(int)

def assess_model(actual, predicted, model_label):
    """Функция для оценки модели по метрике Accuracy."""
    accuracy = accuracy_score(actual, predicted)
    print(f"{model_label}:")
    print(f"Accuracy: {accuracy:.2f}")

# Оценка моделей
assess_model(target_val, pred_logistic, "Логистическая регрессия")
assess_model(target_val, pred_linear, "Линейная регрессия")

# Вычисление матриц ошибок
cm_logistic = confusion_matrix(target_val, pred_logistic)
cm_linear = confusion_matrix(target_val, pred_linear)

# Визуализация матриц ошибок с помощью heatmap
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(cm_logistic, annot=True, fmt='d', cmap='Greens', ax=axes[0])
axes[0].set_xlabel('Предсказано: отрицательный')
axes[0].set_ylabel('Истинное значение')
axes[0].set_title('Матрица ошибок: Логистическая регрессия')

sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Reds', ax=axes[1])
axes[1].set_xlabel('Предсказано: отрицательный')
axes[1].set_ylabel('Истинное значение')
axes[1].set_title('Матрица ошибок: Линейная регрессия')

plt.tight_layout()
plt.show()

# Формирование итогового DataFrame с результатами
output_df = pd.DataFrame({
    'Истинное значение': target_val,
    'Логистическая регрессия': pred_logistic,
    'Линейная регрессия': pred_linear
})

# Сохранение результатов в CSV файл
output_df.to_csv('itog.csv', index=False)
print("Итоговый CSV файл сохранен как 'itog.csv'")
