import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Загрузка данных
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print("Вывод первых строк Датасета:")
print(df_train.head())
print('Вывод кол-ва пропущенных значений и их тип: ')
print(df_train.info())

# Предобработка данных
def preprocess_data(df):
    df = df.drop(columns=['PassengerId', 'Name', 'Cabin'])  # Удаляем ненужные столбцы
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0})  # Преобразуем в числовой формат
    df['VIP'] = df['VIP'].map({True: 1, False: 0})
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination'])  # Кодируем категориальные признаки
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)  # Заполняем только числовые пропуски
    return df

df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

# Разделение данных на X и y
X = df_train.drop(columns=['Transported']) # тренировочная выборка
y = df_train['Transported'].astype(int) # Тестовая переменная
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # для того чтобы среднее=0, а стандартное отклонение 1
X_test = scaler.transform(X_test)

# Логистическая регрессия
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Линейная регрессия
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = (lin_reg.predict(X_test) > 0.5).astype(int)  # Преобразуем в бинарный формат


# Оценка моделей
def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")



evaluate_model(y_test, y_pred_log, "Логистическая регрессия")
evaluate_model(y_test, y_pred_lin, "Линейная регрессия")
# Матрицы ошибок
cm_log = confusion_matrix(y_test, y_pred_log)
cm_lin = confusion_matrix(y_test, y_pred_lin)

# Создание подграфиков
fig, axes = plt.subplots(1, 2, figsize=(13, 5)) # 1 строка, 2 столбца

# Тепловая карта для логистической регрессии
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Greens', ax=axes[0])
axes[0].set_xlabel('Ложный')
axes[0].set_ylabel('Истинный')
axes[0].set_title('Матрица ошибок логистической регрессии')

# Тепловая карта для линейной регрессии
sns.heatmap(cm_lin, annot=True, fmt='d', cmap='Reds', ax=axes[1])
axes[1].set_xlabel('Ложный')
axes[1].set_ylabel('Истинный')
axes[1].set_title('Матрица ошибок линейной регрессии')
 # Предотвращает наложение элементов графика
plt.show()

# Создание итогового DataFrame
results_df = pd.DataFrame({
    'Тестовая': y_test,
    'X': y_pred_log,
    'Y': y_pred_lin
})

# Сохранение в CSV файл
results_df.to_csv('itog.csv', index=False)
print("Итоговый CSV файл сохранен как 'itog.csv'")