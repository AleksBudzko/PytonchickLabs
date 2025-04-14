import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Загрузка данных
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Проверяем, какие столбцы есть в train_data
print("Столбцы в df_train_data:", train_data.columns)

# Удаление ненужных столбцов
train_data = train_data.drop(columns=['PassengerId', 'Name'])
test_data = test_data.drop(columns=['PassengerId', 'Name'])

# Заполнение пропущенных числовых значений
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Заполнение пропущенных категориальных значений корректно
for col in train_data.select_dtypes(include=['object']).columns:
    train_data[col] = train_data[col].astype(str).fillna("None")
for col in test_data.select_dtypes(include=['object']).columns:
    test_data[col] = test_data[col].astype(str).fillna("None")

# Преобразование категориальных данных в числовые
combined_data = pd.concat([train_data, test_data], axis=0)
combined_data = pd.get_dummies(combined_data)

# Проверяем, какие колонки получились
print("Столбцы после one hot encoding:", combined_data.columns)

# Используем Transported_True как целевую переменную
if 'Transported_True' not in combined_data.columns:
    raise KeyError(f"Колонка 'Transported_True' отсутствует. Доступные столбцы: {combined_data.columns}")

# Разделение обратно на train и test
train_data = combined_data.iloc[:len(train_data), :]
test_data = combined_data.iloc[len(train_data):, :]

# Разделение признаков и целевой переменной
X_train = train_data.drop(columns=['Transported_False', 'Transported_True'])
y_train = train_data['Transported_True']
X_test = test_data.drop(columns=['Transported_False', 'Transported_True'], errors='ignore')  # У test нет Transported

# Создание и обучение модели
clf = DecisionTreeClassifier(random_state=42, max_depth=2)

clf.fit(X_train, y_train)

# Прогнозирование
y_pred = clf.predict(X_train)  # Здесь y_pred получается для train, так как в test нет истинных значений

precision = precision_score(y_train, y_pred)  # Оценка на train
print(f'Precision: {precision:.3f}')

conf_matrix = confusion_matrix(y_train, y_pred)
print('Матрица ошибок:\n', conf_matrix)

plt.figure(figsize=(10, 8))
plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['False', 'True'])
plt.show()

"""
CryoSleep_True <= 0.5:
Это условие, по которому данные разделяются. 
Если значение признака CryoSleep_True меньше или равно 0.5, 
то образцы идут в левое поддерево; если больше — в правое.
"""

# gini -степень неоднородности выборки если =0 то неоднородная если =1 однородная
"""
samples = 8693: Общее количество примеров в этом узле.
value = [4315, 4378]: Это количество образцов для каждого класса (в данном случае, скорее всего, True и False).
class = True: Это класс, который будет назначен, если условия в узле выполняются.
"""