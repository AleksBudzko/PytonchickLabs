import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Читаем данные
try:
    train = pd.read_csv('itog.csv')
except FileNotFoundError:
    print("Ошибка: Файл 'itog.csv' не найден. Убедитесь в правильности пути.")
    exit()

# Проверяем наличие необходимых колонок
required_columns = {
    'features': ['Логистическая регрессия', 'Линейная регрессия'],
    'target': 'Истинное значение'
}

print("Доступные колонки в данных:", train.columns.tolist())

# Проверяем наличие всех необходимых колонок
missing_columns = [col for col in required_columns['features'] + [required_columns['target']]
                   if col not in train.columns]
if missing_columns:
    print(f"Ошибка: Отсутствуют необходимые колонки: {missing_columns}")
    exit()

# Формируем данные
X = train[required_columns['features']]
y = train[required_columns['target']]

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Инициализируем модели
models = {
    'Случайный лес': RandomForestClassifier(random_state=42),
    'Градиентный бустинг': GradientBoostingClassifier(random_state=42)
}

# Обучаем и оцениваем модели
for name, model in models.items():
    # Обучение
    model.fit(X_train, y_train)

    # Предсказание
    y_pred = model.predict(X_test)

    # Оценка
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

    # Кросс-валидация
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Val Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")