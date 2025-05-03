import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow import keras

# Шаг 1. Загрузка данных
X = np.loadtxt('dataIn.txt')
Y = np.loadtxt('dataOut.txt')

# Шаг 2. Разделение на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Шаг 3. Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Шаг 4. Создание модели MLP с одним скрытым слоем (logsig = sigmoid)
model = keras.Sequential([
    keras.layers.Dense(16, activation='sigmoid', input_shape=(12,)),
    keras.layers.Dense(2, activation='softmax')
])

# Шаг 5. Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Шаг 6. Обучение модели
history = model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

# Шаг 7. Оценка модели
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

print("Accuracy:", accuracy_score(Y_test_classes, Y_pred_classes))

# Шаг 8. Визуализация потерь
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
