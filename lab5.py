import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(42)

X = np.random.randint(0, 2, size=(1000, 12))

Y = np.zeros((1000, 2), dtype=int)

random_classes = np.random.randint(0, 2, size=100)
Y[np.arange(100), random_classes] = 1

np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

X = np.loadtxt("dataIn.txt")
Y_onehot = np.loadtxt("dataOut.txt")
y = np.argmax(Y_onehot, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='logistic',
    solver='adam',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("\nОтчет:\n", classification_report(y_test, y_pred, zero_division=0))

plt.plot(model.loss_curve_, color='green', linestyle='--', label='Потери на обучающем наборе')

plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.title('потери на обучающем наборе для многослойного перцептрона')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()