import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df_train = pd.read_csv(r"/train.csv")


print("Первые  и последние 5 строк датасета:")
print(df_train)

print("\nКоличество пропущенных значений в каждом столбце:")
print(df_train.isnull().sum())

df_train.fillna({
    'Age': df_train['Age'].median(),
    'Cabin': df_train['Cabin'].mode()[0],
    'HomePlanet': df_train['HomePlanet'].mode()[0]
}, inplace=True)

scaler = MinMaxScaler()
numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])

df_train = pd.get_dummies(df_train, columns=['HomePlanet'], drop_first=True)

df_train.to_csv("final.csv", index=False)
print("\nОбработанный датасет сохранен в 'final.csv'")

df_final = pd.read_csv(r".venv/final.csv")
print("\nКоличество пропущенных значений в каждом столбце:")
print(df_final.isnull().sum())