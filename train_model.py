import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# قراءة البيانات
data = pd.read_csv("fp.csv")


# Features
features = [
    "TeamExp",
    "ManagerExp",
    "YearEnd",
    "Length",
    "Transactions",
    "Entities",
    "PointsNonAdjust",
    "Adjustment",
    "PointsAjust",
    "Language"
]

X = data[features]

# Target
y = data["Effort"]


# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# إنشاء وتدريب النموذج
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)


# حفظ النموذج
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
print("Model saved as model.pkl")