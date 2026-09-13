import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# قراءة البيانات
data = pd.read_csv("fp.csv")

# المدخلات
X = data[
    [
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
]

# المتغير المطلوب التنبؤ به
y = data["Effort"]

# تقسيم البيانات:
# 80% للتدريب و20% للاختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# إنشاء النموذج
model = LinearRegression()

# تدريب النموذج
model.fit(X_train, y_train)

# التنبؤ
predictions = model.predict(X_test)

# تقييم النموذج
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("===== Model Results =====")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

print("\n===== Predictions =====")

for actual, predicted in zip(y_test, predictions):
    print(
        "Actual Effort:",
        round(actual, 2),
        "| Predicted Effort:",
        round(predicted, 2)
    )