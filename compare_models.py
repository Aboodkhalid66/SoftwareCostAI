import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# قراءة البيانات
data = pd.read_csv("fp.csv")


# Features
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


# Target
y = data["Effort"]


# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# تعريف النماذج
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
}


print("===== Model Comparison =====\n")


# تدريب واختبار النماذج
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(name)
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 2))
    print("--------------------------")