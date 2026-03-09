import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib

# load data
df = pd.read_csv("features_dataset.csv")

# keep only jaundice cases
df = df[df["bilirubin"] > 2]

features = [
"mean_r","mean_g","mean_b",
"mean_h","mean_s","mean_v",
"mean_l","mean_a","mean_b_lab",
"yellow_index"
]

X = df[features]
y = df["bilirubin"]

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

model = xgb.XGBRegressor(
n_estimators=300,
learning_rate=0.05,
max_depth=4,
random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test,pred)

print("MAE:",mae)

joblib.dump(model,"bilirubin_regressor.pkl")

print("Model saved as bilirubin_regressor.pkl")