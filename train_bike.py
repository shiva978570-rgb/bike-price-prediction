import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load Data
df = pd.read_csv('Used_Bikes.csv')

# 2. Features Select Karein
features = ['brand', 'owner', 'kms_driven', 'age', 'power']
X = df[features].copy()
y = df['price']

# 3. Categorical Encoding
le_dict = {}
cat_cols = ['brand', 'owner']
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = (le, list(le.classes_))

# 4. Model Train Karein
print("Training Bike Price Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Dropdown options for UI
dropdown_options = {}
for col in features:
    if col in le_dict:
        dropdown_options[col] = le_dict[col][1]
    else:
        dropdown_options[col] = sorted(df[col].unique().tolist())

# 6. Save Model
with open('bike_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'le_dict': le_dict,
        'features': features,
        'dropdown_options': dropdown_options
    }, f)

print("Success! 'bike_model.pkl' created.")