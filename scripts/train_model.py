
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pickle



column_names = ['Frequency', 'Angle of Attack', 'Chord Length', 'Free-stream Velocity', 'Suction Side Displacement', 'Sound Pressure Level']
data = pd.read_csv('../data/airfoil_self_noise.dat', sep='\t', header=None, names=column_names)

# features & target
X = data[['Frequency', 'Angle of Attack', 'Chord Length', 'Free-stream Velocity', 'Suction Side Displacement']]
y = data['Sound Pressure Level']

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature selection with random forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# get feature importances and create dataframe to rank them
importance_scores = rf_model.feature_importances_
features =  X.columns
feature_importance_df = pd.DataFrame({
    'Feature': features, 
    'Importance': importance_scores
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)


# Select top 3 important features
important_features = feature_importance_df['Feature'].head(3).tolist()
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

# train linear regression model with selected features
# model = LinearRegression()
# model.fit(X_train_selected, y_train)

# train random forest regressor model with selected features
model = RandomForestRegressor(random_state=42)
model.fit(X_train_selected, y_train)

# saving trrained model
with open('../models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# evaluate model
y_pred = model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print(f"Mode Trained! Mean Squared Error: {mse}")

