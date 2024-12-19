
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression
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

# Base RF regressor model for feature importance
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
base_model.fit(X_train, y_train)


# Get feature importances and create dataframe to rank them
importance_scores = base_model.feature_importances_
features =  X.columns
feature_importance_df = pd.DataFrame({
    'Feature': features, 
    'Importance': importance_scores
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", feature_importance_df)

# Select top 3 important features
important_features = feature_importance_df['Feature'].head(3).tolist()
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]


# Parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2
)
grid_search.fit(X_train_selected, y_train)


# train linear regression model with selected features
# model = LinearRegression()
# model.fit(X_train_selected, y_train)


# saving the best model
base_model = grid_search.best_estimator_
with open('../models/hyperparams_tuned_model.pkl', 'wb') as f:
    pickle.dump(base_model, f)


# evaluate model
y_pred = base_model.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error of the best model: {mse}")

print('Best parameters; ', grid_search.best_params_)
print('Best Cross-validated MSE', -grid_search.best_score_)