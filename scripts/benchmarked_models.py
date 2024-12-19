
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pickle
import os



####### 1. DATA SET  #######

# Read Self-noise CSV dataset
column_names = ['Frequency', 'Angle of Attack', 'Chord Length', 'Free-stream Velocity', 'Suction Side Displacement', 'Sound Pressure Level']
data = pd.read_csv('../data/airfoil_self_noise.dat', sep='\t', header=None, names=column_names)

# features & target
X = data[['Frequency', 'Angle of Attack', 'Chord Length', 'Free-stream Velocity', 'Suction Side Displacement']]
y = data['Sound Pressure Level']

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



####### 2. FEATURE SELECTION  #######

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




####### 3. MODELS  #######

models = {
    'LinearRegression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'SVM': SVR()
}

# Parameter grids for tuning
param_grids = {
    'LinearRegression': {}, # a simplistic model with no hyperparameters
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}


####### 4. BENCHMARKING HYPERPARAMETER-TUNED MODELS  #######

# benchmarked results:
results = []

# Hyperparameter tuning with GridSearchCV
for model_name, model in models.items():
    print(f'\nTraining {model_name}...')
    if model_name in param_grids and param_grids[model_name]:  # tuning models with params 
        grid_search = GridSearchCV(
            model,
            param_grid=param_grids[model_name],
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=2
        )
        grid_search.fit(X_train_selected, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_mse = -grid_search.best_score_  # converting negative MSE to positive
    else: # For linear Regression [no hyperparams]
        model.fit(X_train_selected, y_train)
        best_model = model
        best_params = "Default parameters"
        best_cv_mse = "N/A"

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error on Test Set: {mse}")
    if model_name in param_grids and param_grids[model_name]:
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-validated MSE: {best_cv_mse}")

    model_filename = os.path.join('../models/', f"{model_name.replace(' ', '_').lower()}_model.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Saved trained {model_name} to {model_filename}")

    results.append({
        "Model": model_name,
        "MSE": mse,
        "Best Parameters": best_params,
        "Best Cross-validated MSE": best_cv_mse,
        "Model Path": model_filename
    })

# Tabulate results into DataFrame & save
results_df = pd.DataFrame(results)
results_df.sort_values(by="MSE", ascending=True, inplace=True)
results_df.to_csv('../data/model_benchmark_results.csv', index=False)
print("\nResults saved to '../data/model_benchmark_results.csv'")

print("\nBenchmarking Results:")
print(results_df)
