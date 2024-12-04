
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle



column_names = ['Frequency', 'Angle of Attack', 'Chord Length', 'Free-stream Velocity', 'Suction Side Displacement', 'Sound Pressure Level']
data = pd.read_csv('../data/airfoil_self_noise.dat', sep='\t', header=None, names=column_names)

# features & target
X = data[['Frequency', 'Angle of Attack', 'Chord Length', 'Free-stream Velocity', 'Suction Side Displacement']]
y = data['Sound Pressure Level']

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

with open('../models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mode Trained! Mean Squared Error: {mse}")

