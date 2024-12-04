import pickle
import numpy as np
import pandas as pd

with open('../models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)


new_data = np.array([[1000, 5.0, 0.3, 71.3, 0.02]]).reshape(1, -1)
columns = ['Frequency', 'Angle of Attack', 'Chord Length', 'Free-stream Velocity', 'Suction Side Displacement']

# new_data_df = np.array(new_data, columns=columns)
new_data_df = pd.DataFrame(new_data, columns=columns)

predicted_spl = model.predict(new_data_df)
print(f"Predicted Sound Pressure Level: {predicted_spl[0]}")