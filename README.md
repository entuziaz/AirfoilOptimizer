# AirfoilOptimizer: Self-Noise Prediction

## Project Overview
This project aims to predict the sound pressure level of airfoil self-noise using machine learning models. The dataset used for this project, stored in [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) sourced from aerodynamic studies, consists of key features related to airfoil characteristics, such as frequency, angle of attack, chord length, and suction side displacement. 

The main goal is to incorporate feature engineering techniques like feature importance and apply various machine learning models specifically the following models for regression analysis:
- Linear Regression 
- Random Forest Regressor

## Features and Methodology
- **Data Preprocessing**: The data is read from a `.dat` file, and prepared for analysis, including defining features and the target variable. The dataset is loaded using `pandas` and split into training and testing sets using `train_test_split`.
- **Feature Engineering**:
  - **Feature Importance Scores (FIS)**: We use a RandomForestRegressor to determine the most influential features. This helps in selecting the most impactful features for improved model performance.Features are ranked based on their importance scores.

- **Model Training**:
  - **Linear Regression**: Initially, a LinearRegresion model was used to observe performance.
  - **RandomForestRegressor**: A RandomForestRegressor model is then trained using the top 3 selected features.  It is a baseline model that leverages [ensemble learning](https://www.ibm.com/topics/ensemble-learning#:~:text=Ensemble%20learning%20is%20a%20machine,than%20a%20single%20model%20alone.) to predict the sound pressure level. 
- **Performance Evaluation**: Model performance is evaluated using mean squared error (MSE) as the primary metric, providing a clear understanding of each model's accuracy.


### Project Structure
- `data/`: Contains the dataset `airfoil_self_noise.dat`.
- `notebooks/`: Jupyter notebook for exploratory data analysis and visualizations (`main_analysis.ipynb`).
- `models/`: Directory for storing the trained models (`trained_model.pkl`).
- `scripts/`: Python scripts for training and prediction (`train_model.py` and `predict.py`).
- `README.md`: Project overview and instructions.
- `requirements.txt`: Lists the required Python libraries for the project.

### Data Description
The dataset (`./data/airfoil_self_noise.dat`) used for this project is the Airfoil Self-Noise Data set which contains the following columns:

- **Frequency**: Frequency of the sound.
- **Angle of Attack**: Angle of attack of the airfoil.
- **Chord Length**: Chord length of the airfoil.
- **Free-stream Velocity**: Velocity of the air stream.
- **Suction Side Displacement**: Measurement related to the displacement on the suction side.
- **Sound Pressure Level (SPL)**: The target variable (dependent variable) representing the sound pressure level.

| Variable Name                          | Role    | Type        | Description | Units | Missing Values |
|----------------------------------------|---------|-------------|-------------|-------|----------------|
| frequency                              | Feature | Integer     |             | Hz    | no             |
| attack-angle                           | Feature | Binary      |             | deg   | no             |
| chord-length                           | Feature | Continuous  |             | m     | no             |
| free-stream-velocity                   | Feature | Continuous  |             | m/s   | no             |
| suction-side-displacement-thickness     | Feature | Continuous  |             | m     | no             |
| scaled-sound-pressure                  | Target  | Continuous  |             | dB    | no             |


#### Running the Scripts

##### 1. Dependencies(`requirements.txt`)

Ensure all dependencies are installed in your Python virtual environment by running:

```bash
pip install -r requirements.txt
```

##### 2. Exploratory Data Analysis (`main_analysis.ipynb`)

The Jupyter notebook provides visualizations and basic statistics of the dataset. Run it in a Jupyter environment to explore the distribution of features and relationships between them.

##### 3. Training the Model (`train_model.py`)

To train the model, run the following command:

```bash
python scripts/train_model.py
```
This script loads the data, performs feature selection, trains the `RandomForestRegressor`, saves the trained model, and outputs the feature importance ranking and MSE.

##### 4. Making Predictions (`predict.py`)

To use the trained model for making predictions, run:
```bash
python scripts/predict.py
```

This script loads the trained model from the `./models/trained_model.pkl`, accepts new input data, and prints the predicted SPL.

