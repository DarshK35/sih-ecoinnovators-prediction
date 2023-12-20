import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

loaded_model = load_model("working-predictor-v7.h5")

new_data = pd.read_csv("Data/export_data.csv")
new_data_features = new_data.iloc[:, 4].values

scaler = joblib.load("scaler.jblb")
new_data_features = new_data_features.reshape(-1, 1)
new_data_normalized = scaler.transform(new_data_features)

# Create sequences for the LSTM model
def create_sequences(data, seq_length):
	sequences = []
	targets = []
	for i in range(len(data) - seq_length):
		seq = data[i:i+seq_length]
		target = data[i+seq_length:i+seq_length+1]
		sequences.append(seq)
		targets.append(target)
	return np.array(sequences), np.array(targets)
sequence_length = 10
X_new, _ = create_sequences(new_data_normalized, sequence_length)

# Make predictions using the loaded model
predictions = loaded_model.predict(X_new)

# Inverse transform the predictions to the original scale
predictions_inv = scaler.inverse_transform(predictions)

# Print or use the predictions as needed
print(predictions_inv)