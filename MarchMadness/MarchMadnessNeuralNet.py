import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Excel data
data = pd.read_excel("C:/Users/cmcna/OneDrive/Documents/Side Projects/Sports/March Madness Statistics.xlsx", header=None).to_numpy()

def build_matrices(start, finish):
    num_rows = finish - start + 1
    A = np.zeros((num_rows, 54))
    B = np.zeros((num_rows, 1))

    for row in range(start, finish + 1, 2):
        row_data = data[row - 1, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,26,28,30,32,34,36]]
        row_data_2 = data[row, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,26,28,30,32,34,36]]

        idx1 = row - start
        idx2 = idx1 + 1

        A[idx1, 0:27] = row_data
        A[idx1, 27:54] = row_data_2
        A[idx2, 0:27] = row_data_2
        A[idx2, 27:54] = row_data

    for row in range(start, finish + 1):
        idx = row - start
        B[idx, 0] = data[row - 1, 38]  # Score differential

    return A, B

# Combine all rounds into one dataset
matchups = {
    "8_9": (9, 104),
    "5_12": (107, 202),
    "4_13": (205, 300),
    "6_11": (303, 398),
    "3_14": (401, 496),
    "7_10": (499, 592),
    "2_15": (595, 690),
    "S1": (693, 788),
    "S2": (791, 886),
    "S3": (889, 984),
    "S4": (987, 1082),
    "SS1": (1085, 1180),
    "SS2": (1183, 1278),
    "EE": (1281, 1376),
    "FF": (1379, 1450),
}

# Stack all data
A_all = []
B_all = []

for start, finish in matchups.values():
    A, B = build_matrices(start, finish)
    A_all.append(A)
    B_all.append(B)

A_all = np.vstack(A_all)
B_all = np.vstack(B_all)

# Normalize features
scaler = StandardScaler()
A_scaled = scaler.fit_transform(A_all)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(A_scaled, B_all, test_size=0.2, random_state=42)

# Read and prepare custom input for prediction (outside the loop since input is same each time)
input_df = pd.read_excel(
    "C:/Users/cmcna/OneDrive/Documents/Side Projects/Neural Net/March Madness Statistics.xlsx",
    sheet_name=0,
    usecols="D:AK",
    nrows=2,
    header=None
)

columns_to_drop = [23, 25, 27, 29, 31, 33, 35]  # Corresponding to X, Z, AB, AD, AF, AH, AJ

# Drop unwanted columns
input_df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Confirm shape is now (2, 27)
assert input_df.shape == (2, 27), f"Expected shape (2, 27), got {input_df.shape}"

# Split team stats
team1_stats = input_df.iloc[0].values
team2_stats = input_df.iloc[1].values
print(f"team 1 stats: {team1_stats}")
print(f"team 2 stats: {team2_stats}")

# Combine into 1 input row for prediction
custom_input = np.hstack([team1_stats, team2_stats]).reshape(1, -1)

# Scale using same scaler as training data
custom_input_scaled = scaler.transform(custom_input)

# Array to hold predictions
predictions = []

# Train and predict 10 times
for i in range(100):
    print(f"\nTraining model iteration {i+1}...")
    
    # Rebuild model each time (fresh weights)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(54,)),
        Dense(64, activation='relu'),
        Dense(1)  # Output: predicted score differential
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0)
    
    # Evaluate on test set (optional, can comment out if not needed)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.4f}")
    
    # Predict score differential for custom input
    predicted_diff = model.predict(custom_input_scaled)[0][0]
    print(f"Predicted score differential (iteration {i+1}): {predicted_diff:.2f}")
    
    predictions.append(predicted_diff)

# Print all predictions
print("\nAll predicted score differentials:", predictions)

# Print average prediction
avg_prediction = np.mean(predictions)
print(f"\nAverage predicted score differential over 100 models: {avg_prediction:.2f}")
