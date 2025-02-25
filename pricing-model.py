import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# Step 1: Generate Synthetic Dataset
# Assume payoff matrix for a simple pricing game
# Companies can choose High Price (H) or Low Price (L)
# Payoffs (profit) are given in the matrix form
payoffs = {
    ('H', 'H'): (50, 50),
    ('H', 'L'): (30, 60),
    ('L', 'H'): (60, 30),
    ('L', 'L'): (40, 40)
}

# Generate synthetic scenarios
np.random.seed(42)
scenarios = [(np.random.choice(['H', 'L']), np.random.choice(['H', 'L'])) for _ in range(1000)]
profits = [payoffs[scenario] for scenario in scenarios]

# Create DataFrame
data = pd.DataFrame(scenarios, columns=['Company_A_Price', 'Company_B_Price'])
data['Company_A_Profit'], data['Company_B_Profit'] = zip(*profits)

# Step 2: Feature Engineering
# Encode prices as features: H -> 1, L -> 0
data['Company_A_Price'] = data['Company_A_Price'].map({'H': 1, 'L': 0})
data['Company_B_Price'] = data['Company_B_Price'].map({'H': 1, 'L': 0})

# Step 3: Neural Network Model
X = data[['Company_A_Price', 'Company_B_Price']]
y = data[['Company_A_Profit', 'Company_B_Profit']]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(2)  # Output layer: Predicting profits for both companies
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Step 4: Metrics and Plots
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plotting actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test.iloc[:, 0], predictions[:, 0], alpha=0.5, label='Company A')
plt.scatter(y_test.iloc[:, 1], predictions[:, 1], alpha=0.5, color='red', label='Company B')
plt.title('Actual vs. Predicted Profits')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.legend()
plt.show()