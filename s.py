import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
import random
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Load the dataset and transfer to GPU
file_path = 'datasets/data/processed_teste_train_config_1_database.csv'
data = pd.read_csv(file_path)

# Convert data to torch tensors and transfer to GPU
data = {col: torch.tensor(data[col].values, dtype=torch.float32, device=device) for col in data.columns}

# Split dataset into training, validation, and test sets
train_data, temp_data = train_test_split(pd.DataFrame(data), test_size=0.2, random_state=42)
val_data, test_data = train_test_split(pd.DataFrame(train_data), test_size=0.2, random_state=42)

# Convert splits to torch tensors
train_data = {col: torch.tensor(train_data[col].values, dtype=torch.float32, device=device) for col in train_data.columns}
val_data = {col: torch.tensor(val_data[col].values, dtype=torch.float32, device=device) for col in val_data.columns}
test_data = {col: torch.tensor(test_data[col].values, dtype=torch.float32, device=device) for col in test_data.columns}

# Fuzzy membership function with Gaussian
def gaussmf(x, c, sigma, epsilon=1e-6):
    sigma = torch.max(sigma, torch.tensor(epsilon, device=device))
    return torch.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# Gradient descent to adjust Gaussian parameters
def gradient_descent_gaussian(y, c, sigma, learning_rate=0.01, epochs=100):
    for _ in range(epochs):
        grad_c = -((y - c) / (sigma ** 2)) * gaussmf(y, c, sigma)
        grad_sigma = ((y - c) ** 2 / (sigma ** 3)) * gaussmf(y, c, sigma)

        c -= learning_rate * grad_c.mean()
        sigma -= learning_rate * grad_sigma.mean()
        
        sigma = torch.max(sigma, torch.tensor(1e-6, device=device))  # Ensure sigma is positive
    
    return c, sigma

# Fuzzy system definition with initial membership based on pctid categories
def fuzzy_system(pctid, y, params):
    # Categorizations for pctid as low, medium, high
    low_pctid = torch.where((20 <= pctid) & (pctid <= 40), 1, 0)
    medium_pctid = torch.where((40 < pctid) & (pctid <= 60), 1, 0)
    high_pctid = torch.where((60 < pctid) & (pctid <= 100), 1, 0)

    # Gaussian membership functions for y
    normal_y = gaussmf(y, params[0], params[1])
    deviating_y = gaussmf(y, params[2], params[3])
    anomalous_y = gaussmf(y, params[4], params[5])

    # Fuzzy rules using pctid categories
    low_anomaly = torch.min(low_pctid, normal_y)
    medium_anomaly = torch.min(medium_pctid, deviating_y)
    high_anomaly = torch.min(high_pctid, anomalous_y)

    # Simple defuzzification (center of mass)
    anomaly_score = (torch.sum(low_anomaly * 0.3 + medium_anomaly * 0.6 + high_anomaly * 1.0) /
                     (torch.sum(low_anomaly) + torch.sum(medium_anomaly) + torch.sum(high_anomaly) + 1e-6))
    return anomaly_score

# Genetic Algorithm with metric tracking
def genetic_algorithm(train_data, val_data, pop_size=20, generations=50):
    # Initialize population with random values
    population = [torch.rand(6, device=device) for _ in range(pop_size)]
    train_mae_history, val_mae_history = [], []

    def fitness(individual, dataset):
        # Evaluate individual on dataset
        predictions = torch.tensor([fuzzy_system(row['pctid'], row['y'], individual) for _, row in dataset.iterrows()], device=device)
        mae = F.l1_loss(dataset['y'], predictions)
        return mae

    for generation in range(generations):
        # Calculate fitness for training and validation
        train_scores = [(fitness(ind, train_data), ind) for ind in population]
        val_scores = [(fitness(ind, val_data), ind) for ind in population]

        # Get best MAE for train and validation
        best_train_mae = min(train_scores, key=lambda x: x[0])[0].item()
        best_val_mae = min(val_scores, key=lambda x: x[0])[0].item()
        train_mae_history.append(best_train_mae)
        val_mae_history.append(best_val_mae)

        # Select the top half for the next generation
        selected = [ind for _, ind in sorted(train_scores, key=lambda x: x[0])[:pop_size // 2]]

        # Crossover and mutation
        new_population = selected.copy()
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            cross_point = random.randint(1, len(parent1) - 1)
            child = torch.cat((parent1[:cross_point], parent2[cross_point:]))
            if random.random() < 0.1:
                child[random.randint(0, len(child) - 1)] += torch.randn(1, device=device) * 0.1
            new_population.append(torch.clamp(child, 0, 1))
        
        population = new_population

    best_individual = min(population, key=lambda ind: fitness(ind, val_data))
    return best_individual, train_mae_history, val_mae_history

# Training and metric history tracking
best_params, train_mae_history, val_mae_history = genetic_algorithm(train_data, val_data)

# Evaluate model on test data
predictions = torch.tensor([fuzzy_system(row['pctid'], row['y'], best_params) for _, row in test_data.iterrows()], device=device)

# Calculate final metrics
mae = F.l1_loss(test_data['y'], predictions).item()
mse = F.mse_loss(test_data['y'], predictions).item()
r2 = r2_score(test_data['y'].cpu().numpy(), predictions.cpu().numpy())
f1 = f1_score(torch.round(test_data['y']).cpu().numpy(), torch.round(predictions).cpu().numpy())

# Binary accuracy calculation
binary_predictions = (predictions >= 0.5).int()
binary_real = (test_data['y'] >= 0.5).int()
accuracy = accuracy_score(binary_real.cpu().numpy(), binary_predictions.cpu().numpy())

# Final loss display
print(f"Loss (MAE) on test set: {mae}")

# Plot MAE history
plt.figure(figsize=(12, 6))
plt.plot(train_mae_history, label='Training MAE')
plt.plot(val_mae_history, label='Validation MAE')
plt.xlabel('Generation')
plt.ylabel('MAE')
plt.title('Genetic Algorithm Performance Over Generations')
plt.legend()
plt.show()

# Display final test performance
print("Test set performance:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# Plot predictions vs real values on test set
plt.figure(figsize=(12, 6))
plt.plot(test_data['y'].cpu().numpy(), label='Real Values')
plt.plot(predictions.cpu().numpy(), label='Model Predictions')
plt.xlabel('Samples')
plt.ylabel('Anomaly (y)')
plt.title('Comparison of Real Values and Predictions on Test Set')
plt.legend()
plt.show()
