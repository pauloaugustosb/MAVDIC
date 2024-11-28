import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Definir a semente randômica
SEED = 73
np.random.seed(SEED)
random.seed(SEED)

# Carregar o dataset (sem ordenação)
file_path = 'datasets/data/processed_teste_train_config_1_database.csv'
data = pd.read_csv(file_path)  # Mantém a ordem original do dataset

# Dividir o conjunto de dados: 60% Treinamento, 20% Validação, 20% Teste
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=SEED)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

# Funções de pertinência fuzzy vetorizadas
def gaussmf(x, c, sigma, epsilon=1e-6):
    sigma = np.maximum(sigma, epsilon)
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

def fuzzy_system_vectorized(pctid, y, params):
    low_pctid = (pctid >= 20) & (pctid <= 40)
    medium_pctid = (pctid > 40) & (pctid <= 60)
    high_pctid = (pctid > 60) & (pctid <= 100)

    normal_y = gaussmf(y, params[0], params[1])
    deviating_y = gaussmf(y, params[2], params[3])
    anomalous_y = gaussmf(y, params[4], params[5])

    low_anomaly = np.minimum(low_pctid, normal_y)
    medium_anomaly = np.minimum(medium_pctid, deviating_y)
    high_anomaly = np.minimum(high_pctid, anomalous_y)

    numerator = np.sum(low_anomaly * 0.3 + medium_anomaly * 0.6 + high_anomaly * 1.0, axis=0)
    denominator = np.sum(low_anomaly + medium_anomaly + high_anomaly, axis=0) + 1e-6

    # Regularização penalizando parâmetros extremos
    regularization = np.sum(params ** 2) * 0.01  # Penalização leve
    return numerator / denominator - regularization

# Função para mutação coordenada
def coordinated_mutation(individual, mutation_rate=0.1, epsilon=1e-6):
    for i in range(0, len(individual), 2):  # Assume que C e S são pares consecutivos
        if random.random() < mutation_rate:
            delta_c = random.uniform(-0.1, 0.1)
            delta_s = random.uniform(-0.05, 0.05)
            individual[i] = np.clip(individual[i] + delta_c, 0, 1)  # Mutação no centro (C)
            individual[i + 1] = np.clip(individual[i + 1] + delta_s, epsilon, 1)  # Mutação no sigma (S)
    return individual

# Algoritmo Genético com cálculo vetorizado
def genetic_algorithm(train_data, val_data, test_data, pop_size=50, generations=101, mutation_rate=0.1):
    print("Iniciando treinamento...")
    population = np.random.uniform(0, 1, (pop_size, 6))
    mse_history_val, mse_history_test = [], []
    mae_history_val, mae_history_test = [], []
    accuracy_history_val, accuracy_history_test = [], []

    def fitness_vectorized(individuals, dataset):
        pctid = dataset['pctid'].values
        y_true = dataset['y'].values
        predictions = np.array([fuzzy_system_vectorized(pctid, y_true, ind) for ind in individuals])
        mse = np.mean((y_true[:, np.newaxis] - predictions.T) ** 2, axis=0)
        mae = np.mean(np.abs(y_true[:, np.newaxis] - predictions.T), axis=0)
        accuracy = np.mean(np.round(y_true[:, np.newaxis]) == np.round(predictions.T), axis=0)
        return mse, mae, accuracy

    for generation in tqdm(range(generations), desc="Gerações", leave=True):
        mse_train, _, _ = fitness_vectorized(population, train_data)
        mse_val, mae_val, acc_val = fitness_vectorized(population, val_data)
        mse_test, mae_test, acc_test = fitness_vectorized(population, test_data)

        # Melhor desempenho atual
        mse_history_val.append(np.min(mse_val))
        mae_history_val.append(np.min(mae_val))
        accuracy_history_val.append(np.max(acc_val))

        mse_history_test.append(np.min(mse_test))
        mae_history_test.append(np.min(mae_test))
        accuracy_history_test.append(np.max(acc_test))

        # Seleção elitista (20% melhores)
        sorted_indices = np.argsort(mse_train)
        best_individuals = population[sorted_indices[:pop_size // 2]]

        # Introdução de diversidade na população
        diversity_individuals = np.random.uniform(0, 1, (pop_size // 10, 6))
        best_individuals = np.vstack((best_individuals, diversity_individuals))

        # Validação cruzada aprimorada
        worst_indices = sorted_indices[-int(pop_size * 0.5):]
        for idx in worst_indices:
            population[idx] = coordinated_mutation(population[idx], mutation_rate)

        # Crossover entre melhores
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(list(best_individuals), 2)
            split = len(parent1) // 2
            child = np.concatenate((parent1[:split], parent2[split:]))
            new_population.append(child)

        population = np.array(new_population[:pop_size])

    # Métricas finais
    print(f"MSE Final Validação: {mse_history_val[-1]:.4f}")
    print(f"MAE Final Validação: {mae_history_val[-1]:.4f}")
    print(f"Accuracy Final Validação: {accuracy_history_val[-1]:.4f}")
    print(f"MSE Final Teste: {mse_history_test[-1]:.4f}")
    print(f"MAE Final Teste: {mae_history_test[-1]:.4f}")
    print(f"Accuracy Final Teste: {accuracy_history_test[-1]:.4f}")

    # Gráficos finais
    os.makedirs('src/GaFuzzy/images', exist_ok=True)

    plt.figure(figsize=(12, 12))

    # MSE vs Gerações
    plt.subplot(311)
    plt.plot(mse_history_val, label='Validação', marker='o')
    plt.plot(mse_history_test, label='Teste', marker='x')
    plt.xlabel('Geração')
    plt.ylabel('MSE')
    plt.title('MSE vs Gerações')
    plt.legend()
    plt.grid(True)

    # MAE vs Gerações
    plt.subplot(312)
    plt.plot(mae_history_val, label='Validação', marker='o')
    plt.plot(mae_history_test, label='Teste', marker='x')
    plt.xlabel('Geração')
    plt.ylabel('MAE')
    plt.title('MAE vs Gerações')
    plt.legend()
    plt.grid(True)

    # Accuracy vs Gerações
    plt.subplot(313)
    plt.plot(accuracy_history_val, label='Validação', marker='o')
    plt.plot(accuracy_history_test, label='Teste', marker='x')
    plt.xlabel('Geração')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Gerações')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('src/GaFuzzy/images/metrics_vs_generations_vetorizado101_3.png')
    plt.close()

    return population[0], mse_history_val, mae_history_val, accuracy_history_val

# Executar o treinamento
best_params, mse_val, mae_val, acc_val = genetic_algorithm(train_data, val_data, test_data)
