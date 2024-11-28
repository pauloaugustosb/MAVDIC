import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
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
data = pd.read_csv(file_path)

# Dividir o conjunto de dados: 60% Treinamento, 20% Validação, 20% Teste
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=SEED)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

# Funções de pertinência fuzzy
def gaussmf(x, c, sigma, epsilon=1e-6):
    sigma = max(sigma, epsilon)
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# Sistema fuzzy
def fuzzy_system(pctid, y, params):
    if 20 <= pctid <= 40:
        low_pctid, medium_pctid, high_pctid = 1, 0, 0
    elif 40 < pctid <= 60:
        low_pctid, medium_pctid, high_pctid = 0, 1, 0
    elif 60 < pctid <= 100:
        low_pctid, medium_pctid, high_pctid = 0, 0, 1

    normal_y = gaussmf(y, params[0], params[1])
    deviating_y = gaussmf(y, params[2], params[3])
    anomalous_y = gaussmf(y, params[4], params[5])

    low_anomaly = np.fmin(low_pctid, normal_y)
    medium_anomaly = np.fmin(medium_pctid, deviating_y)
    high_anomaly = np.fmin(high_pctid, anomalous_y)

    anomaly_score = (np.sum(low_anomaly * 0.3 + medium_anomaly * 0.6 + high_anomaly * 1.0) /
                     (np.sum(low_anomaly) + np.sum(medium_anomaly) + np.sum(high_anomaly) + 1e-6))
    return anomaly_score

# Função para mutação coordenada
def coordinated_mutation(individual, mutation_rate=0.1, epsilon=1e-6):
    for i in range(0, len(individual), 2):  # Assume que C e S são pares consecutivos
        if random.random() < mutation_rate:
            delta_c = random.uniform(-0.1, 0.1)
            delta_s = random.uniform(-0.05, 0.05)
            individual[i] = np.clip(individual[i] + delta_c, 0, 1)  # Mutação no centro (C)
            individual[i + 1] = np.clip(individual[i + 1] + delta_s, epsilon, 1)  # Mutação no sigma (S)
    return individual

# Algoritmo Genético
def genetic_algorithm(train_data, val_data, test_data, pop_size=50, generations=10, mutation_rate=0.1):
    population = [np.random.uniform(0, 1, 6) for _ in range(pop_size)]
    mse_history_val, mse_history_test = [], []
    accuracy_history_val, accuracy_history_test = [], []

    def fitness(individual, dataset):
        predictions = [fuzzy_system(row['pctid'], row['y'], individual) for _, row in dataset.iterrows()]
        mse = mean_squared_error(dataset['y'], predictions)
        accuracy = accuracy_score(np.round(dataset['y']), np.round(predictions))
        return mse, accuracy

    for generation in tqdm(range(generations), desc="Gerações", leave=True):
        scores = [fitness(ind, train_data) for ind in population]
        val_scores = [fitness(ind, val_data) for ind in population]
        test_scores = [fitness(ind, test_data) for ind in population]

        # Melhor desempenho atual
        best_mse_val = min([score[0] for score in val_scores])
        best_accuracy_val = max([score[1] for score in val_scores])
        mse_history_val.append(best_mse_val)
        accuracy_history_val.append(best_accuracy_val)

        best_mse_test = min([score[0] for score in test_scores])
        best_accuracy_test = max([score[1] for score in test_scores])
        mse_history_test.append(best_mse_test)
        accuracy_history_test.append(best_accuracy_test)

        # Seleção elitista (20% melhores)
        sorted_population = sorted(zip(scores, population), key=lambda x: x[0][0])
        best_individuals = [ind for _, ind in sorted_population[:pop_size // 5]]

        # Validação cruzada (80% piores)
        worst_individuals = [ind for _, ind in sorted_population[-int(pop_size * 0.5):]]
        for ind in worst_individuals:
            ind = coordinated_mutation(ind, mutation_rate)

        # Crossover entre melhores
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(best_individuals, 2)
            split = len(parent1) // 2
            child = np.concatenate((parent1[:split], parent2[split:]))
            new_population.append(child)

        population = new_population[:pop_size]

    return mse_history_val, mse_history_test, accuracy_history_val, accuracy_history_test

# Função para rodar múltiplas simulações
def run_multiple_generations(train_data, val_data, test_data, generation_list, pop_size=50, mutation_rate=0.1):
    results = {}
    for generations in generation_list:
        print(f"Rodando com {generations} gerações...")
        mse_val, mse_test, acc_val, acc_test = genetic_algorithm(train_data, val_data, test_data, pop_size, generations, mutation_rate)
        results[generations] = {"mse_val": mse_val, "mse_test": mse_test, "acc_val": acc_val, "acc_test": acc_test}
    return results

# Executar múltiplas simulações
generation_list = [11, 51, 101, 151, 201, 251, 301]
results = run_multiple_generations(train_data, val_data, test_data, generation_list)

# Plotar resultados
os.makedirs('src/GaFuzzy/images', exist_ok=True)

for metric in ['mse', 'acc']:
    plt.figure(figsize=(12, 6))
    for gen in generation_list:
        plt.plot(results[gen][f'{metric}_val'], label=f'Validação ({gen} Gerações)')
        plt.plot(results[gen][f'{metric}_test'], label=f'Teste ({gen} Gerações)')
    plt.xlabel('Geração')
    plt.ylabel('MSE' if metric == 'mse' else 'Accuracy')
    plt.title(f'{metric.upper()} vs Gerações (Múltiplas Simulações)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'src/GaFuzzy/images/{metric}_multi_generations.png')
  
