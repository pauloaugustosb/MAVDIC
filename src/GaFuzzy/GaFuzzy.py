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

# Carregar o dataset
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

# Função para salvar a população inicial
def save_initial_population(population, file_path='src/GaFuzzy/initial_population.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(population, f)

# Função para carregar a população inicial
def load_initial_population(file_path='src/GaFuzzy/initial_population.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# Função para salvar checkpoints
def save_checkpoint(generation, population, mse_history, accuracy_history, file_path='src/GaFuzzy/checkpoint.pkl'):
    checkpoint = {
        'generation': generation,
        'population': population,
        'mse_history': mse_history,
        'accuracy_history': accuracy_history,
    }
    with open(file_path, 'wb') as f:
        pickle.dump(checkpoint, f)

# Função para carregar checkpoints
def load_checkpoint(file_path='src/GaFuzzy/checkpoint.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# Algoritmo Genético
def genetic_algorithm(train_data, val_data, pop_size=50, generations=10, mutation_rate=0.2, checkpoint_path='src/GaFuzzy/checkpoint.pkl', initial_population_path='src/GaFuzzy/initial_population.pkl'):
    # Carregar a população inicial
    initial_population = load_initial_population(initial_population_path)
    checkpoint = load_checkpoint(checkpoint_path)

    if initial_population is None:
        print("Gerando e salvando a população inicial...")
        np.random.seed(73)  # Garante que a população inicial seja sempre a mesma
        initial_population = [np.random.uniform(0, 1, 6) for _ in range(pop_size)]
        save_initial_population(initial_population, initial_population_path)

    # Configuração inicial ou carregamento do checkpoint
    if checkpoint:
        print("Carregando checkpoint...")
        start_generation = checkpoint['generation']
        population = checkpoint['population']
        mse_history = checkpoint['mse_history']
        accuracy_history = checkpoint['accuracy_history']
    else:
        print("Iniciando novo treinamento a partir da população inicial...")
        start_generation = 0
        population = initial_population
        mse_history, accuracy_history = [], []

    def fitness(individual, dataset):
        predictions = [fuzzy_system(row['pctid'], row['y'], individual) for _, row in dataset.iterrows()]
        mse = mean_squared_error(dataset['y'], predictions)
        accuracy = accuracy_score(np.round(dataset['y']), np.round(predictions))
        return mse, accuracy

    for generation in tqdm(range(start_generation, generations), desc="Gerações", leave=True):
        scores = [fitness(ind, train_data) for ind in population]
        mse_scores = [score[0] for score in scores]
        accuracy_scores = [score[1] for score in scores]

        # Melhor desempenho atual
        best_mse = min(mse_scores)
        best_accuracy = max(accuracy_scores)
        mse_history.append(best_mse)
        accuracy_history.append(best_accuracy)

        # Notificar checkpoints
        if generation > 0 and generation % 25 == 0:
            save_checkpoint(generation, population, mse_history, accuracy_history, checkpoint_path)

        # Seleção
        selected = [ind for _, ind in sorted(zip(mse_scores, population), key=lambda x: x[0])[:pop_size // 2]]

        # Crossover 50%
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            split = len(parent1) // 2
            child = np.concatenate((parent1[:split], parent2[split:]))
            # Mutação
            if random.random() < mutation_rate:
                child[random.randint(0, len(child) - 1)] += random.uniform(-0.1, 0.1)
            new_population.append(np.clip(child, 0, 1))

        population = new_population

    # Métricas finais
    best_individual = min(population, key=lambda ind: fitness(ind, val_data)[0])
    val_predictions = [fuzzy_system(row['pctid'], row['y'], best_individual) for _, row in val_data.iterrows()]
    mse = mean_squared_error(val_data['y'], val_predictions)
    f1 = f1_score(np.round(val_data['y']), np.round(val_predictions))
    accuracy = accuracy_score(np.round(val_data['y']), np.round(val_predictions))

    # Gráficos
    plot_path = f'src/GaFuzzy/images/mse_vs_generations{generation}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(mse_history, label='MSE', marker='o')
    plt.xlabel('Geração')
    plt.ylabel('MSE')
    plt.title('MSE vs Gerações')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

    accuracy_plot_path = f'src/GaFuzzy/images/accuracy_vs_generations{generation}.png'
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_history, label='Accuracy', marker='x')
    plt.xlabel('Geração')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Gerações')
    plt.legend()
    plt.grid(True)
    plt.savefig(accuracy_plot_path)
    plt.close()

    return best_individual, mse_history, accuracy_history

# Executar o treinamento
best_params, mse_history, accuracy_history = genetic_algorithm(train_data, val_data)
