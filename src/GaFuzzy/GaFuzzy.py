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

# Função para mutação coordenada
def coordinated_mutation(individual, mutation_rate=0.2, epsilon=1e-6):
    for i in range(0, len(individual), 2):  # Assume que C e S são pares consecutivos
        if random.random() < mutation_rate:
            delta_c = random.uniform(-0.1, 0.1)
            delta_s = random.uniform(-0.05, 0.05)
            individual[i] = np.clip(individual[i] + delta_c, 0, 1)  # Mutação no centro (C)
            individual[i + 1] = np.clip(individual[i + 1] + delta_s, epsilon, 1)  # Mutação no sigma (S)
    return individual

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

# Função para salvar gráficos
def save_graph(history, ylabel, title, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(history, marker='o')
    plt.xlabel('Geração')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

# Algoritmo Genético
def genetic_algorithm(train_data, val_data, pop_size=50, mutation_rate=0.2, generations=10, checkpoint_path='src/GaFuzzy/checkpoint.pkl'):
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint:
        print("Carregando checkpoint...")
        start_generation = checkpoint['generation']
        population = checkpoint['population']
        mse_history = checkpoint['mse_history']
        accuracy_history = checkpoint['accuracy_history']
    else:
        print("Iniciando novo treinamento...")
        start_generation = 0
        population = [np.random.uniform(0, 1, 6) for _ in range(pop_size)]
        mse_history, accuracy_history = [], []

    def fitness(individual, dataset):
        predictions = [fuzzy_system(row['pctid'], row['y'], individual) for _, row in dataset.iterrows()]
        mse = mean_squared_error(dataset['y'], predictions)
        accuracy = accuracy_score(np.round(dataset['y']), np.round(predictions))
        return mse, accuracy

    for generation in tqdm(range(start_generation, start_generation + generations), desc="Gerações", leave=True):
        scores = [fitness(ind, train_data) for ind in population]
        mse_scores = [score[0] for score in scores]
        accuracy_scores = [score[1] for score in scores]

        # Melhor desempenho atual
        best_mse = min(mse_scores)
        best_accuracy = max(accuracy_scores)
        mse_history.append(best_mse)
        accuracy_history.append(best_accuracy)

        # Checkpoints
        if generation > 0 and generation % 25 == 0:
            save_checkpoint(generation, population, mse_history, accuracy_history, checkpoint_path)

        # Seleção
        top_half = sorted(zip(mse_scores, population), key=lambda x: x[0])[:pop_size // 2]
        best_population = [ind for _, ind in top_half]

        # Crossover elitista
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(best_population, 2)
            split = len(parent1) // 2
            child = np.concatenate((parent1[:split], parent2[split:]))
            new_population.append(coordinated_mutation(child, mutation_rate))

        # Mutação nos 50% piores
        worst_half = sorted(zip(mse_scores, population), key=lambda x: x[0])[pop_size // 2:]
        for _, ind in worst_half:
            ind = coordinated_mutation(ind, mutation_rate)

        population = best_population + new_population[:pop_size // 2]

    return population, mse_history, accuracy_history

# Execução do treinamento em etapas
for step, generations in enumerate([10, 25, 50], start=1):
    print(f"Treinamento - Etapa {step}: {generations} Gerações")
    final_population, mse_history, accuracy_history = genetic_algorithm(
        train_data, val_data, pop_size=50, mutation_rate=0.2, generations=generations
    )

    # Salvar gráficos
    save_graph(mse_history, 'MSE', f'MSE vs Gerações (Etapa {step})', f'src/GaFuzzy/images/mse_vs_generations_step{step}.png')
    save_graph(accuracy_history, 'Accuracy', f'Accuracy vs Gerações (Etapa {step})', f'src/GaFuzzy/images/accuracy_vs_generations_step{step}.png')

    # Imprimir métricas
    print(f"Etapa {step} - Resultados:")
    print(f"🔹 Melhor MSE: {mse_history[-1]:.4f}")
    print(f"🔹 Melhor Accuracy: {accuracy_history[-1]:.4f}")
