import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

# Definir a semente randômica
SEED = 73
np.random.seed(SEED)
random.seed(SEED)

# Carregar o dataset
file_path = 'datasets/data/processed_teste_train_config_1_database.csv'
data = pd.read_csv(file_path)

# Caminho para a população inicial fixa
initial_population_path = 'src/GaFuzzy/initial_population.pkl'

# Função para carregar a população inicial fixa
def load_initial_population(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Arquivo de população inicial não encontrado em: {file_path}")

# Classe do Algoritmo Genético com K-Fold Cross-validation
class GeneticAlgorithmFuzzyKFold:
    def __init__(self, data, pop_size=50, generations=10, mutation_rate=0.1, k_folds=5, initial_population=None):
        self.data = data
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.k_folds = k_folds
        self.population = initial_population if initial_population is not None else [
            np.random.uniform(0, 1, 6) for _ in range(pop_size)
        ]
        self.results = {}

    def gaussmf(self, x, c, sigma, epsilon=1e-6):
        sigma = max(sigma, epsilon)
        return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

    def fuzzy_system(self, pctid, y, params):
        if 20 <= pctid <= 40:
            low_pctid, medium_pctid, high_pctid = 1, 0, 0
        elif 40 < pctid <= 60:
            low_pctid, medium_pctid, high_pctid = 0, 1, 0
        elif 60 < pctid <= 100:
            low_pctid, medium_pctid, high_pctid = 0, 0, 1

        normal_y = self.gaussmf(y, params[0], params[1])
        deviating_y = self.gaussmf(y, params[2], params[3])
        anomalous_y = self.gaussmf(y, params[4], params[5])

        low_anomaly = np.fmin(low_pctid, normal_y)
        medium_anomaly = np.fmin(medium_pctid, deviating_y)
        high_anomaly = np.fmin(high_pctid, anomalous_y)

        numerator = np.sum(low_anomaly * 0.3 + medium_anomaly * 0.6 + high_anomaly * 1.0)
        denominator = np.sum(low_anomaly + medium_anomaly + high_anomaly) + 1e-6
        return numerator / denominator

    def fitness(self, individual, dataset):
        predictions = [
            self.fuzzy_system(row['pctid'], row['y'], individual)
            for _, row in dataset.iterrows()
        ]
        mse = mean_squared_error(dataset['y'], predictions)
        accuracy = accuracy_score(np.round(dataset['y']), np.round(predictions))
        f1 = f1_score(np.round(dataset['y']), np.round(predictions), zero_division=1)
        return mse, accuracy, f1

    def coordinated_mutation(self, individual):
        for i in range(0, len(individual), 2):  # Assume que C e S são pares consecutivos
            if random.random() < self.mutation_rate:
                delta_c = random.uniform(-0.1, 0.1)
                delta_s = random.uniform(-0.05, 0.05)
                individual[i] = np.clip(individual[i] + delta_c, 0, 1)  # Mutação no centro (C)
                individual[i + 1] = np.clip(individual[i + 1] + delta_s, 1e-6, 1)  # Mutação no sigma (S)
        return individual

    def run(self):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=SEED)

        mse_history_val, mse_history_test = [], []
        accuracy_history_val, accuracy_history_test = [], []
        f1_history_val, f1_history_test = [], []

        for generation in tqdm(range(self.generations), desc="Gerações"):
            fold_mse_val, fold_accuracy_val, fold_f1_val = [], [], []
            fold_mse_test, fold_accuracy_test, fold_f1_test = [], [], []

            for train_index, val_index in kf.split(self.data):
                train_data = self.data.iloc[train_index]
                val_data = self.data.iloc[val_index]
                test_data = self.data.iloc[val_index]

                scores = [self.fitness(ind, train_data) for ind in self.population]
                val_scores = [self.fitness(ind, val_data) for ind in self.population]
                test_scores = [self.fitness(ind, test_data) for ind in self.population]

                fold_mse_val.append(min([score[0] for score in val_scores]))
                fold_accuracy_val.append(max([score[1] for score in val_scores]))
                fold_f1_val.append(max([score[2] for score in val_scores]))

                fold_mse_test.append(min([score[0] for score in test_scores]))
                fold_accuracy_test.append(max([score[1] for score in test_scores]))
                fold_f1_test.append(max([score[2] for score in test_scores]))

            mse_history_val.append(np.mean(fold_mse_val))
            accuracy_history_val.append(np.mean(fold_accuracy_val))
            f1_history_val.append(np.mean(fold_f1_val))

            mse_history_test.append(np.mean(fold_mse_test))
            accuracy_history_test.append(np.mean(fold_accuracy_test))
            f1_history_test.append(np.mean(fold_f1_test))

        self.results = {
            "mse_val": mse_history_val,
            "mse_test": mse_history_test,
            "accuracy_val": accuracy_history_val,
            "accuracy_test": accuracy_history_test,
            "f1_val": f1_history_val,
            "f1_test": f1_history_test,
        }
        return self.results

    def plot_results(self, generations):
        os.makedirs('src/GaFuzzy/images', exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # MSE
        axes[0].plot(self.results["mse_val"], label='MSE Validação')
        axes[0].plot(self.results["mse_test"], label='MSE Teste')
        axes[0].set_xlabel('Geração')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('MSE vs Gerações')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(self.results["accuracy_val"], label='Accuracy Validação')
        axes[1].plot(self.results["accuracy_test"], label='Accuracy Teste')
        axes[1].set_xlabel('Geração')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy vs Gerações')
        axes[1].legend()
        axes[1].grid(True)

        # F1 Score
        axes[2].plot(self.results["f1_val"], label='F1 Validação')
        axes[2].plot(self.results["f1_test"], label='F1 Teste')
        axes[2].set_xlabel('Geração')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('F1 Score vs Gerações')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f'src/GaFuzzy/images/results_kfold_{generations}.png')
        plt.close()

# Executar múltiplas simulações
generation_list = [11, 51, 101, 151, 201, 251, 301]
initial_population = load_initial_population(initial_population_path)

for generations in generation_list:
    print(f"Rodando simulação com {generations} gerações...")
    ga = GeneticAlgorithmFuzzyKFold(data, generations=generations, initial_population=initial_population)
    results = ga.run()
    ga.plot_results(generations)
