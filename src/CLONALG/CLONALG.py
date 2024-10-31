import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

class ClonalgAnomalyDetection:
    def __init__(self, train_data, clone_factor, mutation_rate, num_generations, threshold):
        # Dados de treino e parâmetros principais do algoritmo CLONALG
        self.normal_data = train_data
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.threshold = threshold
        self.population_size = 50  # Número de soluções na população
        self.problem_size = train_data.shape[1]  # Dimensão dos dados
        # Inicializa a população com soluções aleatórias baseadas nos dados de treino
        self.population = [self.random_solution() for _ in range(self.population_size)]

    def random_solution(self):
        # Cria uma solução aleatória no intervalo dos valores de treino
        min_vals = np.min(self.normal_data, axis=0)
        max_vals = np.max(self.normal_data, axis=0)
        return np.random.uniform(min_vals, max_vals)

    def fitness(self, solution):
        # Calcula a menor distância entre a solução e os dados de treino (menor distância = melhor adequação)
        distances = np.linalg.norm(self.normal_data - solution, axis=1)
        return np.min(distances)

    def clone_and_mutate(self, solution):
        # Clona e realiza mutação em uma solução
        clones = [solution.copy() for _ in range(self.clone_factor)]
        for clone in clones:
            mutation = np.random.uniform(-1, 1, size=self.problem_size) * self.mutation_rate
            clone += mutation
        return clones

    def select(self):
        # Ordena as soluções pelo fitness (menor distância) e seleciona as melhores
        self.population.sort(key=self.fitness)
        return self.population[:self.population_size]

    def detect_anomalies(self, test_data):
        # Detecta anomalias no conjunto de teste e calcula a acurácia
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        for index, sample in enumerate(test_data):
            is_anomaly = self.fitness(sample) > self.threshold
            actual_is_anomaly = index >= len(test_data) - 5  # Assume as últimas 5 amostras como anomalias

            if is_anomaly:
                if actual_is_anomaly:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if actual_is_anomaly:
                    false_negative += 1
                else:
                    true_negative += 1

        # Cálculo da acurácia em porcentagem
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) * 100
        return accuracy

    def evolve(self):
        # Executa as etapas de clonagem e seleção por várias gerações
        fitness_progress = []
        for generation in range(self.num_generations):
            new_population = []
            for solution in self.population:
                clones = self.clone_and_mutate(solution)
                new_population.extend(clones)

            self.population = self.select()
            best_fitness = self.fitness(self.population[0])
            fitness_progress.append(best_fitness)
            print(f"Geração {generation + 1}, Melhor fitness (menor distância): {best_fitness}")
        
        # Retorna o progresso de fitness para plotar o desempenho
        return fitness_progress

def load_and_normalize_data(file_path, train_size, columns):
    # Carrega e normaliza dados a partir de um arquivo CSV
    data = pd.read_csv(file_path)
    data = data.apply(pd.to_numeric, errors='coerce')  # Converte para numérico, forçando erros para NaN
    data = data.dropna(axis=1)  # Remove colunas com NaN
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[columns])  # Normaliza as colunas de interesse
    train_data = normalized_data[:train_size]
    test_data = normalized_data[train_size:]
    return train_data, test_data



# Parâmetros principais do CLONALG
file_path = 'datasets/dataset_config_1.csv'
train_size = 21001
columns_of_interest = ['y']
train_data, test_data = load_and_normalize_data(file_path, train_size, columns_of_interest)

clone_factor = 10
mutation_rate = 0.1
num_generations = 50
threshold = 0.5



# Inicializa e evolui o CLONALG
clonalg = ClonalgAnomalyDetection(train_data, clone_factor, mutation_rate, num_generations, threshold)
fitness_progress = clonalg.evolve()



# Detecta anomalias e calcula a acurácia final
accuracy = clonalg.detect_anomalies(test_data)
print(f"Acurácia do modelo: {accuracy:.2f}%")



# Salva o modelo treinado para uso posterior
joblib.dump(clonalg, 'CLONALG.pkl')
print("Modelo CLONALG salvo como 'CLONALG.pkl'")

