import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ClonalgAnomalyDetection:
    def __init__(self, train_data, clone_factor, mutation_rate, num_generations, threshold):
        # Inicializa dados de treino e parâmetros do algoritmo
        self.normal_data = train_data
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.threshold = threshold
        self.population_size = 50
        self.problem_size = train_data.shape[1]
        self.population = [self.random_solution() for _ in range(self.population_size)]

    def random_solution(self):
        # Gera uma solução aleatória com base nos dados de treino
        min_vals = np.min(self.normal_data, axis=0)
        max_vals = np.max(self.normal_data, axis=0)
        return np.random.uniform(min_vals, max_vals)

    def fitness(self, solution):
        # Calcula a menor distância entre uma solução e os dados de treino
        distances = np.linalg.norm(self.normal_data - solution, axis=1)
        return np.min(distances)

    def clone_and_mutate(self, solution):
        # Clona e aplica mutação em uma solução
        clones = [solution.copy() for _ in range(self.clone_factor)]
        for clone in clones:
            mutation = np.random.uniform(-1, 1, size=self.problem_size) * self.mutation_rate
            clone += mutation
        return clones

    def select(self):
        # Seleciona as melhores soluções
        self.population.sort(key=self.fitness)
        return self.population[:self.population_size]

    def detect_anomalies(self, test_data):
        # Detecta anomalias e calcula a acurácia
        anomalies = []
        first_anomaly = None
        first_anomaly_index = None
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        for index, sample in enumerate(test_data):
            is_anomaly = self.fitness(sample) > self.threshold
            actual_is_anomaly = index >= len(test_data) - 5  # Assume que as últimas 5 amostras são anomalias

            if is_anomaly:
                anomalies.append((index, sample))
                if actual_is_anomaly:
                    true_positive += 1
                    if first_anomaly is None:
                        first_anomaly = sample
                        first_anomaly_index = index
                else:
                    false_positive += 1
            else:
                if actual_is_anomaly:
                    false_negative += 1
                else:
                    true_negative += 1

        # Cálculo da acurácia
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        
        return anomalies, first_anomaly, first_anomaly_index, accuracy

    def evolve(self):
        # Executa o ciclo de clonagem e seleção
        for generation in range(self.num_generations):
            new_population = []
            for solution in self.population:
                clones = self.clone_and_mutate(solution)
                new_population.extend(clones)

            self.population = self.select()
            print(f"Geração {generation + 1}, Melhor fitness (menor distância): {self.fitness(self.population[0])}")

def load_and_normalize_data(file_path, train_size, columns):
    # Carrega os dados do arquivo CSV e normaliza
    data = pd.read_csv(file_path)
    
    # Converte todos os dados para numéricos, forçando erros para NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # Remove colunas que contêm NaN
    data = data.dropna(axis=1)

    # Normaliza os dados das colunas especificadas
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data[columns])

    # Converte para um array NumPy
    normalized_data = normalized_data
    
    train_data = normalized_data[:train_size]
    test_data = normalized_data[train_size:]
    
    return train_data, test_data

# Exemplo de uso:

# Caminho do arquivo CSV e proporção dos dados de treino
file_path = 'datasets/dataset_config_1.csv'
train_size = 21001  # Define os primeiros 21001 registros como treino

# Definindo as colunas de interesse (y e z)
columns_of_interest = ['y']  

# Carrega e normaliza os dados
train_data, test_data = load_and_normalize_data(file_path, train_size, columns_of_interest)

# Parâmetros do algoritmo
clone_factor = 1000
mutation_rate = 0.5
num_generations = 1000
threshold = 0.5  # Distância limite para considerar uma amostra anômala (ajuste conforme necessário)

# Inicializa o CLONALG
clonalg = ClonalgAnomalyDetection(train_data, clone_factor, mutation_rate, num_generations, threshold)

# Evolução do algoritmo
clonalg.evolve()

# Detecção de anomalias
anomalies, first_anomaly, first_anomaly_index, accuracy = clonalg.detect_anomalies(test_data)

# Exibe os resultados
print(f"Acurácia do modelo: {accuracy:.2f}")
