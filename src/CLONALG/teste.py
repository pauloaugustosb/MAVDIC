import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Classe CLONALG para detecção de anomalias
class ClonalgAnomalyDetection:
    def __init__(self, train_data, clone_factor, mutation_rate, num_generations, threshold):
        self.normal_data = train_data
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.threshold = threshold
        self.population_size = 50
        self.problem_size = train_data.shape[1]
        self.population = [self.random_solution() for _ in range(self.population_size)]

    def random_solution(self):
        min_vals = np.min(self.normal_data, axis=0)
        max_vals = np.max(self.normal_data, axis=0)
        return np.random.uniform(min_vals, max_vals)

    def fitness(self, solution):
        distances = np.linalg.norm(self.normal_data - solution, axis=1)
        return np.min(distances)

    def clone_and_mutate(self, solution):
        clones = [solution.copy() for _ in range(self.clone_factor)]
        for clone in clones:
            mutation = np.random.uniform(-1, 1, size=self.problem_size) * self.mutation_rate
            clone += mutation
        return clones

    def select(self):
        self.population.sort(key=self.fitness)
        return self.population[:self.population_size]

    def detect_anomalies(self, test_data):
        true_positive = false_positive = false_negative = true_negative = 0
        is_anomaly_results = []

        for index, sample in enumerate(test_data):
            is_anomaly = self.fitness(sample) > self.threshold
            is_anomaly_results.append(is_anomaly)
            actual_is_anomaly = index >= len(test_data) - 5

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

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) * 100
        return accuracy, is_anomaly_results

    def evolve(self):
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
        
        return fitness_progress

def load_and_normalize_data_v2(data, train_size, columns):
    data = data[columns]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    train_data = normalized_data[:train_size]
    test_data = normalized_data[train_size:]
    return train_data, test_data, scaler

# Carregando e ajustando o conjunto de dados
file_path = 'C:/Users/paulo/OneDrive/Área de Trabalho/Faculdade/TCC/MAVDIC/datasets/processed_first_database.csv'
data = pd.read_csv(file_path)
columns_of_interest = ['y']
new_train_size = 40000

train_data_reduced, test_data, scaler = load_and_normalize_data_v2(data, new_train_size, columns_of_interest)

# Parâmetros CLONALG
clone_factor = 10
mutation_rate = 0.1
num_generations = 50
threshold = 0.5

clonalg_reduced = ClonalgAnomalyDetection(train_data_reduced, clone_factor, mutation_rate, num_generations, threshold)
fitness_progress_reduced = clonalg_reduced.evolve()

# Detectando anomalias e calculando a acurácia
accuracy_reduced, is_anomaly_results_reduced = clonalg_reduced.detect_anomalies(test_data)
print(f"Acurácia do modelo com conjunto reduzido: {accuracy_reduced:.2f}%")

# Transformação Fourier para suavizar a série temporal
fourier_transformed_data = np.fft.fft(test_data.ravel())
frequencies = np.fft.fftfreq(len(fourier_transformed_data))
# Mantendo apenas frequências principais para suavizar
threshold_frequency = 0.1
fourier_transformed_data[np.abs(frequencies) > threshold_frequency] = 0
smoothed_test_data = np.fft.ifft(fourier_transformed_data).real

# Visualizando a série temporal suavizada e anomalias detectadas
plt.figure(figsize=(12, 6))
plt.plot(smoothed_test_data, label='Dados de Teste Suavizados (Fourier)', color='blue')
plt.scatter(np.arange(len(smoothed_test_data))[is_anomaly_results_reduced], smoothed_test_data[is_anomaly_results_reduced],
            color='red', marker='x', label='Anomalias Detectadas')
plt.xlabel('Amostra')
plt.ylabel('Valor Normalizado Suavizado')
plt.legend()
plt.title('Detecção de Anomalias com CLONALG (Transformação Fourier)')
plt.grid(True)
plt.show()
