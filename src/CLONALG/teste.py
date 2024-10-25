import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Classe ajustada para incluir normalização e plotagem
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
        self.fitness_history = []  # Para armazenar o fitness ao longo das gerações

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
        for index, sample in enumerate(test_data):
            if self.fitness(sample) > self.threshold:
                anomalies.append((index, sample))
        return anomalies

    def evolve(self):
        # Executa o ciclo de clonagem e seleção
        for generation in range(self.num_generations):
            new_population = []
            for solution in self.population:
                clones = self.clone_and_mutate(solution)
                new_population.extend(clones)

            self.population = self.select()
            best_fitness = self.fitness(self.population[0])
            self.fitness_history.append(best_fitness)
            
# Função para carregar os dados e normalizá-los
def load_and_normalize_data(file_path, train_size):
    data = pd.read_csv(file_path)
    data = data[['y', 'z']].dropna().values  # Filtra as colunas 'y' e 'z'
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Exemplo de uso do arquivo fictício
file_path = '/mnt/data/dataset_config_1.csv'  # Substituir com o caminho correto do dataset
train_size = 21001

# Carrega e normaliza os dados
train_data, test_data = load_and_normalize_data(file_path, train_size)

# Configura parâmetros
clone_factor = 100
mutation_rate = 0.05
num_generations = 20
threshold = 0.5

# Inicializa o modelo
clonalg = ClonalgAnomalyDetection(train_data, clone_factor, mutation_rate, num_generations, threshold)
clonalg.evolve()
anomalies = clonalg.detect_anomalies(test_data)

# Plotagem dos dados 'y' vs 'z'
plt.figure(figsize=(12, 6))
plt.scatter(train_data[:, 0], train_data[:, 1], label='Normal Data', alpha=0.5, color='blue')
plt.scatter([a[1][0] for a in anomalies], [a[1][1] for a in anomalies], label='Anomalies', color='red', marker='x')
plt.xlabel("Eixo Y")
plt.ylabel("Eixo Z")
plt.title("Visualização de Dados Normal e Anomalias (Y vs Z)")
plt.legend()

# Plotagem do progresso do fitness durante o treinamento
plt.figure(figsize=(12, 6))
plt.plot(clonalg.fitness_history, label='Fitness do Melhor Individuo')
plt.xlabel("Gerações")
plt.ylabel("Fitness (Menor Distância)")
plt.title("Evolução do Fitness ao Longo das Gerações")
plt.legend()

plt.show()
