import numpy as np

class ClonalgAnomalyDetection:
    def __init__(self, normal_data, clone_factor, mutation_rate, num_generations, threshold):
        self.normal_data = normal_data
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.threshold = threshold  # Limiar para detectar anomalias
        self.population_size = len(normal_data)
        self.problem_size = normal_data.shape[1]
        self.population = [self.random_solution() for _ in range(self.population_size)]

    def random_solution(self):
        """Cria uma solução aleatória dentro do intervalo dos dados normais."""
        min_vals = np.min(self.normal_data, axis=0)
        max_vals = np.max(self.normal_data, axis=0)
        return np.random.uniform(min_vals, max_vals)

    def fitness(self, solution):
        """Calcula a distância mínima da solução para os dados normais."""
        distances = np.linalg.norm(self.normal_data - solution, axis=1)
        return np.min(distances)  # Usamos a menor distância como medida de anomalia

    def clone_and_mutate(self, solution):
        """Clona e aplica mutação ao clone."""
        clones = [solution.copy() for _ in range(self.clone_factor)]
        for clone in clones:
            mutation = np.random.uniform(-1, 1, size=self.problem_size) * self.mutation_rate
            clone += mutation
        return clones

    def select(self):
        """Seleciona as melhores soluções com base no fitness (menor distância)."""
        self.population.sort(key=self.fitness)
        return self.population[:self.population_size]

    def detect_anomalies(self, data):
        """Detecta anomalias em um conjunto de dados."""
        anomalies = []
        for sample in data:
            if self.fitness(sample) > self.threshold:
                anomalies.append(sample)
        return anomalies

    def evolve(self):
        """Evolui a população, mantendo as melhores soluções a cada geração."""
        for generation in range(self.num_generations):
            # Clonagem e mutação
            new_population = []
            for solution in self.population:
                clones = self.clone_and_mutate(solution)
                new_population.extend(clones)
            
            # Avaliação e seleção
            self.population = self.select()
            print(f"Geração {generation+1}, Melhor fitness (menor distância): {self.fitness(self.population[0])}")

        # Retorna as soluções finais
        return self.population

# Exemplo de uso para detecção de anomalias
np.random.seed(42)  # Para reprodutibilidade

# Geração de um conjunto de dados normal (padrões normais)
normal_data = np.random.normal(loc=0, scale=1, size=(100, 5))

# Dados com possíveis anomalias
test_data = np.vstack([
    np.random.normal(loc=0, scale=1, size=(95, 5)),  # Dados normais
    np.random.normal(loc=5, scale=1, size=(5, 5))    # Anomalias
])

# Parâmetros do algoritmo
clone_factor = 5
mutation_rate = 0.1
num_generations = 50
threshold = 2.0  # Limiar de distância para considerar uma anomalia

# Execução do CLONALG para detecção de anomalias
clonalg = ClonalgAnomalyDetection(normal_data, clone_factor, mutation_rate, num_generations, threshold)
clonalg.evolve()

# Detectando anomalias
anomalies = clonalg.detect_anomalies(test_data)
print("Anomalias detectadas:", anomalies)
