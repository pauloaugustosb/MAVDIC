import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import pickle  # Para salvar o modelo

# Classe CLONALG otimizada para evitar overfitting
class Clonalg:
    def __init__(self, train_data, clone_factor, mutation_rate, num_generations, threshold, patience=5, population_size=30):
        # Inicialização dos parâmetros do CLONALG
        self.normal_data = train_data
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.threshold = threshold
        self.population_size = population_size
        self.problem_size = train_data.shape[1]
        self.patience = patience
        self.population = [self.random_solution() for _ in range(self.population_size)]

    def random_solution(self):
        # Criação de uma solução aleatória dentro dos limites dos dados
        min_vals = np.min(self.normal_data, axis=0)
        max_vals = np.max(self.normal_data, axis=0)
        min_vals = np.clip(min_vals - 0.1, -1e6, 1e6)
        max_vals = np.clip(max_vals + 0.1, -1e6, 1e6)
        return np.random.uniform(min_vals, max_vals)

    def fitness(self, solution):
        # Calcula a aptidão da solução com base na distância média dos dados normais
        distances = np.linalg.norm(self.normal_data - solution, axis=1)
        return np.mean(distances)

    def clone_and_mutate(self, solution):
        # Gera clones e aplica mutação adaptativa
        clones = [solution.copy() for _ in range(self.clone_factor)]
        for i, clone in enumerate(clones):
            adaptive_mutation_rate = self.mutation_rate * (1 + (i / len(clones)))
            mutation = np.random.normal(0, adaptive_mutation_rate, size=self.problem_size)
            clone += mutation
        return clones

    def select(self):
        # Seleciona as melhores soluções da população atual
        self.population.sort(key=self.fitness)
        unique_solutions = []
        for sol in self.population:
            if len(unique_solutions) == 0 or not any(np.array_equal(sol, u) for u in unique_solutions):
                unique_solutions.append(sol)
            if len(unique_solutions) >= self.population_size:
                break
        return unique_solutions

    def detect_anomalies(self, test_data):
        # Detecta anomalias com base no threshold
        predictions = []
        for sample in test_data:
            is_anomaly = self.fitness(sample) > self.threshold
            predictions.append(is_anomaly)
        return predictions

    def cross_validate(self, k=3, subset_size=5000):
        # Realiza validação cruzada para avaliar a generalização do modelo
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        validation_scores = []

        for train_index, val_index in kf.split(self.normal_data[:subset_size]):
            train_fold = self.normal_data[train_index]
            val_fold = self.normal_data[val_index]

            clonalg_fold = Clonalg(train_fold, self.clone_factor, self.mutation_rate,
                                   self.num_generations, self.threshold, self.patience, self.population_size)
            _, _, val_errors = clonalg_fold.evolve(val_fold)

            validation_scores.append(np.mean(val_errors))

        avg_validation_score = np.mean(validation_scores)
        print(f"Média dos erros de validação após cross-validation: {avg_validation_score:.4f}")

        return avg_validation_score

    def evolve(self, val_data, sample_size=1000):
        # Evolui a população ao longo das gerações e coleta erros de treinamento e validação
        fitness_progress = []
        training_errors = []
        validation_errors = []

        for generation in range(self.num_generations):
            new_population = []
            for solution in self.population:
                clones = self.clone_and_mutate(solution)
                new_population.extend(clones)

            self.population = self.select()
            best_solution = self.population[0]
            best_fitness = self.fitness(best_solution)
            fitness_progress.append(best_fitness)

            train_sample = random.sample(list(self.normal_data), min(sample_size, len(self.normal_data)))
            val_sample = random.sample(list(val_data), min(sample_size, len(val_data)))

            train_errors = [self.fitness(sample) for sample in train_sample]
            val_errors = [self.fitness(sample) for sample in val_sample]

            training_errors.append(np.mean(train_errors))
            validation_errors.append(np.mean(val_errors))

            if generation > self.patience:
                recent_errors = validation_errors[-self.patience:]
                if all(recent_errors[i] >= recent_errors[i - 1] for i in range(1, self.patience)):
                    print("Early stopping acionado.")
                    break

        return fitness_progress, training_errors, validation_errors

# Função para encontrar o melhor threshold com base no F1-score
def find_best_threshold(true_labels, predicted_scores, thresholds):
    best_threshold = 0.5
    best_f1 = 0
    f1_scores = []

    for threshold in thresholds:
        predictions = [1 if score > threshold else 0 for score in predicted_scores]
        f1 = f1_score(true_labels, predictions, zero_division=0)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1, f1_scores

# Função para carregar e dividir os dados
def load_and_split_data(data, train_size, val_size, columns):
    data = data[columns].astype(float)
    train_data = data[:train_size].values
    val_data = data[train_size:train_size + val_size].values
    test_data = data[train_size + val_size:].values
    return train_data, val_data, test_data

# Carregando os dados de treino e teste
file_path = 'datasets/data/processed_teste_train_config_1_database.csv'
data = pd.read_csv(file_path)
columns_of_interest = ['y']
train_size = 30000
val_size = 5000

train_data, val_data, test_data = load_and_split_data(data, train_size, val_size, columns_of_interest)

# Carregando a base de validação
validation_file_path = 'datasets/data/processed_validate_config_2_database.csv'
validation_data = pd.read_csv(validation_file_path)
validation_data = validation_data[columns_of_interest].astype(float).values

# Parâmetros CLONALG
clone_factor = 20
mutation_rate = 0.05
num_generations_reduced = 20
population_size_reduced = 20
threshold = 0.5

# Treinando o modelo com validação cruzada
clonalg_cv_optimized = Clonalg(train_data, clone_factor, mutation_rate,
                               num_generations_reduced, threshold, population_size=population_size_reduced)
avg_validation_score = clonalg_cv_optimized.cross_validate(k=3)

# Avaliando a base de validação
predicted_scores_validation = [clonalg_cv_optimized.fitness(sample) for sample in validation_data]
true_labels_validation = [1 if i >= len(validation_data) - 5 else 0 for i in range(len(validation_data))]
thresholds = np.linspace(0, 1, 100)
best_threshold, best_f1, f1_scores = find_best_threshold(true_labels_validation, predicted_scores_validation, thresholds)

# Calculando métricas de validação
test_predictions_validation = [1 if score > best_threshold else 0 for score in predicted_scores_validation]
accuracy_validation = accuracy_score(true_labels_validation, test_predictions_validation)
precision_validation = precision_score(true_labels_validation, test_predictions_validation, zero_division=0)
recall_validation = recall_score(true_labels_validation, test_predictions_validation, zero_division=0)
f1_validation = f1_score(true_labels_validation, test_predictions_validation, zero_division=0)

fpr_validation, tpr_validation, _ = roc_curve(true_labels_validation, predicted_scores_validation)
roc_auc_validation = auc(fpr_validation, tpr_validation)

# Exibindo métricas
print(f"Melhor threshold na validação: {best_threshold:.2f}")
print(f"Acurácia na validação: {accuracy_validation:.2f}")
print(f"Precisão na validação: {precision_validation:.2f}")
print(f"Revocação na validação: {recall_validation:.2f}")
print(f"F1-Score na validação: {f1_validation:.2f}")
print(f"AUC na validação: {roc_auc_validation:.2f}")

# Treinando o modelo e coletando métricas
fitness_progress, training_errors, validation_errors = clonalg_cv_optimized.evolve(val_data)

# Salvando o modelo treinado
with open('CLONALG_DETECANOM.pkl', 'wb') as model_file:
    pickle.dump(clonalg_cv_optimized, model_file)

# Plotando gráficos de aprendizagem, curva ROC e F1-score vs threshold
plt.figure(figsize=(18, 10))

# Plot de Erro de Treinamento vs Validação
plt.subplot(2, 2, 1)
plt.plot(training_errors, label='Erro de Treinamento', marker='o', linestyle='-')
plt.plot(validation_errors, label='Erro de Validação', marker='o', linestyle='--')
plt.title('Erro de Treinamento vs Validação')
plt.xlabel('Geração')
plt.ylabel('Erro Médio')
plt.legend()
plt.grid(True)

# Plot do Progresso do Fitness
plt.subplot(2, 2, 2)
plt.plot(fitness_progress, label='Progresso do Fitness', marker='o', linestyle='-')
plt.title('Progresso do Fitness ao Longo das Gerações')
plt.xlabel('Geração')
plt.ylabel('Melhor Fitness')
plt.legend()
plt.grid(True)

# Plot da Curva ROC
plt.subplot(2, 2, 3)
plt.plot(fpr_validation, tpr_validation, label=f'ROC Curve (AUC = {roc_auc_validation:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('Curva ROC na Base de Validação')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.legend()
plt.grid(True)

# Plot de F1-Score vs Threshold
plt.subplot(2, 2, 4)
plt.plot(thresholds, f1_scores, marker='o', linestyle='-', label='F1-Score')
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Threshold Ideal: {best_threshold:.2f}')
plt.title('F1-Score em Função do Threshold na Validação')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
