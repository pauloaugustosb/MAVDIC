import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm

# Configurar a semente aleatória
SEED = 73
np.random.seed(SEED)

# Carregar o dataset
file_path = 'data/rms_results_training.csv' 
data = pd.read_csv(file_path)

# Separar as features e a variável-alvo
X = data[['RMS Amplitude', 'RMS Frequency']].values  # Features
y = data['Classe'].values  # Variável-alvo

# Codificar as classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir os dados em treinamento (70%) e validação (30%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=SEED)

# Configurar CLONALG para otimizar hiperparâmetros
class CLONALG:
    def __init__(self, pop_size, clone_rate, mutation_rate, generations, population_file="scripts/models/SVMCLONALG/pop_initial_SVM_Clonalg.pkl"):
        self.pop_size = pop_size
        self.clone_rate = clone_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population_file = population_file

    def initialize_population(self):
        if os.path.exists(self.population_file):
            with open(self.population_file, "rb") as f:
                population = pickle.load(f)
            print(f"População inicial carregada de {self.population_file}")
        else:
            population = [
                {
                    'C': np.random.uniform(0.001, 1000),
                    'gamma': np.random.uniform(0.0001, 1.0)
                } for _ in range(self.pop_size)
            ]
            with open(self.population_file, "wb") as f:
                pickle.dump(population, f)
            print(f"População inicial salva em {self.population_file}")
        return population

    def clone_and_mutate(self, individual):
        clones = [
            {
                'C': np.clip(individual['C'] + np.random.normal(0, self.mutation_rate), 0.001, 1000),
                'gamma': np.clip(individual['gamma'] + np.random.normal(0, self.mutation_rate), 0.0001, 1.0)
            } for _ in range(int(self.clone_rate))
        ]
        return clones

    def fitness(self, individual, X_train, y_train, X_val, y_val):
        model = SVC(kernel='rbf', C=individual['C'], gamma=individual['gamma'], random_state=SEED)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        return f1_score(y_val, y_val_pred, average='weighted')

    def evolve(self, X_train_val, y_train_val):
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        population = self.initialize_population()
        best_individual = None
        best_f1 = 0

        for generation in tqdm(range(self.generations), desc="Evolução das Gerações"):
            fitness_scores = []

            for individual in tqdm(population, desc=f"Avaliação da População na Geração {generation + 1}", leave=False):
                fold_scores = []

                for train_index, val_index in kf.split(X_train_val):
                    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
                    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

                    score = self.fitness(individual, X_train, y_train, X_val, y_val)
                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)
                fitness_scores.append((avg_score, individual))

            fitness_scores.sort(key=lambda x: -x[0])  # Ordenar por melhor F1 Score
            best_f1 = fitness_scores[0][0]
            best_individual = fitness_scores[0][1]

            print(f"Geração {generation + 1}: Melhor F1 Score = {best_f1:.4f}, C = {best_individual['C']:.4f}, gamma = {best_individual['gamma']:.4f}")

            # Selecionar os melhores indivíduos para clonar
            top_individuals = [ind for _, ind in fitness_scores[:self.pop_size // 2]]
            population = top_individuals.copy()

            # Clonar e mutar
            for ind in top_individuals:
                population.extend(self.clone_and_mutate(ind))

            # Reduzir para tamanho fixo
            population = population[:self.pop_size]

        return best_individual, best_f1

# Executar CLONALG
clonalg = CLONALG(pop_size=50, clone_rate=5, mutation_rate=0.1, generations=10)
best_params, best_f1 = clonalg.evolve(X_train_val, y_train_val)

print(f"Melhores parâmetros encontrados: C = {best_params['C']}, gamma = {best_params['gamma']}")

# Treinar modelo final com os melhores parâmetros
final_model = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], random_state=SEED)
final_model.fit(X_train_val, y_train_val)

# Avaliar o modelo no conjunto de teste
y_test_pred = final_model.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_test_pred)

print(f"F1 Score no Teste com os melhores parâmetros: {test_f1:.4f}")

# Matriz de Confusão no Teste
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matriz de Confusão - SVM com CLONALG")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig('svm_clonalg_confusion_matrix_no_scaling.png')
plt.show()

# Plotar RMS Amplitude Ordenado com Classes Reais e Preditas
sorted_indices = np.argsort(data['RMS Amplitude'].values)
data['Predicted Classe'] = label_encoder.inverse_transform(final_model.predict(X))

plt.figure(figsize=(12, 6))
plt.plot(
    np.arange(len(data)),
    data['RMS Amplitude'].values[sorted_indices],
    label='RMS Amplitude Ordenado',
    color='green'
)
plt.scatter(
    np.arange(len(data)),
    data['RMS Amplitude'].values[sorted_indices],
    c=label_encoder.transform(data['Predicted Classe'].values[sorted_indices]),
    cmap='viridis',
    label='Predições',
    alpha=0.6
)
plt.title("RMS Amplitude Ordenado com Predições - SVM")
plt.xlabel("Observações")
plt.ylabel("RMS Amplitude")
plt.legend()
cb = plt.colorbar()
cb.set_label('Classes')
cb.set_ticks(range(len(label_encoder.classes_)))
cb.set_ticklabels(label_encoder.classes_)
plt.grid(True)
plt.savefig('svm_classification_results.png')
plt.show()

# Salvar o modelo treinado
model_path = 'scripts/models/SVMCLONALG/svm_clonalg_model.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(final_model, model_file)
print(f"Modelo SVM com CLONALG salvo em: {model_path}")

# Salvar o Label Encoder
encoder_path = 'scripts/models/SVMCLONALG/label_encoder.pkl'
with open(encoder_path, 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)
print(f"Label Encoder salvo em: {encoder_path}")

# Salvar a população inicial
population_path = 'scripts/models/SVMCLONALG/pop_initial_SVM_Clonalg.pkl'
if not os.path.exists(population_path):
    with open(population_path, 'wb') as population_file:
        pickle.dump(clonalg.initialize_population(), population_file)
    print(f"População inicial salva em: {population_path}")
