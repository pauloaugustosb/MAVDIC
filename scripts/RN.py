import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Configurar a semente aleatória
SEED = 73
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Carregar o dataset
file_path = 'data/rms_results_training.csv'
data = pd.read_csv(file_path)

# Separar as features e a variável-alvo
X = data[['RMS Amplitude', 'RMS Frequency']].values
y = data['Classe'].values

# Codificar as classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Dividir os dados em treinamento/validação (70%) e teste (30%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=SEED)

# Função para criar a Rede Neural
def create_model(params, input_dim, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(int(params[0]), activation='relu'))
    model.add(Dense(int(params[1]), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=params[2]), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# CLONALG para otimização de hiperparâmetros
class CLONALG:
    def __init__(self, pop_size, clone_rate, mutation_rate, generations, population_file="scripts/models/RN/pop_initial_RN_Clonalg.pkl"):
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
                np.array([np.random.randint(10, 100), np.random.randint(10, 100), np.random.uniform(0.0001, 0.01)])
                for _ in range(self.pop_size)
            ]
            with open(self.population_file, "wb") as f:
                pickle.dump(population, f)
            print(f"População inicial salva em {self.population_file}")
        return population

    def clone_and_mutate(self, individual):
        clones = [
            individual + np.random.normal(0, self.mutation_rate, size=len(individual)) for _ in range(int(self.clone_rate))
        ]
        return [np.clip(clone, [10, 10, 0.0001], [100, 100, 0.01]) for clone in clones]

    def fitness(self, individual, X_train_val, y_train_val):
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_scores = []

        for train_index, val_index in kf.split(X_train_val):
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            y_train, y_val = y_train_val[train_index], y_train_val[val_index]

            model = create_model(individual, input_dim=X_train.shape[1], num_classes=y_train.shape[1])
            model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
            y_val_pred = model.predict(X_val)
            score = f1_score(np.argmax(y_val, axis=1), np.argmax(y_val_pred, axis=1), average='weighted')
            fold_scores.append(score)

        return np.mean(fold_scores)

    def evolve(self, X_train_val, y_train_val):
        population = self.initialize_population()
        best_individual = None
        best_f1 = 0

        for generation in tqdm(range(self.generations), desc="Evolução das Gerações"):
            fitness_scores = []

            for individual in tqdm(population, desc=f"Avaliação da População na Geração {generation + 1}", leave=False):
                score = self.fitness(individual, X_train_val, y_train_val)
                fitness_scores.append((score, individual))

            fitness_scores.sort(key=lambda x: -x[0])
            best_f1 = fitness_scores[0][0]
            best_individual = fitness_scores[0][1]

            print(f"Geração {generation + 1}: Melhor F1 Score = {best_f1:.4f}")

            # Selecionar os melhores indivíduos
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

print(f"Melhores parâmetros encontrados: {best_params}")

# Treinar o modelo final com os melhores parâmetros
final_model = create_model(best_params, input_dim=X_train_val.shape[1], num_classes=y_train_val.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = final_model.fit(X_train_val, y_train_val, epochs=100, batch_size=16, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Fazer predições no conjunto de teste
y_test_pred = final_model.predict(X_test)
predicted_classes = label_encoder.inverse_transform(np.argmax(y_test_pred, axis=1))
real_classes = label_encoder.inverse_transform(np.argmax(y_test, axis=1))

# Avaliar o modelo
print("\nRelatório de Classificação no Conjunto de Teste:")
print(classification_report(real_classes, predicted_classes, target_names=label_encoder.classes_))

# Matriz de Confusão
conf_matrix = confusion_matrix(real_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matriz de Confusão - Rede Neural")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig('scripts/images/rn_confusion_matrix.png')
plt.show()

# Plotar RMS Amplitude Ordenado com Classes Reais e Predições
sorted_indices = np.argsort(data['RMS Amplitude'].values)
data['Predicted Classe'] = label_encoder.inverse_transform(np.argmax(final_model.predict(X), axis=1))

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
plt.title("RMS Amplitude Ordenado com Predições - Rede Neural")
plt.xlabel("Observações")
plt.ylabel("RMS Amplitude")
plt.legend()
cb = plt.colorbar()
cb.set_label('Classes')
cb.set_ticks(range(len(label_encoder.classes_)))
cb.set_ticklabels(label_encoder.classes_)
plt.grid(True)
plt.savefig('scripts/images/rn_classification_results.png')
plt.show()

# Salvar o modelo
final_model.save("scripts/models/RN/best_nn_model.keras")
print("Modelo salvo como 'scripts/models/RN/best_nn_model.keras'")
