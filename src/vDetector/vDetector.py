import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Função para análise e resumo da base de dados
def analisar_base_de_dados(data):
    print("Resumo Estatístico da Base de Dados:")
    print(data.describe())

    plt.figure(figsize=(10, 6))
    plt.hist(data['y'], bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribuição dos Valores de y')
    plt.xlabel('y')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()

# Carregar a base de dados
file_path = 'datasets\data\processed_teste_train_config_1_database.csv'
data = pd.read_csv(file_path)
columns_of_interest = ['y']
feature_space = data[columns_of_interest].values

# Analisar a base de dados
analisar_base_de_dados(data)

# Classe V-Detector para detecção de anomalias com ajustes
class VDetector:
    def __init__(self, P, alpha, n, feature_space):
        self.P = P
        self.alpha = alpha
        self.n = n
        self.feature_space = feature_space
        self.detectores = []

    def verificar_cobertura(self, ponto):
        for detector in self.detectores:
            if np.linalg.norm(detector - ponto) < self.P:
                return True
        return False

    def rodar_algoritmo(self):
        N = 0
        while self.n > max(5 / self.P, 5 * (1 - self.P)):
            ponto = random.choice(self.feature_space)

            if not self.verificar_cobertura(ponto):
                self.detectores.append(ponto)
                N += 1

            z = np.random.normal(0, 1)
            if z > self.alpha:
                print("Interrompendo o algoritmo por condição de parada.")
                break

        print(f"Número total de detectores criados: {len(self.detectores)}")

    def avaliar(self, test_data, true_labels):
        predictions = [1 if not self.verificar_cobertura(sample) else 0 for sample in test_data]

        true_positives = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and true_labels[i] == 1)
        false_positives = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and true_labels[i] == 0)
        true_negatives = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and true_labels[i] == 0)
        false_negatives = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and true_labels[i] == 1)

        detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        false_alarm_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

        # Cálculo da acurácia
        accuracy = (true_positives + true_negatives) / len(true_labels)

        print(f"Acurácia: {accuracy:.2f}")
        print(f"Taxa de Detecção (True Positive Rate): {detection_rate:.2f}")
        print(f"Taxa de Falsos Alarmes (False Positive Rate): {false_alarm_rate:.2f}")
        print(f"Verdadeiros Positivos: {true_positives}")
        print(f"Falsos Positivos: {false_positives}")
        print(f"Verdadeiros Negativos: {true_negatives}")
        print(f"Falsos Negativos: {false_negatives}")

        # Gráfico de erro
        erros = [1 if pred != label else 0 for pred, label in zip(predictions, true_labels)]
        plt.figure(figsize=(10, 6))
        plt.plot(erros, label='Erro de Previsão', marker='o')
        plt.title('Gráfico de Erro nas Previsões')
        plt.xlabel('Índice da Amostra')
        plt.ylabel('Erro (1 = Erro, 0 = Correto)')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Gráfico scatter com detectores e dados
        plt.figure(figsize=(10, 6))
        plt.scatter([p[0] for p in self.feature_space], np.zeros_like(self.feature_space), label='Pontos de Dados', color='blue')
        plt.scatter([d[0] for d in self.detectores], np.zeros_like(self.detectores), label='Detectores', color='red', marker='x')
        plt.title('Distribuição de Detectores e Pontos de Dados')
        plt.xlabel('Espaço de Características (y)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Parâmetros ajustados do V-Detector
P = 0.15  # Ajustado para melhorar a sensibilidade
alpha = 0.05
n = 100

# Instanciar e executar o V-Detector melhorado
v_detector = VDetector(P, alpha, n, feature_space)
v_detector.rodar_algoritmo()

# Avaliar o algoritmo com um conjunto de teste
test_data = feature_space[-500:]  # Usando os últimos 500 dados como teste
true_labels = [1 if i >= len(feature_space) - 5 else 0 for i in range(len(test_data))]  # Assumindo que as últimas 5 são anomalias
v_detector.avaliar(test_data, true_labels)
