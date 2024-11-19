import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import tensorflow as tf  # Importar TensorFlow para a nova função de R2

# Definir a semente aleatória para reprodutibilidade
np.random.seed(42)
random.seed(42)

# Carregar o dataset
file_path = 'datasets/data/processed_teste_train_config_1_database.csv'
data = pd.read_csv(file_path)

# Dividir o conjunto de dados em treino, validação e teste
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Função R2 personalizada usando TensorFlow
def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

# Funções de pertinência fuzzy para y com gradiente descendente
def gaussmf(x, c, sigma, epsilon=1e-6):
    sigma = max(sigma, epsilon)
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# Gradiente descendente para ajustar os parâmetros da gaussiana
def gradient_descent_gaussian(y, c, sigma, learning_rate=0.01, epochs=256):
    for epoch in range(epochs):
        grad_c = -((y - c) / (sigma ** 2)) * gaussmf(y, c, sigma)
        grad_sigma = ((y - c) ** 2 / (sigma ** 3)) * gaussmf(y, c, sigma)
        
        c -= learning_rate * grad_c.mean()
        sigma -= learning_rate * grad_sigma.mean()
        
        sigma = max(sigma, 1e-6)
    
    return c, sigma

# Sistema fuzzy: definição de pertinência inicial com categorias para pctid
def fuzzy_system(pctid, y, params):
    if 20 <= pctid <= 40:
        low_pctid, medium_pctid, high_pctid = 1, 0, 0
    elif 40 < pctid <= 60:
        low_pctid, medium_pctid, high_pctid = 0, 1, 0
    elif 60 < pctid <= 100:
        low_pctid, medium_pctid, high_pctid = 0, 0, 1

    normal_y = gaussmf(y, params[0], params[1])
    deviating_y = gaussmf(y, params[2], params[3])
    anomalous_y = gaussmf(y, params[4], params[5])

    low_anomaly = np.fmin(low_pctid, normal_y)
    medium_anomaly = np.fmin(medium_pctid, deviating_y)
    high_anomaly = np.fmin(high_pctid, anomalous_y)

    anomaly_score = (np.sum(low_anomaly * 0.3 + medium_anomaly * 0.6 + high_anomaly * 1.0) /
                     (np.sum(low_anomaly) + np.sum(medium_anomaly) + np.sum(high_anomaly) + 1e-6))
    return anomaly_score

# Algoritmo Genético com monitoramento de métricas e tqdm
def genetic_algorithm(train_data, val_data, pop_size=50, generations=100):
    population = [np.random.uniform(0, 1, 6) for _ in range(pop_size)]
    train_mae_history, val_mae_history = [], []

    def fitness(individual, dataset):
        predictions = [fuzzy_system(row['pctid'], row['y'], individual) for _, row in dataset.iterrows()]
        mae = mean_absolute_error(dataset['y'], predictions)
        return mae

    for generation in tqdm(range(generations), desc="Gerações", leave=True):
        train_scores = [(fitness(ind, train_data), ind) for ind in population]
        val_scores = [(fitness(ind, val_data), ind) for ind in population]
        
        best_train_mae = min(train_scores, key=lambda x: x[0])[0]
        best_val_mae = min(val_scores, key=lambda x: x[0])[0]
        train_mae_history.append(best_train_mae)
        val_mae_history.append(best_val_mae)

        selected = [ind for _, ind in sorted(train_scores, key=lambda x: x[0])[:pop_size // 2]]

        new_population = selected.copy()
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            cross_point = random.randint(1, len(parent1) - 1)
            child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
            if random.random() < 0.1:
                child[random.randint(0, len(child) - 1)] += random.uniform(-0.1, 0.1)
            new_population.append(np.clip(child, 0, 1))
        
        population = new_population

    best_individual = min(population, key=lambda ind: fitness(ind, val_data))
    return best_individual, train_mae_history, val_mae_history

# Treinamento e obtenção do histórico de métricas
best_params, train_mae_history, val_mae_history = genetic_algorithm(train_data, val_data)

# Avaliar o modelo nos dados de teste
predictions = [fuzzy_system(row['pctid'], row['y'], best_params) for _, row in test_data.iterrows()]

# Calcular métricas finais
mae = mean_absolute_error(test_data['y'], predictions)
mse = mean_squared_error(test_data['y'], predictions)
r2 = r2_score(tf.constant(test_data['y'].values, dtype=tf.float32), tf.constant(predictions, dtype=tf.float32)).numpy()
f1 = f1_score(np.round(test_data['y']), np.round(predictions))

binary_predictions = np.where(np.array(predictions) >= 0.5, 1, 0)
binary_real = np.where(test_data['y'].values >= 0.5, 1, 0)
accuracy = accuracy_score(binary_real, binary_predictions)

# Plotando o histórico de MAE durante o treinamento e validação
plt.figure(figsize=(12, 6))
plt.plot(train_mae_history, label='MAE - Treinamento')
plt.plot(val_mae_history, label='MAE - Validação')
plt.xlabel('Geração')
plt.ylabel('MAE')
plt.title('Desempenho do Algoritmo Genético ao Longo das Gerações')
plt.legend()
mae_image_path = 'src/GaFuzzy/images/100-50pop.png'
plt.savefig(mae_image_path)

# Plotar as previsões vs valores reais para o conjunto de teste
plt.figure(figsize=(12, 6))
plt.plot(test_data['y'].values, label='Valores Reais')
plt.plot(predictions, label='Previsões do Modelo')
plt.xlabel('Amostras')
plt.ylabel('Anomalia (y)')
plt.title('Comparação entre Valores Reais e Previsões no Conjunto de Teste')
plt.legend()
plt.savefig('src/GaFuzzy/images/100-50pop.png')

# Exibir as métricas finais
print("Desempenho no conjunto de teste:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
print(f"F1 Score: {f1}")
print(f"Acurácia: {accuracy}")




'''''''''''''''''''''''''''
Telegram
'''''''''''''''''''''''''''
# Configurações do Telegram
TELEGRAM_TOKEN = '7934109114:AAEQV9OiDgTJ7tXR7yHlL6GyUpFVqw53ZLo'
CHAT_ID = '1120442358'

# Função para enviar uma mensagem de texto
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': message}
    response = requests.post(url, data=data)
    return response.json()

# Função para enviar uma imagem
def send_telegram_image(image_path):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
    with open(image_path, 'rb') as image_file:
        data = {'chat_id': CHAT_ID}
        files = {'photo': image_file}
        response = requests.post(url, data=data, files=files)
    return response.json()


# Enviar a mensagem com as métricas e a imagem do MAE
metrics_message = (
    "Treinamento concluído com sucesso! 🏆\n"
    f"MAE: {mae}\n"
    f"MSE: {mse}\n"
    f"R2 Score: {r2}\n"
    f"F1 Score: {f1}\n"
    f"Acurácia: {accuracy}"
)
send_telegram_message(metrics_message)
send_telegram_image(mae_image_path)
