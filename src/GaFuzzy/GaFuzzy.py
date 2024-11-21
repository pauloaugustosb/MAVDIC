import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import os
import requests

# Configuração do Telegram
TELEGRAM_TOKEN = '7934109114:AAEQV9OiDgTJ7tXR7yHlL6GyUpFVqw53ZLo'
CHAT_ID = '1120442358'

# Definir a semente aleatória para reprodutibilidade
np.random.seed(42)
random.seed(42)

# Carregar o dataset
file_path = 'datasets/data/processed_teste_train_config_1_database.csv'
data = pd.read_csv(file_path)

# Dividir o conjunto de dados em treino, validação e teste
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Função R2 personalizada usando TensorFlow
def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

# Funções de pertinência fuzzy
def gaussmf(x, c, sigma, epsilon=1e-6):
    sigma = max(sigma, epsilon)
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# Sistema fuzzy
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

# Função para salvar checkpoints
def save_checkpoint(generation, population, train_mae_history, val_mae_history, file_path='src/GaFuzzy/checkpoint.pkl'):
    checkpoint = {
        'generation': generation,
        'population': population,
        'train_mae_history': train_mae_history,
        'val_mae_history': val_mae_history
    }
    with open(file_path, 'wb') as f:
        pickle.dump(checkpoint, f)

# Função para carregar checkpoints
def load_checkpoint(file_path='src/GaFuzzy/checkpoint.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# Função para enviar mensagem no Telegram
def send_telegram_message(message):
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        data = {'chat_id': CHAT_ID, 'text': message}
        response = requests.post(url, data=data)
        response.raise_for_status()
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

# Função para enviar imagem para o Telegram
def send_telegram_image(image_path):
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
        with open(image_path, 'rb') as image_file:
            data = {'chat_id': CHAT_ID}
            files = {'photo': image_file}
            response = requests.post(url, data=data, files=files)
            response.raise_for_status()
    except Exception as e:
        print(f"Erro ao enviar imagem: {e}")

# Algoritmo Genético com notificacoes de checkpoints e métricas finais
def genetic_algorithm(train_data, val_data, pop_size=50, generations=201, checkpoint_path='src/GaFuzzy/checkpoint.pkl'):
    checkpoint = load_checkpoint(checkpoint_path)
    
    if checkpoint:
        print("Carregando checkpoint...")
        start_generation = checkpoint['generation']
        population = checkpoint['population']
        train_mae_history = checkpoint['train_mae_history']
        val_mae_history = checkpoint['val_mae_history']
    else:
        print("Iniciando novo treinamento...")
        start_generation = 0
        population = [np.random.uniform(0, 1, 6) for _ in range(pop_size)]
        train_mae_history, val_mae_history = [], []

    def fitness(individual, dataset):
        predictions = [fuzzy_system(row['pctid'], row['y'], individual) for _, row in dataset.iterrows()]
        mae = mean_absolute_error(dataset['y'], predictions)
        return mae

    for generation in tqdm(range(start_generation, generations), desc="Gerações", leave=True):
        train_scores = [(fitness(ind, train_data), ind) for ind in population]
        val_scores = [(fitness(ind, val_data), ind) for ind in population]
        
        best_train_mae = min(train_scores, key=lambda x: x[0])[0]
        best_val_mae = min(val_scores, key=lambda x: x[0])[0]
        train_mae_history.append(best_train_mae)
        val_mae_history.append(best_val_mae)

        # Notificar checkpoints
        if generation > 0 and generation % 25 == 0:
            save_checkpoint(generation, population, train_mae_history, val_mae_history, checkpoint_path)
            send_telegram_message(f"📍 Checkpoint salvo na geração {generation}.")

        # Evolução genética
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

    # Calcular métricas finais
    best_params = min(population, key=lambda ind: fitness(ind, val_data))
    predictions = [fuzzy_system(row['pctid'], row['y'], best_params) for _, row in val_data.iterrows()]
    mse = mean_squared_error(val_data['y'], predictions)
    r2 = r2_score(tf.constant(val_data['y'].values, dtype=tf.float32), tf.constant(predictions, dtype=tf.float32)).numpy()
    f1 = f1_score(np.round(val_data['y']), np.round(predictions))
    accuracy = accuracy_score(np.round(val_data['y']), np.round(predictions))

    # Mensagem e gráfico final
    final_message = (
        f"✅ Treinamento Concluído!\n\n"
        f"📊 Resultados Finais:\n"
        f"🔹 MAE Treinamento: {best_train_mae:.4f}\n"
        f"🔹 MAE Validação: {best_val_mae:.4f}\n"
        f"🔹 MSE: {mse:.4f}\n"
        f"🔹 R²: {r2:.4f}\n"
        f"🔹 F1 Score: {f1:.4f}\n"
        f"🔹 Acurácia: {accuracy:.4f}"
    )
    send_telegram_message(final_message)

    final_plot_path = 'src/GaFuzzy/images/final_training.png'
    os.makedirs(os.path.dirname(final_plot_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(train_mae_history, label='MAE - Treinamento', marker='o')
    plt.plot(val_mae_history, label='MAE - Validação', marker='x')
    plt.xlabel('Geração')
    plt.ylabel('MAE')
    plt.title('Desempenho do Algoritmo Genético ao Longo das Gerações')
    plt.legend()
    plt.grid(True)
    plt.savefig(final_plot_path)
    plt.close()
    send_telegram_image(final_plot_path)

    # Gráfico comparando valores reais e previstos
    comparison_plot_path = 'src/GaFuzzy/images/comparison_real_vs_pred.png'
    plt.figure(figsize=(12, 6))
    plt.plot(val_data['y'].values, label='Valores Reais', linestyle='-', marker='')
    plt.plot(predictions, label='Valores Previstos', linestyle='--', marker='')
    plt.xlabel('Amostras')
    plt.ylabel('Valores')
    plt.title('Comparação entre Valores Reais e Previstos no Conjunto de Validação')
    plt.legend()
    plt.grid(True)
    plt.savefig(comparison_plot_path)
    plt.close()

    return best_params, train_mae_history, val_mae_history

# Executar o treinamento
best_params, train_mae_history, val_mae_history = genetic_algorithm(train_data, val_data)
