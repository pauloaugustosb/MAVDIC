import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
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
SEED = 73
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Carregar o dataset
file_path = 'datasets/data/processed_teste_train_config_1_database.csv'
data = pd.read_csv(file_path)

# Dividir o conjunto de dados em treino, validação e teste
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=SEED)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

# Função de pertinência fuzzy
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

# Funções para salvar e carregar checkpoints
def save_checkpoint(generation_index, train_history, val_history, path='src/GaFuzzy/multiple_checkpoint.pkl'):
    checkpoint = {
        'generation_index': generation_index,
        'train_history': train_history,
        'val_history': val_history
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(path='src/GaFuzzy/multiple_checkpoint.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# Função para enviar mensagem no Telegram
def send_telegram_message(message):
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        data = {'chat_id': CHAT_ID, 'text': message}
        requests.post(url, data=data)
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

# Função para enviar imagem para o Telegram
def send_telegram_image(image_path):
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
        with open(image_path, 'rb') as image_file:
            files = {'photo': image_file}
            data = {'chat_id': CHAT_ID}
            requests.post(url, data=data, files=files)
    except Exception as e:
        print(f"Erro ao enviar imagem: {e}")

# Algoritmo genético
def genetic_algorithm(train_data, val_data, pop_size=50, generations=11):
    population = [np.random.uniform(0, 1, 6) for _ in range(pop_size)]
    train_history, val_history = [], []

    def fitness(individual, dataset):
        predictions = [fuzzy_system(row['pctid'], row['y'], individual) for _, row in dataset.iterrows()]
        return mean_squared_error(dataset['y'], predictions)

    for generation in tqdm(range(generations), desc="Gerações"):
        train_scores = [(fitness(ind, train_data), ind) for ind in population]
        val_scores = [(fitness(ind, val_data), ind) for ind in population]

        best_train = min(train_scores, key=lambda x: x[0])[0]
        best_val = min(val_scores, key=lambda x: x[0])[0]
        train_history.append(best_train)
        val_history.append(best_val)

        # Seleção dos melhores
        selected = [ind for _, ind in sorted(train_scores, key=lambda x: x[0])[:pop_size // 2]]
        new_population = selected.copy()
        while len(new_population) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = np.clip(np.mean([p1, p2], axis=0), 0, 1)
            new_population.append(child)
        population = new_population

    # Melhor indivíduo
    best_params = min(population, key=lambda ind: fitness(ind, val_data))
    return best_params, train_history, val_history

# Múltiplas execuções
def run_multiple_generations(
    train_data, 
    val_data, 
    pop_size=50, 
    generation_list=[11, 51, 101, 151, 201, 251, 301, 351, 401],
    checkpoint_path='src/GaFuzzy/multiple_checkpoint.pkl'
):
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        print("🔄 Checkpoint detectado. Retomando progresso...")
        start_index = checkpoint['generation_index']
    else:
        print("🚀 Iniciando novo treinamento...")
        start_index = 0

    all_results = []

    for index in range(start_index, len(generation_list)):
        gen_count = generation_list[index]
        print(f"\n🔄 Executando {gen_count} gerações...\n")
        
        best_params, train_history, val_history = genetic_algorithm(
            train_data=train_data,
            val_data=val_data,
            pop_size=pop_size,
            generations=gen_count
        )
        
        # Calcular métricas
        predictions = [fuzzy_system(row['pctid'], row['y'], best_params) for _, row in val_data.iterrows()]
        mse = mean_squared_error(val_data['y'], predictions)
        f1 = f1_score(np.round(val_data['y']), np.round(predictions))
        accuracy = accuracy_score(np.round(val_data['y']), np.round(predictions))

        all_results.append({
            'generations': gen_count,
            'mse': mse,
            'f1_score': f1,
            'accuracy': accuracy
        })

        # Gráficos
        plot_path = f'src/GaFuzzy/images/results_{gen_count}_generations.png'
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(train_history, label='MSE - Treinamento', marker='o')
        plt.plot(val_history, label='MSE - Validação', marker='x')
        plt.xlabel('Geração')
        plt.ylabel('MSE')
        plt.legend()
        plt.title(f'MSE ao Longo de {gen_count} Gerações')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot([accuracy] * len(train_history), label='Accuracy - Validação', linestyle='--', marker='o')
        plt.xlabel('Geração')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Accuracy ao Longo de {gen_count} Gerações')
        plt.grid()

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        send_telegram_image(plot_path)

        # Checkpoint
        save_checkpoint(index, train_history, val_history, checkpoint_path)

    return all_results


# Executar múltiplas gerações
generation_list = [11, 51, 101, 151, 201, 251, 301, 351, 401]
results = run_multiple_generations(train_data, val_data, generation_list=generation_list)
