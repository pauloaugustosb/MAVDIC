import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Função para calcular o RMS (Root Mean Square)
def calculate_rms(data):
    """
    Calcula o valor RMS para os dados fornecidos.
    
    :param data: Array ou lista de valores numéricos.
    :return: Valor RMS calculado.
    """
    return np.sqrt(np.mean(data**2))

# Função para processar os dados de vibração com classificação (baseada no gráfico)
def process_vibration_data_with_graph_based_classes(data, axis='y', window_size=50):
    """
    Processa os dados de vibração, ordena as observações por RMS Amplitude e classifica com base nas divisões do gráfico.
    """
    rms_dataset = []
    num_windows = len(data) // window_size  # Número de janelas completas

    for i in range(num_windows):
        # Extrair uma janela de dados
        window_data = data[axis].iloc[i * window_size:(i + 1) * window_size].values

        # Transformada de Fourier
        fft_values = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(len(window_data))

        # Calcular RMS de amplitude e frequência
        rms_amplitude = calculate_rms(np.abs(fft_values))
        rms_frequency = calculate_rms(freqs)

        # Salvar os resultados
        rms_dataset.append({
            "RMS Amplitude": rms_amplitude,
            "RMS Frequency": rms_frequency
        })

    # Criar o DataFrame com os resultados
    rms_df = pd.DataFrame(rms_dataset)

    # Ordenar por RMS Amplitude
    rms_df = rms_df.sort_values(by="RMS Amplitude").reset_index(drop=True)

    # Adicionar a classificação com base nas observações
    num_observations = len(rms_df)
    rms_df['Classe'] = 'Normal'  # Default

    rms_df.loc[int(num_observations * 0.3):int(num_observations * 0.7), 'Classe'] = 'Atenção'
    rms_df.loc[int(num_observations * 0.7):, 'Classe'] = 'Anômalo'

    return rms_df

# Função para processar os dados sem classificação (para wconfid == 2 e wconfid == 3)
def process_vibration_data_without_classes(data, axis='y', window_size=50):
    """
    Processa os dados de vibração, divide em janelas, aplica FFT e calcula RMS.
    """
    rms_dataset = []
    num_windows = len(data) // window_size  # Número de janelas completas

    for i in range(num_windows):
        # Extrair uma janela de dados
        window_data = data[axis].iloc[i * window_size:(i + 1) * window_size].values

        # Transformada de Fourier
        fft_values = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(len(window_data))

        # Calcular RMS de amplitude e frequência
        rms_amplitude = calculate_rms(np.abs(fft_values))
        rms_frequency = calculate_rms(freqs)

        # Salvar os resultados
        rms_dataset.append({
            "RMS Amplitude": rms_amplitude,
            "RMS Frequency": rms_frequency
        })

    # Criar o DataFrame com os resultados ordenados
    rms_df = pd.DataFrame(rms_dataset)
    rms_df = rms_df.sort_values(by="RMS Amplitude").reset_index(drop=True)

    return rms_df

# Função para gerar o gráfico de RMS Amplitude ordenado com ou sem classificação
def plot_rms_amplitude(rms_df, title_suffix, save_path=None, include_classes=False):
    """
    Plota o gráfico de RMS Amplitude ordenado.
    
    :param rms_df: DataFrame com os dados processados.
    :param title_suffix: Sufixo para o título do gráfico.
    :param save_path: Caminho para salvar o gráfico (opcional).
    :param include_classes: Se True, inclui as divisões de classes no gráfico.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rms_df['RMS Amplitude'], label=f"RMS Amplitude Ordenado ({title_suffix})", color="green")

    if include_classes:
        num_observations = len(rms_df)
        plt.axvline(x=int(num_observations * 0.3), color='orange', linestyle='--', label='Limite Normal -> Atenção')
        plt.axvline(x=int(num_observations * 0.7), color='red', linestyle='--', label='Limite Atenção -> Anômalo')

    plt.title(f"Transformação RMS: Amplitude Ordenada ({title_suffix})")
    plt.xlabel("Observações")
    plt.ylabel("RMS Amplitude")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}_rms_amplitude.png")  # Salva o gráfico como imagem
    plt.show()

# Carregar o dataset original
file_path_original = 'data/databaseMain.csv'  # Atualize o caminho conforme necessário
article_dataset = pd.read_csv(file_path_original)

# Normalizar os dados
columns_to_normalize = ['x', 'y', 'z']
scaler = MinMaxScaler()
article_dataset[columns_to_normalize] = scaler.fit_transform(article_dataset[columns_to_normalize])

# Separar as bases por wconfid
dataset_training = article_dataset[article_dataset['wconfid'] == 1]
dataset_test_2 = article_dataset[article_dataset['wconfid'] == 2]
dataset_test_3 = article_dataset[article_dataset['wconfid'] == 3]

# Processar os dados de treinamento com classificação (wconfid == 1)
rms_results_training = process_vibration_data_with_graph_based_classes(dataset_training, axis='y', window_size=50)

# Processar os dados de teste (wconfid == 2 e wconfid == 3)
rms_results_test_2 = process_vibration_data_without_classes(dataset_test_2, axis='y', window_size=50)
rms_results_test_3 = process_vibration_data_without_classes(dataset_test_3, axis='y', window_size=50)

# Gerar gráficos
plot_rms_amplitude(rms_results_training, "Base de Treinamento (wconfid=1)", save_path="scripts/images/rms_training", include_classes=True)
plot_rms_amplitude(rms_results_test_2, "Base de Teste (wconfid=2)", save_path="scripts/images/rms_test_2", include_classes=False)
plot_rms_amplitude(rms_results_test_3, "Base de Teste (wconfid=3)", save_path="scripts/images/rms_test_3", include_classes=False)

# Salvar as bases processadas
rms_results_training.to_csv('data/rms_results_training.csv', index=False)
rms_results_test_2.to_csv('data/rms_results_test_2.csv', index=False)
rms_results_test_3.to_csv('data/rms_results_test_3.csv', index=False)

print("Bases processadas e salvas com sucesso:")
print("1. Base de Treinamento: rms_results_training.csv")
print("2. Base de Teste 2: rms_results_test_2.csv")
print("3. Base de Teste 3: rms_results_test_3.csv")
