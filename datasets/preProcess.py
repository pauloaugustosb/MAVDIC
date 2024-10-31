import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Abrindo o arquivo principal e atribuindo a coluna de tempo para plote da série temporal
dataset = pd.read_csv(r"datasets\first_database.csv")
dataset['time'] = pd.timedelta_range(start='00:00:00', periods=len(dataset), freq='s')

# Não definir 'time' como índice
# dataset.set_index('time', inplace=True) 

dataset.drop(columns=['x', 'z'], inplace=True)   # Excluir dados em X, autor não utiliza essa coluna, gerando dataset de trabalho

# Separando as configurações
dataset_config1 = dataset[dataset['wconfid'] == 1]
dataset_config2 = dataset[dataset['wconfid'] == 2]
dataset_config3 = dataset[dataset['wconfid'] == 3]
# dataset_config12 = dataset[(dataset['wconfid'] == 1) | (dataset['wconfid'] == 2)]
# dataset_config13 = dataset[(dataset['wconfid'] == 1) | (dataset['wconfid'] == 3)]

# Exportando para arquivos CSV
dataset_config1.to_csv('datasets/dataset_config_1.csv', index=False)
dataset_config2.to_csv('datasets/dataset_config_2.csv', index=False)
dataset_config3.to_csv('datasets/dataset_config_3.csv', index=False)
# dataset_config12.to_csv('datasets/dataset_config_12.csv', index=False)
# dataset_config13.to_csv('datasets/dataset_config_13.csv', index=False)

# Plotando os gráficos
plt.figure(1)
plt.subplot(211)
plt.plot(dataset_config1['time'], dataset_config1['y'], label='Vibração y')
plt.legend()
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Configuração normal (1)')
plt.grid()
plt.subplot(212)
plt.plot(dataset_config1['time'], dataset_config1['z'], 'y', label='Vibração z')
plt.legend()
plt.xlabel('Time')
plt.ylabel('z')
plt.grid()
plt.savefig('datasets/config1.png')
plt.show()


plt.figure(2)
plt.subplot(211)
plt.plot(dataset_config2['time'], dataset_config2['y'], label='Vibração y')
plt.legend()
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Configuração vizinha (2)')
plt.grid()
plt.subplot(212)
plt.plot(dataset_config2['time'], dataset_config2['z'], 'y',label='Vibração z')
plt.legend()
plt.xlabel('Time')
plt.ylabel('z')
plt.grid()
plt.savefig('datasets/config2.png')
plt.show()


plt.figure(3)
plt.subplot(211)
plt.plot(dataset_config3['time'], dataset_config3['y'], label='Vibração y')
plt.legend()
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Configuração oposta (3)')
plt.grid()
plt.subplot(212)
plt.plot(dataset_config3['time'], dataset_config3['z'], 'y',label='Vibração z')
plt.legend()
plt.xlabel('Time')
plt.ylabel('z')
plt.grid()
plt.savefig('datasets/config3.png')
plt.show()


# # Plot do gráfico principal
# plt.figure(figsize=(12, 7))
# plt.plot(dataset_config1['y'], linewidth=2.5, color='teal')
# plt.title('Schematic of Vibration Signal Transformation', fontsize=16, fontweight='bold')
# plt.xlabel('Dataset Observations')
# plt.ylabel('Dataset Amplitudes')

# # Inserir subgráfico com sinal coletado
# inset_ax1 = plt.axes([0.2, 0.7, 0.15, 0.15])  # [left, bottom, width, height]
# inset_ax1.plot(dataset_config1['y'][100:200], color='black')
# inset_ax1.set_title('Collected Signal', fontsize=10)
# inset_ax1.set_xticks([])
# inset_ax1.set_yticks([])

# # Inserir subgráfico com sinal RMS
# rms_amplitude = np.sqrt(np.mean(np.square(dataset_config1['y'][100:200])))
# rms_signal = np.ones(100) * rms_amplitude

# inset_ax2 = plt.axes([0.5, 0.4, 0.15, 0.15])  # [left, bottom, width, height]
# inset_ax2.plot(rms_signal, color='teal')
# inset_ax2.set_title('RMS Amplitude', fontsize=10)
# inset_ax2.set_xticks([])
# inset_ax2.set_yticks([])

# # Exibir o gráfico
# plt.show()
