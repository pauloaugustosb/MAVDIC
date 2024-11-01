import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset
data = pd.read_csv('datasets/data/main/first_database.csv')

# Remover colunas e linhas completamente vazias
data = data.dropna(axis=0, how='all')
data = data.dropna(axis=1, how='all')

# Remover as colunas 'x' e 'z'
data = data.drop(columns=['x', 'z'], errors='ignore')

# Converter a coluna 'y' para valores numéricos
data['y'] = pd.to_numeric(data['y'], errors='coerce')

# Remover linhas com valores ausentes na coluna 'y'
data = data.dropna(subset=['y'])

# Normalizar a coluna 'y' entre 0 e 1
scaler = MinMaxScaler()
data['y'] = scaler.fit_transform(data[['y']])

# Filtrar para manter apenas as linhas onde 'wconfid' é igual a 1, 2 e 3
filtered_data_config_1 = data[data['wconfid'] == 1]
filtered_data_config_2 = data[data['wconfid'] == 2]
filtered_data_config_3 = data[data['wconfid'] == 3]

# Exportar os datasets filtrados para novos arquivos CSV
filtered_data_config_1.to_csv('datasets/data/processed_teste_train_config_1_database.csv', index=False)
filtered_data_config_2.to_csv('datasets/data/processed_validate_config_2_database.csv', index=False)
filtered_data_config_3.to_csv('datasets/data/processed_validate_config_3_database.csv', index=False)

# Criar uma coluna de timestamps simulados (por exemplo, a cada segundo)
for dataset in [filtered_data_config_1, filtered_data_config_2, filtered_data_config_3]:
    dataset['timestamp'] = pd.date_range(start='2024-01-01', periods=len(dataset), freq='S')

# Criar e salvar gráficos como séries temporais
datasets = {
    'config_1': filtered_data_config_1,
    'config_2': filtered_data_config_2,
    'config_3': filtered_data_config_3
}

for config_name, dataset in datasets.items():
    plt.figure(figsize=(10, 6))
    plt.plot(dataset['timestamp'], dataset['y'], marker='o', linestyle='-', label='y Normalizado')
    plt.title(f'Série Temporal - Configuração {config_name}')
    plt.xlabel('Timestamp')
    plt.ylabel('Valor Normalizado de y')
    plt.grid()
    plt.legend()
    plt.savefig(f'datasets/images/plot_{config_name}.png')  # Salvar o gráfico
    plt.close()  # Fechar a figura para liberar memória

print("Pré-processamento completo! Gráficos salvos como séries temporais com sucesso.")
