import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# Mapeamento de classes para números
class_mapping = {
    "Normal": 0,
    "Atenção": 1,
    "Anômalo": 2
}

# Função para carregar e mapear os dados
def load_and_map_data(file_path):
    data = pd.read_csv(file_path)
    for column in ['SVM Classe', 'RN Classe', 'Consensus Classe']:
        if column in data.columns:
            data[column] = data[column].map(class_mapping)
    return data

# Função para salvar relatório de classificação
def save_classification_report(y_true, y_pred, model_name, base_name, save_path="scripts/reports"):
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Atenção', 'Anômalo'])
    save_file = f"{save_path}/{base_name}_{model_name}_classification_report.txt"
    with open(save_file, "w") as f:
        f.write(report)
    print(f"Relatório de classificação salvo: {save_file}")

# Função para salvar F1-score por base
def save_f1_scores(y_true, predictions, base_name, save_path="scripts/reports"):
    f1_scores = {}
    for model_name, y_pred in predictions.items():
        f1 = f1_score(y_true, y_pred, average='weighted')
        f1_scores[model_name] = f1
    
    # Salvar F1-scores em um arquivo
    save_file = f"{save_path}/{base_name}_f1_scores.txt"
    with open(save_file, "w") as f:
        for model_name, f1 in f1_scores.items():
            f.write(f"{model_name}: {f1:.4f}\n")
    print(f"F1-scores salvos: {save_file}")

# Função para plotar gráficos
def plot_predictions(data, base_name, save_path="scripts/images"):
    sorted_indices = np.argsort(data['RMS Amplitude'].values)

    # Plotar predições separadas
    for model_name, column, color in zip(
        ['SVM', 'RN', 'Consensus'],
        ['SVM Classe', 'RN Classe', 'Consensus Classe'],
        ['blue', 'orange', 'red']
    ):
        if column in data.columns:
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
                c=data[column].values[sorted_indices],
                cmap='viridis',
                label=f'Predições ({model_name})',
                alpha=0.6
            )
            plt.title(f"RMS Amplitude Ordenado com Predições - {model_name} ({base_name})")
            plt.xlabel("Observações")
            plt.ylabel("RMS Amplitude")
            plt.legend()
            cb = plt.colorbar()
            cb.set_label('Classes')
            cb.set_ticks([0, 1, 2])
            cb.set_ticklabels(['Normal', 'Atenção', 'Anômalo'])
            plt.grid(True)
            save_file = f"{save_path}/{base_name}_{model_name}_classification_results.png"
            plt.savefig(save_file)
            print(f"Gráfico salvo: {save_file}")
            plt.show()

    # Plotar predições combinadas
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.arange(len(data)),
        data['RMS Amplitude'].values[sorted_indices],
        label='RMS Amplitude Ordenado',
        color='green'
    )
    for model_name, column, color in zip(
        ['SVM', 'RN', 'Consensus'],
        ['SVM Classe', 'RN Classe', 'Consensus Classe'],
        ['blue', 'orange', 'red']
    ):
        if column in data.columns:
            plt.scatter(
                np.arange(len(data)),
                data['RMS Amplitude'].values[sorted_indices],
                c=data[column].values[sorted_indices],
                cmap='viridis',
                label=f'Predições ({model_name})',
                alpha=0.6
            )
    plt.title(f"RMS Amplitude Ordenado com Predições dos Modelos ({base_name})")
    plt.xlabel("Observações")
    plt.ylabel("RMS Amplitude")
    plt.legend()
    cb = plt.colorbar()
    cb.set_label('Classes')
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(['Normal', 'Atenção', 'Anômalo'])
    plt.grid(True)
    save_file = f"{save_path}/{base_name}_combined_classification_results.png"
    plt.savefig(save_file)
    print(f"Gráfico salvo: {save_file}")
    plt.show()

# Caminhos para os arquivos de predições
prediction_files = [
    'scripts/predictions/rms_results_training_predictions.csv',
    'scripts/predictions/rms_results_test_2_predictions.csv',
    'scripts/predictions/rms_results_test_3_predictions.csv'
]

# Plotar, salvar relatórios e calcular F1-scores para cada arquivo
for file_path in prediction_files:
    base_name = file_path.split('/')[-1].replace('_predictions.csv', '')
    data = load_and_map_data(file_path)
    plot_predictions(data, base_name, save_path="scripts/images")

    # Calcular e salvar relatórios e F1-scores
    if 'True Classe' in data.columns:
        true_classes = data['True Classe'].values
        predictions = {
            'SVM': data['SVM Classe'].values,
            'RN': data['RN Classe'].values,
            'Consensus': data['Consensus Classe'].values
        }
        for model_name, pred_classes in predictions.items():
            save_classification_report(true_classes, pred_classes, model_name, base_name, save_path="scripts/reports")
        save_f1_scores(true_classes, predictions, base_name, save_path="scripts/reports")
