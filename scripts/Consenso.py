import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.models import load_model # type: ignore

# Configuração de semente
SEED = 73
np.random.seed(SEED)

# Caminhos dos modelos
SVM_MODEL_PATH = "scripts/models/SVMCLONALG/svm_clonalg_model.pkl"
RN_MODEL_PATH = "scripts/models/RN/best_nn_model.keras"
LABEL_ENCODER_PATH = "scripts/models/SVMCLONALG/label_encoder.pkl"

# Carregar modelos
with open(SVM_MODEL_PATH, 'rb') as f:
    svm_model = pickle.load(f)

rn_model = load_model(RN_MODEL_PATH)

# Carregar label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Bases de teste
test_files = [
    'data/rms_results_training.csv',
    'data/rms_results_test_2_classified.csv',
    'data/rms_results_test_3_classified.csv'
]

# Pesos para a média ponderada (ajuste conforme necessário)
alpha = 0.5  # Peso do SVM
beta = 0.5   # Peso da Rede Neural (α + β = 1)

# Função para calcular consenso com média ponderada
def calculate_consensus(svm_probs, rn_probs, alpha=0.5, beta=0.5):
    consensus_probs = alpha * svm_probs + beta * rn_probs
    return np.argmax(consensus_probs, axis=1)  # Pega a classe com maior probabilidade

# Função para gerar resultados
def generate_results(base_path, base_name):
    # Carregar os dados
    data = pd.read_csv(base_path)
    X = data[['RMS Amplitude', 'RMS Frequency']].values

    # Garantir que o SVM retorna probabilidades
    if hasattr(svm_model, "predict_proba"):
        svm_probs = svm_model.predict_proba(X)
    else:
        raise ValueError("O modelo SVM não suporta predict_proba. Certifique-se de que foi treinado com probability=True")

    # Obter as probabilidades da RN
    rn_probs = rn_model.predict(X)

    # Modelo de consenso (média ponderada das probabilidades)
    consensus_classes = calculate_consensus(svm_probs, rn_probs, alpha, beta)

    # Relatórios
    if 'Classe' in data.columns:
        real_classes = label_encoder.transform(data['Classe'])

        # F1-Scores
        svm_f1 = f1_score(real_classes, np.argmax(svm_probs, axis=1), average='weighted')
        rn_f1 = f1_score(real_classes, np.argmax(rn_probs, axis=1), average='weighted')
        consensus_f1 = f1_score(real_classes, consensus_classes, average='weighted')

        print(f"F1-Scores para a base '{base_name}':")
        print(f"SVM: {svm_f1:.4f}")
        print(f"RN: {rn_f1:.4f}")
        print(f"Consensus: {consensus_f1:.4f}")

        # Relatórios de classificação
        unique_classes = np.unique(real_classes)  # Apenas as classes presentes nos dados

        svm_report = classification_report(real_classes, np.argmax(svm_probs, axis=1),
                                        target_names=label_encoder.classes_, labels=unique_classes, zero_division=0)

        rn_report = classification_report(real_classes, np.argmax(rn_probs, axis=1),
                                        target_names=label_encoder.classes_, labels=unique_classes, zero_division=0)

        consensus_report = classification_report(real_classes, consensus_classes,
                                                target_names=label_encoder.classes_, labels=unique_classes, zero_division=0)

        # Salvar relatórios
        with open(f'scripts/reports/{base_name}_svm_report.txt', 'w') as f:
            f.write(svm_report)

        with open(f'scripts/reports/{base_name}_rn_report.txt', 'w') as f:
            f.write(rn_report)

        with open(f'scripts/reports/{base_name}_consensus_report.txt', 'w') as f:
            f.write(consensus_report)


# Executar para todas as bases
def main():
    for file_path in test_files:
        base_name = file_path.split('/')[-1].replace('.csv', '')
        try:
            generate_results(file_path, base_name)
        except Exception as e:
            print(f"Erro ao processar a base '{base_name}': {e}")

if __name__ == "__main__":
    main()
