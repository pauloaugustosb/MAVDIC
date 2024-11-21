# Use uma imagem base com Python e TensorFlow
FROM tensorflow/tensorflow:latest

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean

# Defina o diretório de trabalho
WORKDIR /app

# Instale as dependências
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão
CMD ["python3", "src/GaFuzzy/GaFuzzy.py"]
