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

# Copie os arquivos do projeto para o container
COPY . /app

# Instale as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Defina o comando padrão para executar o script principal
CMD ["python3", "src/GaFuzzy/GaFuzzy.py"]
