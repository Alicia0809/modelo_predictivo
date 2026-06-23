# Utilizar una imagen oficial ligera de Python basada en Linux
FROM python:3.9-slim

# Establecer la carpeta interna de trabajo del servidor
WORKDIR /code

# Copiar e instalar las dependencias de Python
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar el código fuente y el cerebro .pkl entrenado en Colab
COPY . .

# Habilitar los permisos de ejecución del puerto interno seguro exigido por Hugging Face
RUN chmod -R 777 /code
EXPOSE 7860

# Comando ejecutable definitivo para arrancar el servidor web de Panel HoloViz
CMD ["panel", "serve", "app.py", "--address", "0.0.0.0", "--port", "7860", "--allow-websocket-origin=*"]
