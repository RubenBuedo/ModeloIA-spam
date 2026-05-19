# Usar una imagen oficial de Python ligera
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias (usamos el nombre exacto que le diste)
COPY requeriments.txt .

# Instalar las dependencias de Python
# Añadimos --no-cache-dir para mantener la imagen ligera
RUN pip install --no-cache-dir -r requeriments.txt

# Copiar los archivos necesarios para la API
COPY api.py .
COPY modelo_spam.pkl .

# Exponer el puerto en el que correrá FastAPI
EXPOSE 8000

# Comando para arrancar la aplicación usando Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
