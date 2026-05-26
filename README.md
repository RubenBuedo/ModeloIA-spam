No hemos podido subirlo a streamlit por errores de versiones, por lo que hemos utilizado FastAPI
1. Desde la terminal en la carpeta ProyectoIA creas un entorno: python3 -m venv mi_entorno
2. Instalas las bibliotecas necesarias: pip install -r requeriments.txt
3. Activas el entorno: source mi_entorno/bin/activate
4. Activas api.py: uvicorn api:app --host 0.0.0.0 --port 8000
5. Sin cerrar la terminal, abres una pagina web y arrastras el archivo "Index.html" a la url
