# Resumen del proceso

1. Se importan librerías correspondientes y se cargan los datos. El dataset contiene 2999 url y la etiqueta spam o no spam. El objetivo es crear un modelo usando NLP y SVM para detectar si una url es o no spam.
2. Se realiza el preprocesamiento de la variable predictora (url): se pasa a minúsculas, se eliminan strings que no aportan (como https, com, www, etc.), se eliminan url duplicadas, etc.
3. Se crea un vectorizador para las url y se convierte variable target en numérica para poder estimar el modelo.
4. Se estima modelo inicial, usando SVM. Luego se optimiza mediante grid search, y se encuentra el conjunto de parámetros óptimo (en app.py solamente se estima el modelo con parámetros optimizados).
5. Se guardan los datos procesados y el modelo final.