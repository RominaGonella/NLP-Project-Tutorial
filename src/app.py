## STEP 1 ##

# librerás #

# ejecutar en consola: pip install -r requirements.txt

# a pesar de ejecutar requirements, debo ejecutar esto para que funcione
! pip install pandas
! pip install sklearn

# importo librerias
import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

# datos #

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')

# guardo datos iniciales
df_raw.to_csv('../data/raw/datos_iniciales.csv', index = False)

# hago copia para limpiar
df = df_raw.copy()

# elimino duplicados y reseteo índice (nombre de las filas)
df = df.drop_duplicates().reset_index(drop = True)

# funciones usadas #

def comas(text):
    """
    Elimina comas del texto
    """
    return re.sub(',', ' ', text)

def espacios(text):
    """
    Elimina enters dobles por un solo enter
    """
    return re.sub(r'(\n{2,})','\n', text)

def minuscula(text):
    """
    Cambia mayusculas a minusculas
    """
    return text.lower()
def comillas(text):
    """
    Sustituye comillas por un espacio
    Ej. hola 'pepito' como le va? -> hola pepito como le va?
    """
    return re.sub("'"," ", text)
def esp_multiple(text):
    """
    Sustituye los espacios dobles entre palabras
    """
    return re.sub(' +', ' ',text)
def url(text):
    """
    Elimina los https
    """
    return re.sub(r'(https://www|https://)', '', text)
def caracteres_no_alfanumericos(text):
    """
    Sustituye caracteres raros, no digitos y letras
    Ej. hola 'pepito' como le va? -> hola pepito como le va
    """
    return re.sub("(\\W)+"," ",text)
def elimino_com_org(text):
    """
    Elimina los com y org (extensiones de url)
    """
    return re.sub(r'( com | org )', ' ', text)

# se aplica la limpieza de texto
df['url_clean'] = df['url'].apply(espacios).apply(comas).apply(minuscula).apply(esp_multiple).apply(comillas)
df['url_clean'].values[:]

# se limpia url
df['url_clean'] = df['url_clean'].apply(url).apply(caracteres_no_alfanumericos)

# guardo en csv para ver si encuentro más patrones a depurar
df.to_csv('../data/interim/datos_limpieza_parcial.csv', index = False)

# elimino 'com' y 'org'
df['url_clean'] = df['url_clean'].apply(elimino_com_org)

# se convierte taret a binario
df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0)

## STEP 2 ##

# creo vectorizador y ajusto sobre variable predictora
vec = CountVectorizer().fit_transform(df['url_clean'])

# creo muestras de entrenamiento y control (esta vez se hace directo sin crear primero X e y)
X_train, X_test, y_train, y_test = train_test_split(vec, df['is_spam'], stratify = df['is_spam'], random_state = 2207)

## STEP 3 ##

# se construye clasificador utilizando SVM, con los parámetros optimizados por grid search en explore.ipynb
best_model = SVC(C = 10, kernel = 'rbf', gamma = 0.1)

# se ajusta el clasificador con los datos de entrenamiento
best_model.fit(X_train, y_train)

# guardo datos finales
df.to_csv('../data/processed/datos_finales.csv', index = False)

# se guarda el modelo
filename = '../models/nlp_model.sav'
pickle.dump(best_model, open(filename,'wb'))
