#%%
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import plotly.express as px


#%%
#Paso 1. Abre el archivo de datos y estudia la información general 
df=pd.read_csv('data_frame/games.csv')
print(df.head(10))
print()
print(df.info(10))

#%% 
#Paso 2. Prepara los datos
#Reemplazamos los nombres de las columnas en minusculas
df.columns=df.columns.str.lower()
'''
Debemos pasar los valores de la columna user_score a tipo float, 
sin embargo, no podemos aún sin tratar los valores 'tbd'.
'''
df['year_of_release']=df['year_of_release'].astype('str')
print(df.info())
'''
Pasamos year_of_release a string, ya que al ser años, no se pueden tratar como un número 
ni hacer operaciones con estos.
'''
#Vamos a tratar los 'tbd' como nan, por lo tanto los pasaremos a ausentes y luego pasaremos los datos a float
df['user_score']=df['user_score'].replace('tbd',np.nan)
print(df['user_score'].unique())
df['user_score']=df['user_score'].astype('float')
df.info()
'''Pasamos todos los tipos de datos incorrectos al correcto, ahora vamos a trabajar con los ausentes'''
print(100*df.isna().sum()/df.shape[0])
df.dropna(axis=0,inplace=True,subset=['name','genre'])
print(100*df.isna().sum()/df.shape[0])
''' 
Borramos los ausentes de la columna name y genre, ya que no son significantes en el dataframe.
'''
#Calculamos el total de ventas de todas las regiones.
df['total_sales']=df['na_sales']+df['eu_sales']+df['jp_sales']
print(df.iloc[:,4:])

# %%
