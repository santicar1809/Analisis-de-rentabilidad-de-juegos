# %% [markdown]
#  # Análisis exploratorio de datos:

# %% [markdown]
# ## 1. Importar librerías

# %%
import pandas as pd
import numpy as np
from scipy import stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

# %% [markdown]
# ## 2. Importar archivos

# %%
cabs=pd.read_csv('datasets/moved_project_sql_result_01.csv')
trips=pd.read_csv('datasets/moved_project_sql_result_04.csv')
loop_trips=pd.read_csv('datasets/moved_project_sql_result_07.csv')

# %% [markdown]
# ## 3. Estudio de los datos de los ficheros

# %% [markdown]
# ### 3.1 Tipos de datos

# %%
cabs.head(5)


# %%
trips.head(5)


# %%
loop_trips.head(5)

# %%
# Revisamos los tipos de datos de los dataframes
cabs.info()
trips.info()
loop_trips.info()

# %% [markdown]
# Los datos de fecha están como strings, asi que lo pasaremos a tipo datetime.

# %%
#Cambiamos el tipo de dato a datetime
loop_trips['start_ts']=pd.to_datetime(loop_trips['start_ts'])
loop_trips.info()

# %% [markdown]
# ### 3.2 Tratamiento de ausentes y duplicados

# %%
#Calculamos los ausentes
print('Ausentes cabs:\n',cabs.isna().sum())
print('Ausentes trips:\n',trips.isna().sum())
print('Ausentes loop_trips:\n',loop_trips.isna().sum())      

# %%
#Calculamos los duplicados
print('Duplicados cabs:\n',cabs.duplicated().sum())
print('Duplicados trips:\n',trips.duplicated().sum())
print('Duplicados loop_trips:\n',loop_trips.duplicated().sum()) 

# %%
#Revisamos los duplicados
print(loop_trips[loop_trips.duplicated()].head(10))
print()
#Calculamos el porcentaje de duplicados
print('Porcentaje de duplicados: ',100*loop_trips.duplicated().sum()/loop_trips.shape[0])

# %%
#Al tener un porcentaje de duplicados tan grande, los eliminamos.
loop_trips.drop_duplicates(inplace=True)
print('Duplicados loop_trips:\n',loop_trips.duplicated().sum()) 
print()
print(loop_trips.info())

# %% [markdown]
# Respecto a los ausentes, no tenemos problemas, debido a que no existen valores ausentes en ninguno de los dataframes. Por otra parte, tenemos duplicados únicamente en el dataframe loop_trips, con un porcentaje del 18% aproximadamente, por esta razón los eliminamos.

# %% [markdown]
# ## 4. Análisis

# %% [markdown]
# ### 4.1 Barrios destino más populares.

# %%
#Ordenamos los valores de mayor a menor promedio de viajes y redondeamos
hood=trips.sort_values(by='average_trips',ascending=False)
hood_top_10=hood.head(10)
hood_top_10['average_trips']=hood_top_10['average_trips'].round(2)
hood_top_10

# %%
#Creamos un diagrama de barras para visualiazar mejor
fig=px.bar(hood_top_10,x='dropoff_location_name',y='average_trips',title='Top 10 barrios principales', color='dropoff_location_name')
fig.update_traces(showlegend=False)
fig.show()

# %% [markdown]
# Podemos ver en la anterior gráfica los 10 barrios principales, destacando a **Loop, River North y Streeterville** como los 3 más populares con **10727.47, 9523.67 y 6664.67** viajes en promedio respectivamente.

# %% [markdown]
# ### 4.2 Número de viajes por empresas de taxi

# %%
#Ordenamos de mayor a menor las empresas con más viajes
taxi_company=cabs.sort_values(by='trips_amount',ascending=False)
top10_taxi_company=taxi_company.head(10)
top10_taxi_company

# %%
#Creamos diagrama de barras para visualizar mejor
fig1=px.bar(top10_taxi_company,x='company_name',y='trips_amount',title='Top 10 empresas de taxi principales', color='company_name')
fig1.update_traces(showlegend=False)
fig1.show()

# %%
print('Número total de viajes: ',cabs['trips_amount'].sum())

# %% [markdown]
# En la gráfica anterior podemos ver las empresas más popilares por numero de viajes realizados, de los cuales tenemos a la empresa **Flash Cab** como la que tiene más viajes con **19558 viajes**, seguido por la empresa **Taxi Afiliation Services** con **11422 viajes** y la empresa **Medallion Leasing** con **10367 viajes** en total de un total de **64 empresas y 137311 viajes**.

# %% [markdown]
# ## 5.Conclusiones

# %% [markdown]
# 1. En conclusión los 3 barrios más populares en cuanto a finalización de viajes son Loop, River North y Streeterville debido a que su promedio de viajes es más alto, esto se debe a que la mayoría de estos barrios y el resto de barrios de la lista quedan en el Downtown y cerga al lago michigan los cuales son las zonas turisticas más representativas de Chicago. El unico barrio que no queda cerca a la zona es O'Hare, el cual es el barrio donde está el aeropuerto, por esta razón se encuentra en la lista, al Chicago ser una de las ciudades más turisticas de Estados Unidos, recibe más viajes en sus destinos turisticos.
# 
# 2. Las 3 empresas más conocidas de Chicago son Flash Cab, Taxi Afiliation Services y Medallion Leasing, los cuales son buenas opciones para invertir debido a su alta cantidad de viajes, sin embargo, sería importante hacer un análisis más detallado de estas empresas.

# %% [markdown]
# ## 6. Prueba de hipótesis

# %% [markdown]
# ### 6.1 Prueba de Hipotesis entre el barrio Loop y el Aeropuerto Internacional O'Hare 

# %% [markdown]
# "La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare cambia los sábados lluviosos".
# 
# H0: La duración promedio de los viajes entre Loop hasta el Aeropuerto Internacional O'hare los días lluviosos y los días soleados es igual.  
# H1: La duración promedio de los viajes entre Loop hasta el Aeropuerto Internacional O'hare los días lluviosos y los días soleados **NO** es igual.  
# Indice de significancia del 5%

# %% [markdown]
# Debido a que debemos validar si los promedios son iguales o no, dependiendo el clima y que no conocemos las desviaciones estandar de cada grupo, usaremos una prueba T-test de una cola.

# %%
#Revisamos el dataframe de los viajes
loop_trips.sample(10)

# %%
loop_trips['weather_conditions'].value_counts()

# %%
#Separamos los datos de días lluviosos y dias soleados
good_trips=loop_trips[loop_trips['weather_conditions']=='Good']['duration_seconds']
bad_trips=loop_trips[loop_trips['weather_conditions']=='Bad']['duration_seconds']
var_good=np.var(good_trips)
var_bad=np.var(bad_trips)
print(f'Varianza de los días soleados: {var_good}\nVarianza de los días lluviosos: {var_bad}')

# %%
#Hacemos la prueba de hipotesis
alpha=0.05
results=st.ttest_ind(good_trips,bad_trips)
print(f'El promedio de los viajes soleados : {good_trips.mean()}\nEl promedio de los viajes lluviosos es : {bad_trips.mean()}\n\nt-statistic: {results[0]}\np-value: {results[1]}')
if results.pvalue < alpha: # comparar el valor p con el umbral
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula") 


