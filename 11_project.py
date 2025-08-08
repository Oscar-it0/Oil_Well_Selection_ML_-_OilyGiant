#!/usr/bin/env python
# coding: utf-8

# # Hola Oscar! <a class="tocSkip"></a>
# 
# Mi nombre es Oscar Flores y tengo el gusto de revisar tu proyecto. Si tienes algún comentario que quieras agregar en tus respuestas te puedes referir a mi como Oscar, no hay problema que me trates de tú.
# 
# Si veo un error en la primera revisión solamente lo señalaré y dejaré que tú encuentres de qué se trata y cómo arreglarlo. Debo prepararte para que te desempeñes como especialista en Data, en un trabajo real, el responsable a cargo tuyo hará lo mismo. Si aún tienes dificultades para resolver esta tarea, te daré indicaciones más precisas en una siguiente iteración.
# 
# Te dejaré mis comentarios más abajo - **por favor, no los muevas, modifiques o borres**
# 
# Comenzaré mis comentarios con un resumen de los puntos que están bien, aquellos que debes corregir y aquellos que puedes mejorar. Luego deberás revisar todo el notebook para leer mis comentarios, los cuales estarán en rectángulos de color verde, amarillo o rojo como siguen:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
#     
# Muy bien! Toda la respuesta fue lograda satisfactoriamente.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Existen detalles a mejorar. Existen recomendaciones.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario de Reviewer</b> <a class="tocSkip"></a>
# 
# Se necesitan correcciones en el bloque. El trabajo no puede ser aceptado con comentarios en rojo sin solucionar.
# </div>
# 
# Cualquier comentario que quieras agregar entre iteraciones de revisión lo puedes hacer de la siguiente manera:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante.</b> <a class="tocSkip"></a>
# </div>
# 
# Mucho éxito en el proyecto!

# ## Resumen de la revisión 1 <a class="tocSkip"></a>

# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Buen trabajo Oscar! Tu notebook está muy completo y en el orden adecuado. Tan solo falta que corrijas un par de defectos en la parte del bootstrapping, te dejé comentarios detallados al respecto.
#     
# Saludos!    
# 
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante</b> <a class="tocSkip"></a>
#     
# Hola Oscar, gracias por tus cometarios. 
#     
# He seguido tus indicaciones ya que de hecho el asunto de utilizar los valores reales fue una duda que tuve desde la primera entrega, pues anteriormente había hecho pruebas con dichos valores notando que en Región 1 no se formaba la "campana" y también por temas de interpretación ("Utilizando las predicciones que almacenaste en el paso 4.2, emplea la técnica del bootstrapping con 1000 muestras para hallar la distribución de los beneficios"), se incrementó mi duda en cuanto a qué datos utilizar, pues la selección con ambos datos la determiné desde el punto 4.
# 
# El intervalo de confianza y porcentaje de pérdida fueron corregidos según tus instrucciones.
#     
# </div>

# ## Resumen de la revisión 2<a class="tocSkip"></a>

# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor v2</b> <a class="tocSkip"></a>
# 
# Ok Oscar entiendo. Bien hecho al corregir la parte de las métricas de bootstrapping, pero para el cálculo del bootstrapping mismo aún hay detalles por corregir. Te dejé un comentario detallado con las correcciones a realizar.
#     
# Saludos!    
# 
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante v2</b> <a class="tocSkip"></a>
#     
# Hola Oscar, corregido.
#     
# </div>

# ------

# # Descripción del proyecto
# 
# Trabajas en la compañía de extracción de petróleo OilyGiant. Tu tarea es encontrar los mejores lugares donde abrir 200 pozos nuevos de petróleo.
# 
# Para completar esta tarea, tendrás que realizar los siguientes pasos:
# 
# * Leer los archivos con los parámetros recogidos de pozos petrolíferos en la región seleccionada: calidad de crudo y volumen de reservas.
# * Crear un modelo para predecir el volumen de reservas en pozos nuevos.
# * Elegir los pozos petrolíferos que tienen los valores estimados más altos.
# * Elegir la región con el beneficio total más alto para los pozos petrolíferos seleccionados.
# 
# Tienes datos sobre muestras de crudo de tres regiones. Ya se conocen los parámetros de cada pozo petrolero de la región. Crea un modelo que ayude a elegir la región con el mayor margen de beneficio. Analiza los beneficios y riesgos potenciales utilizando la técnica bootstrapping.
# 
# Condiciones:
# 
# * Solo se debe usar la regresión lineal para el entrenamiento del modelo.
# * Al explorar la región, se lleva a cabo un estudio de 500 puntos con la selección de los mejores 200 puntos para el cálculo del beneficio.
# * El presupuesto para el desarrollo de 200 pozos petroleros es de 100 millones de dólares.
# * Un barril de materias primas genera 4.5 USD de ingresos. El ingreso de una unidad de producto es de 4500 dólares (el volumen de reservas está expresado en miles de barriles).
# * Después de la evaluación de riesgo, mantén solo las regiones con riesgo de pérdidas inferior al 2.5%. De las que se ajustan a los criterios, se debe seleccionar la región con el beneficio promedio más alto.
# 
# Los datos son sintéticos: los detalles del contrato y las características del pozo no se publican.
# 
# Descripción de datos:
# 
# Los datos de exploración geológica de las tres regiones se almacenan en archivos:
# 
# * geo_data_0.csv. Descarga el conjunto de datos
# * geo_data_1.csv. Descarga el conjunto de datos
# * geo_data_2.csv. Descarga el conjunto de datos
# * id — identificador único de pozo de petróleo
# * f0, f1, f2 — tres características de los puntos (su significado específico no es importante, pero las características en sí son significativas)
# * product — volumen de reservas en el pozo de petróleo (miles de barriles).

# # Instrucciones del proyecto

# ## Descarga y prepara los datos. Explica el procedimiento.

# In[1]:


# Carga de librerías
import pandas as pd
from sklearn import set_config
set_config(print_changed_only=False)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt


# In[2]:


# Importación de dataframes
geo_data_0 = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/geo_data_0.csv', index_col='id')
geo_data_1 = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/geo_data_1.csv', index_col='id')
geo_data_2 = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/geo_data_2.csv', index_col='id')


# In[3]:


# Revisión de filas duplicadas en churn_df
print("Filas duplicadas en geo_data_0: ",geo_data_0.index.duplicated().sum())
print("Filas duplicadas en geo_data_1: ",geo_data_1.index.duplicated().sum())
print("Filas duplicadas en geo_data_2: ",geo_data_2.index.duplicated().sum())


# In[4]:


# Eliminación de duplicados
geo_data_def_0 = geo_data_0.loc[~geo_data_0.index.duplicated(keep='first')]
geo_data_def_1 = geo_data_1.loc[~geo_data_1.index.duplicated(keep='first')]
geo_data_def_2 = geo_data_2.loc[~geo_data_2.index.duplicated(keep='first')]


# ## Entrena y prueba el modelo para cada región:
# 
# ### Divide los datos en un conjunto de entrenamiento y un conjunto de validación en una proporción de 75:25

# In[5]:


# Función de separación de conjuntos
def ensemble (df,target_col,size):
    # Determinación de características y objetivos
    features = df.drop(target_col, axis=1) # extrae las características
    target = df[target_col] # extrae los objetivos
    # 2.1 Divide los datos en un conjunto de entrenamiento y un conjunto de validación en una proporción de 75:25
    features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=size, random_state=12345) # segmenta el 25% de los datos para hacer el conjunto de validación
    
    return  features, target, features_train, features_valid, target_train, target_valid


# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Bien, correcto el uso de train_test_split
#     
# 
# </div>

# ### Entrena el modelo y haz predicciones para el conjunto de validación.
# 
# ### Guarda las predicciones y las respuestas correctas para el conjunto de validación.

# In[6]:


# Función de separación de entrenamiento y validación del modelo
def lr_model (features_to_train, target_to_train, features_to_valid, target_to_valid):
    # 2.2 Entrena el modelo y haz predicciones para el conjunto de validación.
    model = LinearRegression() # inicializa el constructor de modelos
    model_fited = model.fit(features_to_train, target_to_train) # entrena el modelo en el conjunto de entrenamiento
    predict_valid = model.predict(features_to_valid) # obtén las predicciones del modelo en el conjunto de validación
    predict_valid_mean = pd.Series(target_to_train.mean(), index=target_to_valid.index)
    predict_valid_median = pd.Series(target_to_train.median(), index=target_to_valid.index)
    model_tr_score = model.score(features_to_train, target_to_train)
    model_vl_score = model.score(features_to_valid, target_to_valid)
    
    # 2.3 Guarda las predicciones y las respuestas correctas para el conjunto de validación.
    return model_fited, predict_valid, predict_valid_mean, predict_valid_median, model_tr_score, model_vl_score


# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Muy bien, correctos todos, aunque no nos interesa para esta parte el score (R2) de la regresión lineal
#     
# 
# </div>

# ### Muestra el volumen medio de reservas predicho y RMSE del modelo.

# In[7]:


# Función de determinación de métricas
def lr_metrics (target_v, predict_v, predict_v_m, predict_v_mdn):
    # 2.4 Muestra el volumen medio de reservas predicho y RMSE del modelo.
    me = max_error(target_v, predict_v)
    mae = mean_absolute_error(target_v, predict_v)
    mae_median = mean_absolute_error(target_v, predict_v_mdn)    
    mse = mean_squared_error(target_v, predict_v)
    rmse = mean_squared_error(target_v, predict_v, squared = False) # mse ** .5 calcula la RECM en el conjunto de validación
    mse_mean = mean_squared_error(target_v, predict_v_m)
    rmse_mean = mean_squared_error(target_v, predict_v_m, squared = False) # ** .5  calcula la RECM en el conjunto de validación
    r2 = r2_score(target_v, predict_v)
    d = {'metrica': ["M","MAE", "MAE True values", "Test MAE", "MSE", "RMSE", "RMSE True values", "Test RMSE", "R2"],
         'unidades': [me, mae, mae_median, mae - mae_median, mse, rmse, rmse_mean, rmse - rmse_mean, r2]}
    
    return me, mae, mae_median, mse, rmse, mse_mean, rmse_mean, r2, (pd.DataFrame(data=d))


# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Ok, está bien esta función para las métricas. Te sugiero que al guardarlas no escribas un nombre abreviado como M, pues luego confunde al ver el resultado, no hay problema en escribir en palabras el nombre completo.
#     
# 
# </div>

# In[8]:


# Aplicación de funciones Región 0
features_0, target_0, features_train_0, features_valid_0, target_train_0, target_valid_0 = ensemble(geo_data_def_0, 'product', .25)
model_fited_0, predict_valid_0, predict_valid_mean_0, predict_valid_median_0, model_tr_score_0, model_vl_score_0 = lr_model(features_train_0, target_train_0, features_valid_0, target_valid_0)
me_0, mse_0, rmse_0, mse_mean_0, rmse_mean_0, r2_0, mae_0, mae_median_0, info_metrics_0 = lr_metrics (target_valid_0, predict_valid_0, predict_valid_mean_0, predict_valid_median_0)


# In[9]:


# Aplicación de funciones Región 1
features_1, target_1, features_train_1, features_valid_1, target_train_1, target_valid_1 = ensemble(geo_data_def_1, 'product', .25)
model_fited_1, predict_valid_1, predict_valid_mean_1, predict_valid_median_1, model_tr_score_1, model_vl_score_1 = lr_model(features_train_1, target_train_1, features_valid_1, target_valid_1)
me_1, mse_1, rmse_1, mse_mean_1, rmse_mean_1, r2_1, mae_1, mae_median_1, info_metrics_1 = lr_metrics (target_valid_1, predict_valid_1, predict_valid_mean_1, predict_valid_median_1)


# In[10]:


# Aplicación de funciones Región 2
features_2, target_2, features_train_2, features_valid_2, target_train_2, target_valid_2 = ensemble(geo_data_def_2, 'product', .25)
model_fited_2, predict_valid_2, predict_valid_mean_2, predict_valid_median_2, model_tr_score_2, model_vl_score_2 = lr_model(features_train_2, target_train_2, features_valid_2, target_valid_2)
me_2, mse_2, rmse_2, mse_mean_2, rmse_mean_2, r2_2, mae_2, mae_median_2, info_metrics_2 = lr_metrics (target_valid_2, predict_valid_2, predict_valid_mean_2, predict_valid_median_2)


# ### Analiza los resultados.

# In[11]:


# DF resumen de métricas por región
metrics_0_1 = info_metrics_0.merge(info_metrics_1,on='metrica', how='left')
metrics_by_region = metrics_0_1.merge(info_metrics_2,on='metrica', how='left')
metrics_by_region = metrics_by_region.rename(columns={'unidades_x':'region_0', 'unidades_y':'region_1', 'unidades':'region_2'})
metrics_by_region


# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Buen trabajo, correctos los resultados. Lo solicitado era el RMSE.
# 
# </div>

# #### Comentarios Sección 2
# 
# Se presentan las métricas para las tres regiones.
# 
# Existen similitudes en las magnitudes entre la Reg0 y Reg2 con valores máximos de error entre el valor verdadero y el valor predicho abunda en 120 y 130 us. Siendo Reg1 el de menor magnitud y representa errores más pequeños.
# 
# Lo anterior se refuerza con el Error absoluto promedio que también representa una afectación mínima en Reg1.
# 
# Posteriormente en Reg1 el MSE y RMSE representan un mejor ajuste del modelo aproximándose a 0. Así también el coeficiente de determinación R2 es el que más se aproxima a 1, confirmando buenas métricas para la región.
# 
# Al momento Reg1 se presenta como la mejor región para invertir.

# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Ok, bien, pero ojo que hasta ahora no hay ninguna justificación de negocio que indique que la región 1 es la ideal. No basta con que tenga poco error el modelo para decir que es la mejor para invertir, tiene que haber un balance entre el ingreso generado, el costo y la incertidumbre de los resultados en esa zona.
# 
# </div>

# ## Prepárate para el cálculo de ganancias:
# 
# ### Almacena todos los valores necesarios para los cálculos en variables separadas.

# In[12]:


# Definición de variables generales
tot_investment = 100000000
oil_well_budg = 200
oil_well_cost = tot_investment / oil_well_budg
oil_well_unit_price = 4500
oil_well_prod_min_expect = oil_well_cost / oil_well_unit_price

print("Información presupuestada para la inversión\n",
      "\nImporte Total de la inversión\n", tot_investment,
      "\n" 
      "\nNúmero de pozos presupuestados\n", oil_well_budg,
      "\n"
      "\nCosto presupuestado por pozo\n", tot_investment / oil_well_budg, 
      "\n"
      "\nPrecio de mercado por unidad (1000 barriles)\n", oil_well_unit_price, 
      "\n"
      "\nUnidades producidas por pozo para recuerar la inversión\n", oil_well_cost / oil_well_unit_price)


# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Muy bien, correcto
# 
# </div>

# ### Dada la inversión de 100 millones por 200 pozos petrolíferos, de media un pozo petrolífero debe producir al menos un valor de 500,000 dólares en unidades para evitar pérdidas (esto es equivalente a 111.1 unidades). Compara esta cantidad con la cantidad media de reservas en cada región.

# In[13]:


# Función de comparación de promedio de unidades producidas por región contra el mínimo requerido para recuperar la información
def invest_comp_reg (region):
    oil_well_mean_reg = region['product'].mean()
    comparison = oil_well_mean_reg - oil_well_prod_min_expect
    
    return print("\nPromedio de unidades producidas por pozo en la Region\n", oil_well_mean_reg, 
                 "\nDiferencia respecto a unidades para recuerar la inversión\n", comparison)


# In[14]:


# Aplicación de función de comparación
print("Comparación unidades para recuperar la inversión Región 0, 1 y 2")
print("\nUnidades producidas por pozo para recuerar la inversión\n", oil_well_cost / oil_well_unit_price)
invest_comp_reg (geo_data_0)
invest_comp_reg (geo_data_1)
invest_comp_reg (geo_data_2)


# ### Presenta conclusiones sobre cómo preparar el paso para calcular el beneficio.

# #### Comentarios Sección 3
# 
# Se establecen las variables generales para en análisis.
# 
# En la acción comparativa entre el promedio de unidades producidas por región contra el mínimo requerido para recuperar la inversión, las tres regiones quedan por debajo de los 111 k.
# 
# Al momento no se tiene certeza al elegir alguna región, solamente las consideraciones de los modelos.
# 
# En lo posterior, requerimos hacer los cálculos de ingresos posibles según los valores predichos y los precios de venta.
# 

# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Bien. Con el cálculo anterior pareciera que ninguna región es rentable, pero esto ocurre puesto que estamos evaluando explotar pozos promedio y esa no será la estrategia.
#     
# 
# </div>

# ## Escribe una función para calcular la ganancia de un conjunto de pozos de petróleo seleccionados y modela las predicciones:
# 
# ### Elige los 200 pozos con los valores de predicción más altos de cada una de las 3 regiones (es decir, archivos 'csv').

# In[15]:


# Función para elegir el número deseado de pozos con los valores de predicción más altos de cada región (y sus valores verdaderos)
def select_ensemb (predict_data, features_data, num_select, valid_data):
    select_pred = pd.Series(predict_data, index=features_data.index).sort_values(ascending=False).iloc[0:num_select]
    # DF Selección (valores predichos-verdaderos)
    select_p_v_df = pd.concat([select_pred,valid_data],axis=1, join='inner')
    select_p_v_df.columns = ['predicted', 'true']
    
    return select_p_v_df


# In[16]:


# Aplicación de Función
select_200_p_v_df_0 = select_ensemb (predict_valid_0, features_valid_0, oil_well_budg, target_valid_0)
select_200_p_v_df_1 = select_ensemb (predict_valid_1, features_valid_1, oil_well_budg, target_valid_1)
select_200_p_v_df_2 = select_ensemb (predict_valid_2, features_valid_2, oil_well_budg, target_valid_2)


# In[17]:


# Función para determinar producción e ingreso total posibles, beneficio y margen de beneficio
def produc_benef_bdg (data, oil_well_unitprice, investment):
    df_selected = data.to_frame(name='production')
    df_selected['benefit'] = df_selected['production'] * oil_well_unitprice
    prod_total = df_selected['production'].sum()
    prod_mean = df_selected['production'].mean()
    revenue_total = df_selected['benefit'].sum()
    gross_profit = revenue_total - investment
    gross_margin = gross_profit / investment
    
    
    return df_selected, prod_total, prod_mean, revenue_total, gross_profit, gross_margin


# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Muy bien con ambas funciones. Lo importante de la primera son los valores reales.
#     
# 
# </div>

# In[18]:


# Aplicación de Función a datos de predicción
df_selected_p_0, prod_total_p_0, prod_mean_p_0, revenue_total_p_0, gross_profit_p_0, gross_margin_p_0 = produc_benef_bdg (select_200_p_v_df_0['predicted'], oil_well_unit_price, tot_investment)
df_selected_p_1, prod_total_p_1, prod_mean_p_1, revenue_total_p_1, gross_profit_p_1, gross_margin_p_1 = produc_benef_bdg (select_200_p_v_df_1['predicted'], oil_well_unit_price, tot_investment)
df_selected_p_2, prod_total_p_2, prod_mean_p_2, revenue_total_p_2, gross_profit_p_2, gross_margin_p_2 = produc_benef_bdg (select_200_p_v_df_2['predicted'], oil_well_unit_price, tot_investment)


# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Ok, bien, para esta parte calcularemos los beneficios con los valores de predicción.
# 
# </div>

# ### Resume el volumen objetivo de reservas según dichas predicciones. Almacena las predicciones para los 200 pozos para cada una de las 3 regiones.

# In[19]:


# DF Resumen de predicciones de volúmenes producidos por región
pd.options.display.float_format = '{:.2f}'.format
forecast_prod_p = {'concepto':["Volumen total", "Volumen promedio"], 
                  'Region_0':[prod_total_p_0, prod_mean_p_0],
                  'Region_1':[prod_total_p_1, prod_mean_p_1], 
                  'Region_2':[prod_total_p_2, prod_mean_p_2]}

f_prod_df_p = pd.DataFrame(data=forecast_prod_p)
f_prod_df_p


# ### Calcula la ganancia potencial de los 200 pozos principales por región. Presenta tus conclusiones: propón una región para el desarrollo de pozos petrolíferos y justifica tu elección.

# In[20]:


# DF Resumen de predicciones de ingresos y beneficios por región
forecast_rev_p = {'concepto':["Ingresos potenciales", "Beneficio Bruto", "Margen Bruto"],
                  'Region_0':[revenue_total_p_0, gross_profit_p_0, gross_margin_p_0],
                  'Region_1':[revenue_total_p_1, gross_profit_p_1, gross_margin_p_1],
                  'Region_2':[revenue_total_p_2, gross_profit_p_2, gross_margin_p_2]}

f_rev_df_p = pd.DataFrame(data=forecast_rev_p)
f_rev_df_p


# #### Comentarios Sección 4
# 
# Siguiendo las instrucciones de seleccionar los 200 pozos con mayor nivel predictivo por región, se calcula el ingreso posible por cada pozo, a su vez los totales de producción e ingresos y por último, el margen bruto.
# 
# Respecto a los volúmenes de producción Reg0 y Reg2 se muestran con aproximadamente 30,000 k mientras que Reg1 con 27,000 k.
# 
# Del mismo modo, Reg0 y Reg2 superan los 130 millones de dólares en posibles ventas. Reg1 queda por debajo de 124 millones.
# 
# Conclusiones:
# 
# * Al elegir los mejores 200 pozos, se descarta una buena cantidad de pozos de baja producción (en especial en Reg1) haciendo que la comparativa actual quede mejor balanceada entre las 3 regiones.
# 
# * De cualquier manera y en concordancia con los resultados comparativos de la sección 3, Reg0 y Reg2 se muestran con los mejores resultados posibles.
# 

# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Correcto. En este escenario tenemos el mejor caso, puesto que estamos escogiendo los mejores pozos. Sin embargo, en la realidad, no podemos tener completa certeza de que los pozos escogidos como los mejores mediante el modelo, sean realmente los mejores.
# 
# </div>

# #### Paso extra 4. Aplicación del procedimiento de volúmenes e ingresos a los valores verdaderos de la selección de 200 pozos.
# 
# 

# In[21]:


# Aplicación de Función a datos de verdaderos
df_selected_v_0, prod_total_v_0, prod_mean_v_0, revenue_total_v_0, gross_profit_v_0, gross_margin_v_0 = produc_benef_bdg (select_200_p_v_df_0['true'], oil_well_unit_price, tot_investment)
df_selected_v_1, prod_total_v_1, prod_mean_v_1, revenue_total_v_1, gross_profit_v_1, gross_margin_v_1 = produc_benef_bdg (select_200_p_v_df_1['true'], oil_well_unit_price, tot_investment)
df_selected_v_2, prod_total_v_2, prod_mean_v_2, revenue_total_v_2, gross_profit_v_2, gross_margin_v_2 = produc_benef_bdg (select_200_p_v_df_2['true'], oil_well_unit_price, tot_investment)


# In[22]:


# DF Resumen de volúmenes producidos por región
pd.options.display.float_format = '{:.2f}'.format
forecast_prod_v = {'concepto':["Volumen total", "Volumen promedio"], 
                  'Region_0':[prod_total_v_0, prod_mean_v_0],
                  'Region_1':[prod_total_v_1, prod_mean_v_1], 
                  'Region_2':[prod_total_v_2, prod_mean_v_2]}

f_prod_df_v = pd.DataFrame(data=forecast_prod_v)
f_prod_df_v


# In[23]:


# DF Resumen de ingresos y beneficios por región
forecast_rev_v = {'concepto':["Ingresos potenciales", "Beneficio Bruto", "Margen Bruto"],
                  'Region_0':[revenue_total_v_0, gross_profit_v_0, gross_margin_v_0],
                  'Region_1':[revenue_total_v_1, gross_profit_v_1, gross_margin_v_1],
                  'Region_2':[revenue_total_v_2, gross_profit_v_2, gross_margin_v_2]}

f_rev_df_v = pd.DataFrame(data=forecast_rev_v)
f_rev_df_v


# #### Conclusión Paso extra 4.
# 
# Aplicando el mismo procedimiento de volúmenes e ingresos posibles a los 200 pozos elegidos en sus valores predichos y en sus valores verdaderos, podemos observar que los volúmenes totales son significativamente menores en Reg0 y Reg2, en cuanto a la información verdadera.
# 
# Esto nos dice que los modelos entrenados y aplicados a dichas regiones, sobrevaloraron las predicciones, no así para Reg1. Asumiéndose, según la información de las métricas, como la mejor región para invertir.
# 
# Cabe señalar que los pozos de producción máxima de la Reg1 (137 k) quedan por debajo de las Reg0 y Reg2 (185 k y 190 k) y a su vez mayor cantidad de pozos de baja producción (Medianas: Reg0=90 k, Reg1= 80 k, Reg2= 94 k), pero esto último cambia su afectación al limitarnos a 200 pozos.
# 
# Reg1 no presenta diferencia importante entre el volumen de producción predicho y verdadero (27,000 k).
# 
# En la comparación del margen bruto, se obtiene la misma afectación que en volúmenes, Reg1 se mantiene muy similar en ambos casos.
# 
# La región 1 es la mejor opción para invertir.
# 

# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Buen trabajo. Con esto ya tenemos una mejor idea del resultado real cuando escogemos los mejores pozos mediante el valor de predicción.
# 
# </div>

# ## Calcula riesgos y ganancias para cada región:
# 
# ### Utilizando las predicciones que almacenaste en el paso 4.2, emplea la técnica del bootstrapping con 1000 muestras para hallar la distribución de los beneficios.

# In[24]:


# Aplicación de Función select_ensemb para conjunto completo
select_all_p_v_df_0 = select_ensemb (predict_valid_0, features_valid_0, len(predict_valid_0), target_valid_0)
select_all_p_v_df_1 = select_ensemb (predict_valid_1, features_valid_1, len(predict_valid_1), target_valid_1)
select_all_p_v_df_2 = select_ensemb (predict_valid_2, features_valid_2, len(predict_valid_2), target_valid_2)


# In[25]:


#Función para determinar bootstraping
def sub_sample (selection_data, count, repet, points, oil_well_unitprice, investment):
    state = np.random.RandomState(12345)    
    values_rev = []
    for i in range(repet):
        data_subsample = selection_data.sample(n=points, replace=True, random_state=state)
        vol_rev = data_subsample.sort_values(by='predicted',ascending=False).head(count)['true'].sum()
        values_rev.append((vol_rev * oil_well_unitprice) - investment)
    return values_rev


# In[26]:


# Aplicación de función con bootstrapping
values_allv_rev_0 = sub_sample (select_all_p_v_df_0, oil_well_budg, 1000, 500, oil_well_unit_price, tot_investment)
values_allv_rev_1 = sub_sample (select_all_p_v_df_1, oil_well_budg, 1000, 500, oil_well_unit_price, tot_investment)
values_allv_rev_2 = sub_sample (select_all_p_v_df_2, oil_well_budg, 1000, 500, oil_well_unit_price, tot_investment)


# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Bien con el procedimiento de bootstrapping, pero para esta parte se requiere almacenar la ganancia de cada iteración, es decir, el ingreso menos la inversión. Además, para ese cálculo, se debe proceder de la misma forma que antes, es decir, seleccionar el top 200 por valor de predicción y luego usar los valores reales correspondientes para el cálculo del ingreso.
# 
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante</b> <a class="tocSkip"></a>
#     
# He descontado el monto de la inversión y la aplicación a los valores reales.
#     
# </div>

# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor v2</b> <a class="tocSkip"></a>
# 
# El problema con esta parte es que estás usando como dato fijo el top 200 de los pozos. En realidad, debes usar una función que de forma dinámica, con una data nueva, obtenga el beneficio según la metodología. En detalle, para la región 0 por ejemplo, veo que más adelante usas:
#     
#     values_b200v_rev_0 = sub_sample (select_200_valid_0, oil_well_budg, 1000, 500)
#     
# pero select_200_valid_0 ya tiene 200 pozos, lo cual no genera mucha variación en el método de bootstrapping y tampoco se está obteniendo la variabilidad completa de la región. Para ello, el bootstrapping debería recibir los 25000 pozos de validación, con sus valores reales y de predicción, realizar el muestreo de 500 y sobre eso aplicar la metodología de cálculo del beneficio. Algo como esto:
#     
#     state = np.random.RandomState(12345) 
#     values_rev=[]
#     for _ in range(1000):
#         data_subsample = df_data.sample(n=500, replace=True, random_state=state)
#         vol_rev = data_subsample.sort_values(by='pred',ascending=False).head(200)['real'].sum()
#         values_rev.append(vol_rev*4500-100_000_000)
# 
# donde df_data tiene 25000 filas (pozos) con columnas de real y pred, que contienen los valores reales y de predicción respectivamente para cada pozo.
#     
#     
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante v2</b> <a class="tocSkip"></a>
#     
# * Al crear el DF con valores predichos y verdaderos me generaron problema algunos pozos duplicados (los eliminé desde la sección uno, me aseguré de que la afectación no fuera importante)
# 
# * La formación del DF la agregué al paso 4.1 ya que por cómo había trabajado la sección 4, requería conocer los valores por dichas selecciones, para hacer la comparativa (previa al bootstrapping) de los valores predichos y verdaderos, pues ya lo había incluido en el flujo de la información desde las primeras entregas.
#     
# </div>

# ### Encuentra el beneficio promedio, el intervalo de confianza del 95% y el riesgo de pérdidas. La pérdida es una ganancia negativa, calcúlala como una probabilidad y luego exprésala como un porcentaje.

# In[27]:


#Función para presentar gráfico e información de ingresos
def rev_summ (data_rev, repet, reg):
    values_rev_s = pd.Series(data_rev)
    values_rev_min = values_rev_s.min()
    values_rev_max = values_rev_s.max()
    values_rev_mean = values_rev_s.mean()
    values_rev_std = values_rev_s.std()
    lower = values_rev_s.quantile(.025)
    upper = values_rev_s.quantile(0.975)
    neg_rev = 0
    for i in values_rev_s:
        if i < 0:
            neg_rev += 1
    loss_rate = neg_rev / repet * 100
    plt.figure(figsize=(10, 5))
    plt.axvline(lower, label= 'IC: Límite Inferior', color='deeppink', linestyle='--')
    plt.axvline(upper, label= 'IC: Límite Superior', color='deeppink', linestyle='--')
    plt.axvline(values_rev_mean, label= 'Media', color='lime', linestyle='-')
    plt.hist(values_rev_s, bins=100, alpha=.6)
    plt.title("Distribución Muestral de Utilidad")
    plt.xlabel("\nUtilidad de las Muestras")
    plt.ylabel("Frecuencia")
    plt.show()    
    
    return print(f'Utilidad {reg} \nValor de utilidad mínimo: {values_rev_min:.2f} y máximo: {values_rev_max:.2f} \nUtilidad promedio: {values_rev_mean:.2f} \nDesviación estándar: {values_rev_std:.2f} \nIntervalo de confianza del 95 %: {lower:.2f} , {upper:.2f} \nPorcentaje de probabilidad de pérdida {loss_rate:.2f}%')


# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# El intervalo de confianza correcto es el de los cuantiles. De hecho, nota que el calculado mediante a la distribución t-student no hace sentido en el gráfico. Esto ocurre puesto que no podemos asumir que la distribución de la variable aleatoria es de ese tipo, más bien nos apoyamos en el resultado del bootstrapping para obtener un intervalo de confianza según la distribución empírica.
#     
# Por otro lado, el resto de los cálculos en el gráfico está bien. Una vez corregido el bootstrapping se deberían obtener los resultados correctos. 
#     
# Respecto al texto de output, se debe corregir el intervalo de confianza (como indiqué arriba) y el riesgo se calcula como el porcentaje de veces que se obtuvo un resultado negativo.
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta estudiante</b> <a class="tocSkip"></a>
#     
# Eliminé el cálculo con t-student, dejando los cuantiles como el intervalo de confianza y agregué un bucle for para el conteo de los submuestreos negativos y el cálculo del porcentaje de pérdida.
#     
# </div>

# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor v2</b> <a class="tocSkip"></a>
# 
# Muy bien, corregidos los cálculos de métricas. Ahora tan solo falta corregir el cálculo del bootstrapping que expliqué en más detalle arriba.
# </div>

# In[28]:


# Aplicación de función de resumen de Utilidad
print(rev_summ (values_allv_rev_0, 1000, 'Región 0'))
print(rev_summ (values_allv_rev_1, 1000, 'Región 1'))
print(rev_summ (values_allv_rev_2, 1000, 'Región 2'))


# #### Comentarios Sección 5
# 
# Al aplicar el Bootstrapping a los valores verdaderos de los mejores 200 pozos predichos por región, con muestras a 500 puntos y 1000 repeticiones, creamos nuestros nuevos universos de datos.
# 
# La distribución de datos en los tres casos muestra la forma de "campana" se presentan los datos de la Utilidad.
# 
# En este paso se agrega la consideración de la probabilidad de obtener pérdida, siendo Reg0 y Reg2 cuyos porcentajes de pérdida sobrepasan el 2.5% requerido para su descarte.
# 
# En Reg0 y Reg2 la utilidad promedio refleja una disminución de entre 7% y 10% en comparación con los valores predichos debido a la baja eficiencia de los modelos para cada región. Sin embargo, considerando dicha disminución ambos superan el porcentaje de utilidad de Reg1, por lo mismo, si se desea un mejor porcentaje de ganancias (sin certeza en los cálculos y valores a obtener) estas dos regiones podrían ser buenas para invertir.
# 
# Por otro lado Reg0 no tiene variaciones importantes entre los valores predichos y verdaderos, asegurando un margen de utilidad del 24%.
# 
# Dicho lo anterior y a efectos de considerar un margen de ganancia conocido y certeza en los datos para su consecución, Reg1 es la mejor opción para invertir.

# ### Presenta tus conclusiones: propón una región para el desarrollo de pozos petrolíferos y justifica tu elección. ¿Coincide tu elección con la elección anterior en el punto 4.3?

# #### Conclusión General.
# 
# A pesar de que en la información presentada Reg0 y Reg2 parecen ofrecer un mejor margen de bruto de utilidad, la Región 1  presenta mejores resultados en las métricas del modelo por lo tanto, resultados predichos más apegados a los valores verdaderos, esto da más efectividad al elegir los 200 pozos mejor predichos y del mismo modo, de mejor producción real.
# 
# Se recomienda invertir en la REGIÓN 1.
# 

# <div class="alert alert-block alert-success"> <b>¡Excelente trabajo, Oscar! 🎉</b> <a class="tocSkip"></a> Tu proyecto ha mejorado significativamente con cada iteración de revisión. 
#     
# Lograste **corregir los cálculos del bootstrapping, mejorar la estimación del intervalo de confianza y evaluar correctamente el riesgo de inversión**.
# 
# 🔹 Fortalezas destacadas:
#     
# ✅ Implementación correcta del modelo de regresión lineal.
# ✅ Cálculo preciso del beneficio y riesgo utilizando bootstrapping.
# ✅ Justificación bien fundamentada sobre la mejor región para invertir.
# ✅ Aplicación correcta de cuantiles en la estimación del intervalo de confianza.
# 
# 🚀 Sigue aplicando este nivel de detalle y rigor en tus análisis, ya que esto te preparará bien para el mundo real en Data Science!
# 
# </div>
