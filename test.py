
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def replace_text(listae):
    rango= {}
    contador = 0
    for item in listae:
        if not item in rango:
            rango[item] = contador
            contador += 1
    salida = []        
    for item in listae:
        salida.append(int(rango[item]))
    return salida
            
        
    
    
    
datos = pd.read_csv('/home/arturo/Vídeos/NataSquad-AI-Hackathon/1_Machine_Learning/1_Customer_Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
datos = datos.drop(columns=['customerID'])

    
datos.columns
ajustar = ['gender', 'Partner', 'Dependents','PhoneService', 'MultipleLines',
           'InternetService','OnlineSecurity', 'OnlineBackup','DeviceProtection',
           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
           'PaperlessBilling', 'PaymentMethod', 'Churn'         
           
           
           ]
for item in ajustar:
    print(item)
    print(datos[item][:10])
    datos[item] = replace_text(datos[item])
    
lista = []    
for item in datos['TotalCharges']:
    print(item)
    try:
        lista.append(float(item))
    except:
        lista.append(0.0)

datos['TotalCharges'] = lista
    
for item in datos:
    print(item)
    print(datos[item][:10])  
    
    
datos.head()    

#sns.heatmap(datos.corr(), cmap='coolwarm')
#sns.clustermap(datos.corr(),cmap='coolwarm')
#sns.clustermap(datos.corr(),cmap='coolwarm',standard_scale=1);

selected_features = ['OnlineSecurity', 'TechSupport', 'Contract','PaymentMethod', 
                     'tenure', 'TotalCharges']  
##     'gender', 'SeniorCitizen', 'Partner', 'Dependents',        
#        'DeviceProtection', 
#       'StreamingTV', 'StreamingMovies',  'PaperlessBilling',
#        'MonthlyCharges', 'OnlineBackup', 'PhoneService', 'MultipleLines', 'InternetService',

datos_x = datos[selected_features]
datos_y = datos['Churn']

  
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
datos_x = scaler.fit_transform(datos_x)

#Normalizando ouutput
#datos_y = datos_y.values.reshape(-1,1)
#datos_y = scaler.fit_transform(datos_y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(datos_x, datos_y, test_size = 0.2, random_state = 2)



print('X Train: {}, X Test: {}, y_train: {}, y_test: {}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

n_entradas = X_train.shape[1]

print(len(datos.columns))
##Definiendo modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(6 , )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.summary()

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics='accuracy')
#model.compile(loss='binary_crossentropy')

epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.2)

epochs_hist.history.keys()

#Grafico
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso del Modelo durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])


test_loss, test_accuracy = model.evaluate(X_test, y_test)
#


    
#Haciendo prediccion
#y_predict_1 = model.predict(X_test_1)
#print( model.predict(X_test_1), int(y_test[indice]))









