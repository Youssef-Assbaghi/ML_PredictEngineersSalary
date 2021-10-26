# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:46:36 2021

@author: Youssef
"""
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from Regressor import Regressor
pd.options.mode.chained_assignment = None

from sklearn import preprocessing

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

def drop_data(dataset):
    dataset.drop('12board',inplace=True,axis=1)
    dataset.drop('10board',inplace=True,axis=1)
    dataset.drop('ID',inplace=True,axis=1) #Id no necesario 
    dataset.drop('CollegeID',inplace=True,axis=1) # Id no necesario
    dataset.drop('CollegeCityID',inplace=True,axis=1) #ID no necesario
    dataset.drop('12percentage',inplace=True,axis=1) #Lo mismo que 10 percentageno es necesario
    #dataset.drop('Degree',inplace=True,axis=1)
    #dataset.drop('Specialization',inplace=True,axis=1)
    dataset.drop('CollegeState',inplace=True,axis=1)
    
    """
    Valores nullos (No se han presentado al examen)
    ElectronicsAndSemicon    2133
    ComputerScience          2298
    MechanicalEngg           2811
    ElectricalEngg           2876
    TelecomEngg              2724
    CivilEngg                2972
    """
    dataset.drop('ElectronicsAndSemicon',inplace=True,axis=1)
    dataset.drop('MechanicalEngg',inplace=True,axis=1)
    dataset.drop('TelecomEngg',inplace=True,axis=1)
    dataset.drop('CivilEngg',inplace=True,axis=1)
    dataset.drop('ComputerScience',inplace=True,axis=1)
    dataset.drop('ElectricalEngg',inplace=True,axis=1)
    return dataset
    #dataset.drop('DOB',inplace=True,axis=1)
    #dataset.drop('Gender',inplace=True,axis=1)

def check_nulls(dataset):
    dataset.replace(-1, np.NaN,inplace=True)
    print("Per comptar el nombre de valors no existents:")
    #dataset.drop('ID',inplace=True,axis=1)
    #print(dataset.isnull().sum())
    return dataset

def fill_nulls_by_mean(dataset,missing_values_columns):
    data = dataset.copy()
    '''Filling missing values with mean'''
    for col in missing_values_columns:
        data[col] = data[col].fillna(data[col].mean())
     
    return data

def change_DOB_to_age(dataset):
    dataset['DOB']=((pd.to_datetime('today') - pd.to_datetime(list(dataset['DOB']))).days / 365).astype(int)
    return dataset

def set_gender_to_binary(dataset):
    dataset.loc[dataset.Gender=="f","Gender"]=0
    dataset.loc[dataset.Gender=="m","Gender"]=1
    #print (dataset.head)
    return dataset
def map_to_other_specialization(var):
    ''' if count of unique category is less than 10, replace the category as other '''
    value_count = dataset['Specialization'].value_counts()
    
    if var in value_count[value_count<=10]:
        return 'other'
    else:
        return var
def normalize (dataset2):
    x=dataset2.values
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scale=min_max_scaler.fit_transform(x)
    df=pd.DataFrame(x_scale,columns=dataset2.columns)
    return df

def mean_squeared_error(y1, y2):
    # comprovem que y1 i y2 tenen la mateixa mida
    assert(len(y1) == len(y2))
    mse = 0
    for i in range(len(y1)):
        mse += (y1[i] - y2[i])**2
    return mse / len(y1)

def regression(x, y): # Creem un objecte de regressió de sklearn regr = LinearRegression()
    regr = LinearRegression()
    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def mse(v1, v2):
    return ((v1 - v2)**2).mean()

if __name__ == '__main__':
    file="Engineering_graduate_salary.csv"
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    # Carreguem dataset d'exemple
    dataset = load_dataset('Engineering_graduate_salary.csv')
    data = dataset.values
    dataset=drop_data(dataset)
    x = data[:, :2]
    y = data[:, 2]
    dataset=check_nulls(dataset)
    missing_values = [col for col in dataset.columns if dataset.isnull().sum()[col] > 0]
    dataset = fill_nulls_by_mean(dataset,missing_values)
    dataset=change_DOB_to_age(dataset)
    dataset=set_gender_to_binary(dataset)
    
    #Hacemos replace de la misma categoria pero con diferente nombre
    dataset['Specialization'] = dataset['Specialization'].str.replace('electronics & instrumentation eng',\
                                                'electronics and instrumentation engineering')
    
    dataset['Specialization'] = dataset.Specialization.apply(map_to_other_specialization)
    #print(dataset.Specialization.value_counts())
    #NORMALIZAR LAS VARIABLES
    dataset2=dataset.copy()
    dataset2.drop('Degree',inplace=True,axis=1)
    dataset2.drop('Specialization',inplace=True,axis=1)
    dataset2.drop('nueroticism',inplace=True,axis=1)
    dataset2.drop('Gender',inplace=True,axis=1)
    dataset2.drop('CollegeTier',inplace=True,axis=1)
    dataset2.drop('CollegeCityTier',inplace=True,axis=1)
    dataset2.drop('GraduationYear',inplace=True,axis=1)
    dataset2.drop('openess_to_experience',inplace=True,axis=1)
    dataset2.drop('12graduation',inplace=True,axis=1)
    dataset2.drop('agreeableness',inplace=True,axis=1)
    dataset2.drop('extraversion',inplace=True,axis=1)
    dataset2.drop('conscientiousness',inplace=True,axis=1)
    #print(dataset.dtypes)
    df=normalize(dataset2)
    #print(df.dtypes)

    train,test=train_test_split(df,test_size=0.1)
    entrenar,validar=train_test_split(train,test_size=0.3)  
    y = entrenar['Salary'].to_numpy()
    entrenar.drop('Salary',inplace=True,axis=1)
    regr = regression(entrenar.to_numpy(),y)
    
    y = validar['Salary'].to_numpy()
    X=y.mean()
    validar.drop('Salary',inplace=True,axis=1)
    predicted = regr.predict(validar.to_numpy())
    dif = y - predicted
    # Mostrem l'error (MSE i R2)
    s=Regressor(entrenar.DOB, entrenar['10percentage'], entrenar.collegeGPA , entrenar.English , entrenar.Logical , entrenar.Quant , entrenar.Domain , entrenar.ComputerProgramming , 0.01)
    MSE = mse(y, predicted)
    r2 = r2_score(y, predicted)
    
    print("Mean squeared error: ", MSE, " La media de salarios normalizado: ", X)
    VAR= 100*MSE/X
    print("El porcentaje de error general es ",VAR)
    print("R2 score: ", r2)
    
    print ("TEST")
    y = test['Salary'].to_numpy()
    X=y.mean()
    test.drop('Salary',inplace=True,axis=1)
    predicted = regr.predict(test.to_numpy())
    dif = y - predicted
    # Mostrem l'error (MSE i R2)
    MSE = mse(y, predicted)
    r2 = r2_score(y, predicted)
    
    print("Mean squeared error: ", MSE, " La media de salarios normalizado tiene un valor de ", X)
    VAR= 100*MSE/X
    print("El porcentaje de error general es ",VAR)
    print("R2 score: ", r2)
    
    #s=Regressor(df.DOB, df['10percentage'], df.collegeGPA , df.English , df.Logical , df.Quant , df.Domain , df.ComputerProgramming , df.salary , 0.01)
    #s=Regressor(df.DOB , df.10percentage , df.collegeGPA , df.English , df.Logical , df.Quant , df.Domain , df.ComputerProgramming , df.salary , 0.01)
    
    
    
    print("Dimensionalitat de la BBDD:", df.shape)
    print("Dimensionalitat de les entrades X", x.shape)
    print("Dimensionalitat de l'atribut Y", y.shape)

# Funcio per a llegir dades en format csv
"""
Haciendo drop de variables que no usamos
Mean squeared error:  0.002600512805967778
R2 score:  0.1579873287617345

sinhacer drop de variables que no usamos
Mean squeared error:  0.0017761771482014727
R2 score:  0.20151831505094409



Per comptar el nombre de valors no existents:
Mean squeared error:  0.0028147319620326675  La media de salarios normalizado:  0.06911680911680913
El porcentaje de error general es  4.072427529569111
R2 score:  0.1486233428189837
TEST
Mean squeared error:  0.002619040842244249  La media de salarios normalizado tiene un valor de  0.07349726775956285
El porcentaje de error general es  3.5634533392616916
R2 score:  0.06948972599665348
Dimensionalitat de la BBDD: (2998, 9)
Dimensionalitat de les entrades X (2998, 2)
Dimensionalitat de l'atribut Y (300,)



Per comptar el nombre de valors no existents:
Mean squeared error:  0.0028244566304016444  La media de salarios normalizado:  0.06914670029424129
El porcentaje de error general es  4.084730895881769
R2 score:  0.155425809185748
TEST
Mean squeared error:  1.6220724521919403  La media de salarios normalizado tiene un valor de  0.06707440100882724
El porcentaje de error general es  2418.318207535643
R2 score:  -562.2355305378046
Dimensionalitat de la BBDD: (2998, 19)
Dimensionalitat de les entrades X (2998, 2)
Dimensionalitat de l'atribut Y (300,)
"""

