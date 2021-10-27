from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
pd.options.mode.chained_assignment = None #Quitamos los warnings por usar copy en dataframe
from Regressor import Regressor

from sklearn import preprocessing

#Funcion que lee el dataset usando pandas
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Funcion para eliminar datos innecesarios
def drop_data(dataset):

    dataset.drop('ID',inplace=True,axis=1) #Id no necesario 
    dataset.drop('CollegeID',inplace=True,axis=1) # Id no necesario
    dataset.drop('CollegeCityID',inplace=True,axis=1) #ID no necesario
    dataset.drop('12percentage',inplace=True,axis=1) #Lo mismo que 10 percentageno es necesario
    dataset.drop('CollegeState',inplace=True,axis=1)
    
    #Datos que mayoritariamente son nulos o -1 
    dataset.drop('ElectronicsAndSemicon',inplace=True,axis=1)
    dataset.drop('MechanicalEngg',inplace=True,axis=1)
    dataset.drop('TelecomEngg',inplace=True,axis=1)
    dataset.drop('CivilEngg',inplace=True,axis=1)
    dataset.drop('ComputerScience',inplace=True,axis=1)
    dataset.drop('ElectricalEngg',inplace=True,axis=1)
    
    # Datos que no tienen una correlacion estrecha con el salario
    dataset.drop('12board',inplace=True,axis=1)
    dataset.drop('10board',inplace=True,axis=1)
    dataset.drop('Degree',inplace=True,axis=1)
    dataset.drop('Specialization',inplace=True,axis=1)
    dataset.drop('nueroticism',inplace=True,axis=1)
    dataset.drop('Gender',inplace=True,axis=1)
    dataset.drop('CollegeTier',inplace=True,axis=1)
    dataset.drop('CollegeCityTier',inplace=True,axis=1)
    dataset.drop('GraduationYear',inplace=True,axis=1)
    dataset.drop('openess_to_experience',inplace=True,axis=1)
    dataset.drop('12graduation',inplace=True,axis=1)
    dataset.drop('agreeableness',inplace=True,axis=1)
    dataset.drop('extraversion',inplace=True,axis=1)
    dataset.drop('conscientiousness',inplace=True,axis=1)
    
    return dataset


def check_nulls(dataset):
    return dataset.replace(-1, np.NaN,inplace=True)
     

def fill_nulls_by_mean(dataset,missing_values_columns):
    data = dataset.copy()
    '''Filling missing values with mean'''
    for col in missing_values_columns:
        data[col] = data[col].fillna(data[col].mean())
     
    return data

def change_DOB_to_age(dataset):
    dataset['DOB']=((pd.to_datetime('today') - pd.to_datetime(list(dataset['DOB']))).days / 365).astype(int)
    return dataset

#Damos formato de genero a valor numerico
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
#funcion que normaliza los datos del dataset
def normalize (dataset2):
    x=dataset2.values
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scale=min_max_scaler.fit_transform(x)
    df=pd.DataFrame(x_scale,columns=dataset2.columns)
    return df

#Funcion que calcula el mean squared error
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

#Funcion que calcula el mean squared error
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
    #dataset=set_gender_to_binary(dataset)
    #Hacemos replace de la misma categoria pero con diferente nombre finalmente no es necesario
    #ya que no tiene relacion direcata con salario
    #dataset['Specialization'] = dataset['Specialization'].str.replace('electronics & instrumentation eng',\
    #                                            'electronics and instrumentation engineering')
    #dataset['Specialization'] = dataset.Specialization.apply(map_to_other_specialization)
    #print(dataset.Specialization.value_counts())
    
    #NORMALIZAR LAS VARIABLES
    df=normalize(dataset)


    train,test=train_test_split(df,test_size=0.1)
    entrenar,validar=train_test_split(train,test_size=0.3)  
    y = entrenar['Salary'].to_numpy()
    entrenar.drop('Salary',inplace=True,axis=1)
    regr = regression(entrenar.to_numpy(),y)
    

    w=np.random.uniform(0.0001,0.5,9) # Generacion de W con random
    s=Regressor(w, 0.0001,entrenar.to_numpy(),y)
    s.trains(10000,0.1)
    
    y = validar['Salary'].to_numpy()
    X=y.mean()
    validar.drop('Salary',inplace=True,axis=1)
    predicciones = s.predict(validar.to_numpy())
    errorVal = s.calcularError(predicciones,y)
    print("El error que nos da con el conjunto de validacion es de: ", errorVal)
    predicted = regr.predict(validar.to_numpy())
    dif = y - predicted
    # Mostrem l'error (MSE i R2)

    MSE = mse(y, predicted)
    r2 = r2_score(y, predicted)
    VAR= 100*MSE/X
    
    print("Mean squeared error: ", MSE, " La media de salarios normalizado: ", X)
    print("El porcentaje de error general es ",VAR)
    print("R2 score: ", r2)
    
    print ("TEST")
    y = test['Salary'].to_numpy()
    X=y.mean()
    test.drop('Salary',inplace=True,axis=1)
    predicciones = s.predict(test.to_numpy())
    errorVal = s.calcularError(predicciones,y)
    print("El error que nos da el error de test es: ", errorVal)
    predicted = regr.predict(test.to_numpy())
    dif = y - predicted
    # Mostrem l'error (MSE i R2)
    MSE = mse(y, predicted)
    r2 = r2_score(y, predicted)
    VAR= 100*MSE/X
    
    print("Mean squeared error: ", MSE, " La media de salarios normalizado tiene un valor de ", X)
    print("El porcentaje de error general es ",VAR)
    print("R2 score: ", r2)
    
