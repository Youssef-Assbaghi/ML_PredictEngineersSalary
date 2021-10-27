import copy as cp
import numpy as np
from matplotlib import pyplot as plt
class Regressor(object):
    #Inicialziació de variables del regressor
    def __init__(self, w, alpha, train, y):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        self.w = cp.deepcopy(w) # [w0, ...w8]
        self.alpha = alpha
        self.train = cp.deepcopy(train)
        self.errores = []
        self.y = y
        
    #Funcio de predicció
    def predict(self, x):
        columna = 0
        datos = cp.deepcopy(x)
        for indice in range(1,len(self.w)):
           datos[:,columna] *= self.w[indice]
           columna += 1
        datos += self.w[0]
        predicciones = np.sum(datos, axis = 1)
        return predicciones
        pass
    #Funcion que calcula el error cuadratico de la prediccion
    def calcularError(self,y_validarPred, y):
        restas = np.add(y_validarPred,-y)
        cuadrados = np.power(restas,2)
        costeTotal = (1/(len(y))) * np.sum(cuadrados)
        return costeTotal
    #Funcion que recalcula los pesos de las w
    def __update(self, hy, y):
        # actualitzar aqui els pesos donada la prediccio (hy) i la y real.
        restas = np.add(hy,-y)
        cuadrados = np.power(restas,2)
        costeTotal = (1/(len(y))) * np.sum(cuadrados) #Creo que len(y) = m
        self.errores.append(costeTotal)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.alpha * (costeTotal + (self.epsilon/len(y))*self.w[i])
        pass
    
    def trains(self, max_iter, epsilon):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        self.epsilon = epsilon
        for i in range(max_iter):
            prediccio = self.predict(self.train)
            self.__update(prediccio, self.y)
        plt.figure()
        plt.title("Model de Entrenar")
        plt.xlabel("Iteracions")
        plt.ylabel("Error")
        
        plt.scatter(range(max_iter),self.errores)

        pass