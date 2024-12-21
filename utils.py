import numpy as np
import numpy as np
from scipy.optimize import linear_sum_assignment

def ajustar_rótulos(reais, previstos):
    classes_reais = np.unique(reais)
    classes_previstas = np.unique(previstos)
    
    # Criar a matriz de confusão
    matriz_confusao = np.zeros((len(classes_reais), len(classes_previstas)))
    for i, real in enumerate(classes_reais):
        for j, previsto in enumerate(classes_previstas):
            matriz_confusao[i, j] = np.sum((reais == real) & (previstos == previsto))
    
    # Resolver o problema de atribuição com o Algoritmo Húngaro
    linha_ind, coluna_ind = linear_sum_assignment(-matriz_confusao)  # Negativo para maximizar
    
    # Criar um mapeamento de classes
    mapeamento = {classes_previstas[coluna]: classes_reais[linha] for linha, coluna in zip(linha_ind, coluna_ind)}
    
    # Ajustar os rótulos previstos
    previstos_ajustados = np.array([mapeamento[r] for r in previstos])
    
    return previstos_ajustados

def acuracia_e_desvio_padrao(matriz_confusao):
    acuracia = 100*np.trace(matriz_confusao) / np.sum(matriz_confusao)
    desvio_padrao = np.std(matriz_confusao, ddof=1)  
    return acuracia, desvio_padrao

def matriz(reais, previstos, tamanho):
    matriz_confusao = np.zeros((tamanho,tamanho))
    for i in range(len(reais)):
        matriz_confusao[int(reais[i])][int(previstos[i])] += 1
    
    return matriz_confusao

def matriz_pertinencias(amostras_testes, centros):
    print(amostras_testes)