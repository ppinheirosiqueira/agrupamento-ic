import scipy.io
from agrupador import gustafson_kessel
from utils import acuracia_e_desvio_padrao, qual_cluster, ajustar_rótulos, matriz
import numpy as np

def aproximacao(data):
    data_sem_classes = data[:, :-1]

    num_clusters = int(1 + max(data[:,data.shape[1]-1]))
    memberships, centers = gustafson_kessel(data_sem_classes, num_clusters)

    previstos = qual_cluster(memberships)
    previstos_ajustados = ajustar_rótulos(data[:,data.shape[1]-1], previstos)

    print("Centros:\n", centers)
    print("Graus de Pertinência:\n", memberships)
    print("Previstos:\n", previstos)
    print("Previstos ajustados:\n", previstos_ajustados)
    matriz_confusao = matriz(data[:,data.shape[1]-1],previstos_ajustados,num_clusters)
    with np.printoptions(precision=0, suppress=True, formatter={'all': lambda x: str(int(x))}):
        print("Matriz de Confusão:\n", matriz_confusao)
    acuracia, desvio = acuracia_e_desvio_padrao(matriz_confusao)
    print(f"Acurácia: {acuracia}%")
    print("Desvio Padrão: ", desvio)
    return centers, acuracia, desvio, matriz_confusao

def testar_30_vezes(arquivo):
    dados = scipy.io.loadmat(arquivo)
    data = dados['data']  
    acuracias = []
    desvios = []
    matrizes = []
    centros = []
    index_70 = int(0.7 * len(data))
    for i in range(2):
        np.random.shuffle(data)
        amostras_treinamento, amostras_testes = data[:index_70], data[index_70:]
        center, acu, des, mc = aproximacao(amostras_treinamento)
        # ACURÁCIA DO TREINAMENTO ESTÁ MUITO BAIXA
        # PRECISO FAZER OS TESTES AQUI COM AS AMOSTRAS TESTES
        centros.append(center)
        acuracias.append(acu)
        desvios.append(des)
        matrizes.append(mc)
    print(f'Melhor Acurácia: {max(acuracias)}')
    print(f'Pior Acurácia: {min(acuracias)}')

def __main__():
    aux = True
    while aux:
        print("Qual arquivo deseja utilizar?")
        print("1 - Adult.mat")
        print("2 - Dry_bean.mat")
        print("S - Sair")
        arquivo = input("1, 2 ou S: ")
        while arquivo != "1" and arquivo != "2" and arquivo != "S" and arquivo != "s":
            arquivo = input("1 ou 2 ou S: ")
        if arquivo == "s" or arquivo == "S":
            break
        elif arquivo == "1":
            testar_30_vezes("Adult.mat")
        elif arquivo == "2":
            testar_30_vezes("Dry_bean.mat")

if __name__ == "__main__":
    __main__()