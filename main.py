import scipy.io
from agrupador import gustafson_kessel
from utils import acuracia_e_desvio_padrao, ajustar_rótulos, matriz
import numpy as np

def aproximacao(data):
    index_70 = int(0.7 * len(data))
    amostras_treinamento, amostras_testes = data[:index_70], data[index_70:]

    data_sem_classes = amostras_treinamento[:, :-1]

    num_clusters = int(1 + max(data[:,data.shape[1]-1]))
    F, centers = gustafson_kessel(data_sem_classes, num_clusters)

    preditos_testes = []
    for point in amostras_testes[:, :-1]:
        distances = []
        for k in range(len(centers)):
            diff = point - centers[k]
            inv_cov = np.linalg.inv(F[k] + np.eye(len(point)) * 1e-6)
            mahalanobis_dist = diff.T @ inv_cov @ diff
            distances.append(mahalanobis_dist)
        preditos_testes.append(np.argmin(distances))

    preditos_arrumados = ajustar_rótulos(amostras_testes[:,data.shape[1]-1],preditos_testes)
    matriz_confusao = matriz(amostras_testes[:,data.shape[1]-1],preditos_arrumados,num_clusters)
    with np.printoptions(precision=0, suppress=True, formatter={'all': lambda x: str(int(x))}):
        print("Matriz de Confusão:\n", matriz_confusao)
    acuracia, desvio = acuracia_e_desvio_padrao(matriz_confusao)
    print(f"Acurácia: {acuracia}%")
    print("Desvio Padrão: ", desvio)

    return centers, F, acuracia, desvio, matriz_confusao

def testar_30_vezes(arquivo):
    dados = scipy.io.loadmat(arquivo)
    data = dados['data']  
    acuracias = []
    desvios = []
    matrizes = []
    centros = []
    Fs = []
    for i in range(1,31,1):
        print(f"Iteração {i}")
        np.random.shuffle(data)
        center, F, acu, des, mc = aproximacao(data)

        centros.append(center)
        Fs.append(F)
        acuracias.append(acu)
        desvios.append(des)
        matrizes.append(mc)

    best_index = np.argmax(acuracias)
    best_center = centros[best_index]
    best_F = Fs[best_index]

    print(f'Melhor Acurácia: {max(acuracias)}')
    print(f'Pior Acurácia: {min(acuracias)}')

    salvar_arquivo = f'melhores_centros_{arquivo.split(".")[0]}.npz'
    np.savez(salvar_arquivo, best_center=best_center, best_F=best_F)
    print(f"Melhores dados salvos no arquivo: {salvar_arquivo}\n\n")

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