import numpy as np

def gustafson_kessel(data, num_clusters, max_iter=100, m=2.0, tol=1e-4):
    n_samples, n_features = data.shape

    # Inicializar a matriz de pertinência (U) aleatoriamente
    U = np.random.rand(n_samples, num_clusters)
    U = U / np.sum(U, axis=1, keepdims=True)

    # Inicializar os centróides dos clusters
    centers = np.random.rand(num_clusters, n_features)

    for iteration in range(max_iter):
        print(f"Iteração {iteration}")
        U_old = U.copy()

        # Atualizar centróides
        for j in range(num_clusters):
            numerador = np.sum((U[:, j]**m)[:, np.newaxis] * data, axis=0)
            denominador = np.sum(U[:, j]**m)
            centers[j] = numerador / denominador

        # Atualizar matrizes de covariância e pertinência
        F = []
        for j in range(num_clusters):
            diff = data - centers[j]
            U_m = U[:, j]**m
            cov_matrix = (U_m[:, np.newaxis] * diff).T @ diff / np.sum(U_m)
            F.append(cov_matrix)

        F = np.array(F)

        for i in range(n_samples):
            for j in range(num_clusters):
                diff = data[i] - centers[j]
                inv_F = np.linalg.inv(F[j] + np.eye(n_features) * 1e-6)  # Regularização para estabilidade
                mahalanobis_dist = diff @ inv_F @ diff.T

                denom = 0
                for k in range(num_clusters):
                    diff_k = data[i] - centers[k]
                    inv_F_k = np.linalg.inv(F[k] + np.eye(n_features) * 1e-6)
                    mahalanobis_dist_k = diff_k @ inv_F_k @ diff_k.T
                    denom += (mahalanobis_dist / mahalanobis_dist_k)**(1 / (m-1))

                U[i, j] = 1 / denom

        # Checar critério de parada
        if np.linalg.norm(U - U_old) < tol:
            break

        # print("Centroides:", centers)
        # print("Covariâncias (F):", F)
        # print("Pertinências (U):", U)

    return U, centers
