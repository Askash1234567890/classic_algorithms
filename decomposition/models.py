import numpy as np

class PCA():
    def __init__(self, k: int):
        self.k = k
    
    def fit(self, x):
        # центруем и запоминаем среднее для будущих преобразований
        self.x_mean = x.mean(axis=0)
        x_centered = x - self.x_mean

        # считаем матрицу ковариации и собственные вектора и значения
        cov = np.cov(x_centered, rowvar=False)
        eig_values, eig_vectors = np.linalg.eigh(cov)

        # сортируем по убыванию дисперсии
        ind = eig_values.argsort()[::-1]
        eig_values = eig_values[ind]
        eig_vectors = eig_vectors[:, ind]

        # получаем первые k собственных векторов
        principal_vectors = eig_vectors[:, :self.k]
        self.principal_vectors = principal_vectors

        # print(f"собственные значения: {eig_values}")
        # print(eig_vectors.T.dot(x_centered.dot(eig_vectors)))

    def transform(self, x):
        # получаем новое пространство
        return (x - self.x_mean).dot(self.principal_vectors)
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    