import numpy as np
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

np.set_printoptions(precision=2)

# data
datas = [
    np.array([
        [10,20,30,40],
        [12,21,30,40],
        [6,5,4,9],
        [12,6,6,10]
    ]),
    np.array([
        [10,20,30,40],
        [12,21,30,40],
        [6,5,4,9],
        [12,6,6,10],
        [12,6,6,99],
        [1200,600,600,9900]
    ]),
]

# normalize: default is L2 norm and axis=1 (= indep each sample)
norm_datas = [normalize(data) for data in datas]

for data, norm_data in zip(datas, norm_datas):

    for v, nv in zip(data, norm_data):
        print(f'v = {v} nv = {nv}')

    for n in [2, 3, 4]:

        print(f'--- components = {n}')

        # method chosen from
        # https://scikit-learn.org/stable/modules/clustering.html
        for model in [
            KMeans(n_clusters=n, random_state=42),
            GaussianMixture(n_components=n, random_state=42),
            GaussianMixture(n_components=n, init_params='kmeans', random_state=42)
            # https://towardsdatascience.com/gaussian-mixture-models-vs-k-means-which-one-to-choose-62f2736025f0
        ]:

            # do the clustering
            model.fit(norm_data)

            # get cluster ids
            res = model.predict(norm_data)

            print(f'--- model = {model}')
            for elem, cluster_id in zip(data, res):
                print(f'{cluster_id} <-- {elem}')

            # GM és data[0] és n=2 esetén:
            # [1, 1, 0, 0] -- kiváló! :)

