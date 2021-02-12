import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance
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
    np.array([ # here KMeans and GaussianMixture differs, if n = 2
               # both results are plausible :)
        [10,11,12],
        [10,12,12],
        [11,12,13],
        [11,11,13],
    ])
]

# normalize: default is L2 norm and axis=1 (= each sample indep)
norm_datas = [normalize(data) for data in datas]

for data, norm_data in zip(datas, norm_datas):

    print()
    for elem, norm_elem in zip(data, norm_data):
        print(f'{elem} {norm_elem}')

    for i in range(len(norm_data)):
        for j in range(len(norm_data) - 1, i, -1):
            sim = 1 - distance.cosine(norm_data[i], norm_data[j])
            print(f'cos({i}, {j}) = {sim:.4f}', end=' ')
        print()
    # XXX distance.cosine(v1, v2) is SLOW
    #     acc to https://stackoverflow.com/questions/18424228

    print()
    for n in [2, 3, 4]:

        print(f'--- components = {n}')

        # method chosen from
        # https://scikit-learn.org/stable/modules/clustering.html
        for model in [
            KMeans(n_clusters=n, random_state=42),
            GaussianMixture(n_components=n, init_params='kmeans', random_state=42)
            # https://towardsdatascience.com/gaussian-mixture-models-vs-k-means-which-one-to-choose-62f2736025f0
        ]:

            # do the clustering
            model.fit(norm_data)

            # get cluster ids
            res = model.predict(norm_data)

            print(f'--- model = {model}')
            print(f'{res}')
            # data[0] és n=2 esetén: [1, 1, 0, 0] -- kiváló! :)

