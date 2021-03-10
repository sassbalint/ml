"""
Learning scikit-learn clustering techniques.
"""

import argparse
import sys

import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def handle_nullvector(data):
    """Null vector is not allowed as cos-distance is not defined for it."""
    # XXX ócska hekk, ami a nu_vectors.csv -re épp működik
    return data + 1


def main():
    """Do the thing."""
    args = get_args()

    np.set_printoptions(precision=2)

    # XXX kategoriális adat -> OneHotEncoder()-rel kellene csinálni!
    with open(args.file, encoding='utf-8') as inputfile:
        entities = []
        vectors = []
        for line in inputfile:
            if line[0] == '#': continue
            entity, *vector = line.strip().split('\t')
            entities.append(entity)
            vectors.append([int(v) for v in vector])
    data = np.array(vectors)

    # cos distance used -> null vectors are not allowed
    if args.handle_nullvector:
        data = handle_nullvector(data)

    # normalize: default is L2 norm (and axis=1 = each sample indep)
    norm_data = normalize(data) if args.normalize else data

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
    for n in range(2, min(args.max_clusters, len(norm_data)) + 1):

        print()
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

            # get cluster identifiers
            res = model.predict(norm_data)

            print(f'--- model = {model}')
            print(f'--- result = {res}')

            for x, y, wl in zip(data, res, entities):
                print(f'{"-".join(str(xv) for xv in x)}\t{y}\t{wl}')


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--file',
        help='input csv file: entity + vector',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-m', '--max-clusters',
        help='maximum number of clusters',
        type=int,
        default=8
    )
    parser.add_argument(
        '-n', '--normalize',
        help='normalize vectors using L2 norm',
        action='store_true'
    )
    parser.add_argument(
        '-0', '--handle-nullvector',
        help='modify data not to have null vectors',
        action='store_true'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    main()
