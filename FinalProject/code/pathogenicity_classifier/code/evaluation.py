import pandas as pd
import numpy as np
import os
from pandas.core.frame import DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, StratifiedShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score 
import argparse
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
import xgboost as xgb
from collections import Counter, defaultdict
import matplotlib.style as style
import matplotlib
from sklearn.svm import SVR
import gower
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.cluster import KMeans
from operator import itemgetter

def dissimilarityMatrix():
    explainers = ["LIME", "TCXP", "ELI5", "SHAP"]

    X_test = pd.read_csv('../output_files/X_test.tsv', sep = '\t', low_memory = False)
    X_train = pd.read_csv('../output_files/X_train.tsv', sep = '\t', low_memory = False)

    n = X_test.shape[0]
    s = X_test.shape[1]

    data = []
    for method in explainers:
        d = pd.read_csv('../output_files/34K_'+method+'_missense_predictions.maf', sep = '\t', low_memory=False)
        d = d.loc[:, d.columns.str.contains('_weight')] #  or d.columns.str.contains('<BIAS>')
        data.append(d)


    dissimilarity = []

    L = gower.gower_matrix(data[0].to_numpy()) 
    dissimilarity.append(L)
    
    T = gower.gower_matrix(data[1].to_numpy())
    dissimilarity.append(T)

    E = gower.gower_matrix(data[2].to_numpy())
    dissimilarity.append(E)

    S = gower.gower_matrix(data[3].to_numpy())
    dissimilarity.append(S)

    range_n_clusters = list (range(2,10))

    index = 0
    m = defaultdict(list)
    for d in dissimilarity:
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=0, n_init=5)
            preds = clusterer.fit_predict(d)
            centers = clusterer.cluster_centers_



            sscore = silhouette_score (d, preds, metric='euclidean')
            dbiscore = davies_bouldin_score(d, clusterer.labels_)
            
            if n_clusters == 2:
                print ("{}: For n_clusters = {}, silhouette score is {}".format(explainers[index], n_clusters, sscore))
                print ("{}: For n_clusters = {}, DBI score is {}".format(explainers[index], n_clusters, dbiscore))

                if explainers[index] == 'ELI5':
                    pass
            m[explainers[index]].append((sscore, dbiscore, n_clusters))

        # print("\n")
        index +=1

    # print("Silhoutte Scores: ", [str(k) +": " +str(max(group, key=itemgetter(0))) for k,group in m.items()])
    # print("DBI Scores: " , [str(k) +": " +str(min(group, key=itemgetter(1))) for k,group in m.items()])
    


def main():
    dissimilarityMatrix()

if __name__ == "__main__":
    main()