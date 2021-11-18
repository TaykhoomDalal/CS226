import datetime, os, pprint, re, sys, time, random, math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

def loadData(genotype_f):
    
    """
    :param genotype_f: the file containing all of the genotype values
    :param phenotype_f: the file containing all of the phenotype values
    :return: lists with the phenotypes and genotype values
    """
    genoList = []
    try:
        with open(genotype_f, 'r') as gFile:
            print("Reading Values")
            for line in gFile:
                genos = [int(x) for x in line.strip() if x!= ' ']
                genoList.append(genos)
    except IOError:
        print("Could not read file: ", genotype_f)
        exit(1)

    return np.array(genoList)

def mixtureComponent(r, K, n, pi):

    # we have a matrix of K values

    #loop over each cluster
    # each mixture components is the sum over all individuals of r[i]][k] divided by total number of individuals
    pi = r.sum(axis=0)/n

    # return the normalized pi values
    return pi/pi.sum()
        
def bernoulliFValue(r, K, x, n, m, f):

    #r = n x K matrix
    #x = n x M matrix

    # we have a matrix of m x k values of the bernoulli parameter

    #first we loop over number of snps (rows)
    for j in range(m):
        
        #loop over number of cluster (columns)
        for k in range(K):

            #numerator of f[j][k] is the sum over all individuals of r[i][k]*x[i][j]
            num = (r[:, k]*x[:, j]).sum()

            #denominator of f[j][k] is sum over all individuals of r[i][k]
            denom = r[:, k].sum()
        
            # we divide the two values
            f_val = num/denom

            # if f == 0, then we will run into issues when taking the log of this
            if f_val == 0:
                f[j][k] = 0.00001
            # if numerator and denominator == each other, we will run into issues when taking log of 1 - this
            elif f_val == 1:
                f[j][k] = 0.99999
            else:
                f[j][k] = f_val
            
    return f

def hiddenVarProbaAndLL(pi, f, x, r, m, n, K):

    logLikelihood = 0

    lf = np.log(f)
    lf_minus = np.log(1-f)
    lpi = np.log(pi)

    # x = n x m matrix
    # f = m x K
    # arr1 = n x K
    arr1 = (x @ lf) + ((1-x) @ lf_minus)
    
    # loop over all i individuals
    for i in range(n):
        d_sum = 0
        lstar = 0
        denom = np.zeros(K)

        # Here we will calculate the value of l_i* 
        # for each cluster calculate the value of l_ik' over all m snps
        # and then add the log of pi_k' to the value of l_ik'
        
        denom = arr1[i] + lpi
        # since l_i* = max (over k') of l_ik', we take the max value in the denom array
        lstar = denom.max()

        # the full value of the denominator is thus: exp(l_ik' - lstar) summed together (values in denom are l_ik)
        # the value of the numerator for r[i][k] is the same as the l_ik calculated for the denominator
        numerators = np.exp(denom - lstar)
        d_sum = numerators.sum()
        
        # now for each individual, we have k r values, so we need to calculate each one
        # divide the numerator by the sum over all k
        r[i] = numerators/d_sum
        
        # also we can calculate the loglikelihood using the r[i][k] we just found
        logLikelihood += (r[i]*lpi).sum() + (r[i]*arr1[i]).sum()


    return r, logLikelihood

def emAlgorithm(X, n, m, K, plot2a = False):
    diff = 1
    iterations = list(range(100))
    ll_list = []
    oldLL = 1

    # initialize values of r, f, and pi
    r = np.zeros((n, K))
    f = np.random.uniform(size = (m, K))
    pi = np.random.exponential(size=(K))

    #normalize pi
    pi = pi/pi.sum()

    # loop 100 times
    for i in iterations:
        # E-step: calculate the soft assignment probabilities and the value of the likelihood
        r, newLL = hiddenVarProbaAndLL(pi, f, X, r, m, n, K)
        ll_list.append(newLL)
        
        print(newLL)

        #if the absolute value of the difference between the old and new likelihood is small enough, break
        diff = abs(newLL - oldLL)
        if diff < 10**(-8) and i != 0:
            print(diff, newLL, oldLL)
            break
        
        # M-step, using the estimate of the soft assignments, calculate the max of the other parameters
        pi = mixtureComponent(r, K, n, pi)
        f = bernoulliFValue(r, K, X, n, m, f)

        print(i, pi)
        oldLL = newLL

    if plot2a:
        fig, ax = plt.subplots(figsize=(9,6))
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.title('Q2a: Log Likelihood for single run of EM', fontsize=14)
        ax.plot(iterations[:len(ll_list)], np.array(ll_list),'ro-')
        plt.ylabel('Log Likelihood')
        plt.xlabel('Iteration')
        plt.savefig('q2a.png')
        plt.clf()
        print(ll_list[-1])

    # swap labels (for 2 cluster case) to go along with true values
    if K > 1 and pi[0] > pi[1]:
                r[:, [0, 1]] = r[:, [1, 0]]

    # set the probabilities to their respective hard assignments
    r[r < 0.5] = 0
    r[r > 0.5] = 1

    return (ll_list[-1], r, pi)

def q2(X, K, iters =3, plot2a = False, plot2e = False, plot2g = False,  Z = None, ax2e = None, fig2e = None):
    n,m = X.shape
    print(n, m)
    results = []

    for k in K:
        for i in range(iters):
            results.append(emAlgorithm(X, n, m, k, plot2a and k==2 and i ==0))

    likelihood = [v[0] for v in results]

    if Z is not None and not plot2e:
        accuracy = np.array([(v[1] == Z).sum()/(v[1] == Z).size for v in results])
        print(accuracy)

    # if Z is not None:
    #     return

    if plot2g:
        fig, ax = plt.subplots(figsize=(9,6))
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.title('Q2g: Log Likelihood across K values', fontsize=14)
        ax.plot(K, likelihood,'bo-')
        plt.ylabel('Log Likelihood')
        plt.xlabel('K')
        plt.savefig('q2g.png')
        plt.clf()
    
    if plot2e:
        if m == 1000:
            print('')
        accuracy = np.array([(v[1] == Z).sum()/(v[1] == Z).size for v in results])
        return accuracy

def q3(X):

    #create a PCA model object
    pca = PCA(n_components=2, random_state=0)

    #fit the data with the given parameters above and return the transformed data given the components
    projections = pca.fit_transform(X.iloc[:, 2:].to_numpy())

    #get a list of the labels
    labels = X.iloc[:, 1]

    #make a dict with colors associated with the population
    colors = dict({'EUR':'red', 'ASN': 'blue', 'AFR':'green', 'AMR': 'orange'})
    # loop over the population : # individuals pairs and print them out in order of greatest to fewest individuals
    print('True number of individuals per population')
    for p in Counter(labels).most_common():
        print('%s: %d'% (p[0], p[1]))

    #create a scatter plot of each population and label it with the color and name of the population
    fig, ax = plt.subplots(figsize=(10,10))
    for label in np.unique(labels):
        plt.scatter(projections[labels == label, 0], projections[labels == label, 1], label = label, c = colors[label])
    ax.grid('on')
    plt.legend()
    plt.title('PCA plot', fontsize=14)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('q3a.png')
    plt.clf()

    #create a KMeans model object
    kmeans = KMeans(n_clusters = 4, n_init = 5, random_state=0)

    #fit the data with the given model parameters, making sure to only pass the SNPS to the clustering
    # this function then returns a list of the labels assigned to each individual in order
    assignments = kmeans.fit_predict(X.iloc[:, 2:].to_numpy())

    # get a count of the frequency per each label
    cluster_sizes = Counter(assignments)

    #then create a dictionary with the cluster label being the key, and the value 
    # being a list of indices for which individuals are in which cluster
    indices = defaultdict(list)
    for index, cluster in enumerate(assignments):
        indices[cluster].append(index)

    # loop over the frequency of each cluster in order of which cluster has most individuals from KMeans
    print("\nKMeans predicted number of individuals per population")
    for i in cluster_sizes.most_common():
        # the first part in this complicated print statement gets the column of values in the dataframe
        # with population names, then we grab an index from our dictionary, and index into the dataframe
        # and then we return the population associated with that individual (we choose the second individual because some cluster assignments were wrong)
        # we then print the number of individuals in that cluster
        population = X.iloc[:, 1].loc[[indices[i[0]][1]]].to_string(index=False, header = False)
        
        #change the key name from the population to the cluster label
        colors[i[0]] = colors.pop(population)
        print(population,': ', i[1])

    
    # create a scatter plot of each cluster using the values from the PCA, by using the indices to index into the PCA projections,
    # and then labeling that cluster with the corresponding number, where Cluster1 has most individuals, and Cluster2 has least
    fig, ax = plt.subplots(figsize=(10,10))
    count = 1
    for cluster in cluster_sizes.most_common():
        #color = colors[cluster[0]] (to set color according to cluster label that is inline with previous plot)
        plt.scatter(projections[indices[cluster[0]], 0], projections[indices[cluster[0]], 1], label = 'Cluster%d'%(count)) 
        count +=1
    ax.grid('on')
    plt.legend()
    plt.title('KMeans PCA plot', fontsize=14)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('q3c.png')
    plt.clf()





def main():
    # q2_data1_geno = np.transpose(loadData("ps3/data/Q2/mixture1.geno"))
    # q2_data1_hidden = loadData("ps3/data/Q2/mixture1.ganc")
    
    # # q2(q2_data1_geno,[2], 3, True, False, False, q2_data1_hidden)

    # SNPS = [10, 100, 1000, 5000]
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # accuracy = []
    # for num_snps in SNPS:
    #     accuracy.append(q2(q2_data1_geno[:, :num_snps],[2], 1, False, True, False, q2_data1_hidden))
    
    # ax.ticklabel_format(useOffset=False, style='plain')
    # ax.plot(SNPS, accuracy,'go-')
    # plt.title('Q2e: Change in accuracy of predicting population label 1', fontsize=14)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of SNPS')
    # plt.savefig('q2e.png')
    # plt.clf()

    # q2_data2_geno = np.transpose(loadData("ps3/data/Q2/mixture2.geno"))
    # q2(q2_data2_geno,[1, 2, 3, 4], 1, False, False, True)

    q3_data = pd.read_csv("ps3/data/Q3/q3.data", sep = '\t')
    q3(q3_data)


if __name__ == "__main__":
    main()