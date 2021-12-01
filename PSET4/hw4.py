import datetime, os, pprint, re, sys, time, random, math
from numpy.core.numeric import full
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import linregress, norm
import statsmodels.api as sm 
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, chi2, chisquare

def loadData(genotype_f, file_type = 'numpy'):
    
    """
    :param genotype_f: the file containing all of the genotype values
    :return: lists with the phenotypes and genotype values
    """

    if file_type == 'dataframe':
        return pd.read_csv(genotype_f, sep = '\t')
    else:
        genoList = []
        try:
            with open(genotype_f, 'r') as gFile:
                print("Reading Values")
                for line in gFile:
                    frequencies = [float(freq) for freq in line.strip().split(' ')]
                    genoList.append(frequencies)
        except IOError:
            print("Could not read file: ", genotype_f)
            exit(1)

        return np.array(genoList)

def pValGraphs(p_list, m, letter):
    expected = -1*np.log([i/(m+1) for i in range(1, m+1)])
    observed = -1*np.log(sorted(p_list))
    plt.title('Q1%s: Q-Q Plot of P-values'%letter, fontsize=14)
    plt.plot([min(observed),max(observed)], [min(observed), max(observed)], 'r-')
    plt.scatter(expected, observed, color = 'b')
    plt.ylabel('Observed Quantiles')
    plt.xlabel('Theoretical Quantiles')
    plt.savefig('q1%s.png'%letter)
    plt.clf()

    plt.title('Q1%s: Sorted P-Value plot'%letter, fontsize=14)
    plt.scatter([i for i in range(m)], sorted(p_list), color = 'b')
    plt.xlabel('Index')
    plt.ylabel('Sorted P-Values')
    plt.savefig('q1%s_indexplot.png'%letter)
    plt.clf()

def q1(genotype, phenotype):
    m, n = genotype.shape
    sig_snps = 0
    alpha = 0.05

    p_list = []
    for i in range(m):

        geno = sm.add_constant(genotype[i])
        fit = sm.OLS(phenotype, geno).fit()

        p = fit.pvalues[1]
        p_list.append(p)
        
        if p <= alpha/m:
            sig_snps+=1

    print("Q1a: The number of significant SNPs is: " + str(sig_snps))
    pValGraphs(p_list, m, 'a')


    #create a PCA model object
    pca = PCA(n_components=2, random_state=0)

    #fit the data with the given parameters above and return the transformed data given the components
    projections = pca.fit_transform(np.transpose(genotype))

    fig, ax = plt.subplots()
    
    #create a scatter plot of PC1 and PC2
    plt.scatter(projections[:, 0], projections[:, 1])
    ax.grid('on')
    plt.title('PCA plot', fontsize=14)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('q1b.png')
    plt.clf()

    sig_snps = 0
    p_list = []

    first_pc = np.reshape(projections[:, 0], (-1, 1))

    for i in range(m):
        geno = sm.add_constant(genotype[i])
        geno = np.append(geno, first_pc, 1)

        fit = sm.OLS(phenotype, geno).fit()
        p = fit.pvalues
        p_list.append(p[1])

        if p[1] <= alpha/m:
            sig_snps+=1

    print("Q1c: The number of significant SNPs is: " + str(sig_snps))
    pValGraphs(p_list, m, 'c')

def q2(EUR, AFR, ASN):
    d_hat = np.multiply((EUR - AFR), ASN).sum()/EUR.shape[0]

    print("Q2d: The value of d_hat from these 3 populations is:", d_hat)

    #two tailed test (multiply by 2) because we are testing if the expectation is greater OR less than 0
    print("Q2e: The p_value for H0 is:", norm(loc=0, scale = 4/1000).sf(d_hat)*2)


def main():
    q1_data = np.transpose(loadData('data/Q1/q1.data', 'dataframe').to_numpy())
    q1(q1_data[:-1], q1_data[-1])

    q2_data = loadData('data/Q2/Q2.data')
    q2(q2_data[:, 0], q2_data[:, 1], q2_data[:, 2])




if __name__ == "__main__":
    main()