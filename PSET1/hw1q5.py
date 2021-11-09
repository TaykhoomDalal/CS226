import datetime, os, pprint, re, sys, time, matplotlib, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2, chisquare
from collections import defaultdict

def loadData(phenotype_f, genotype_f):
    
    """
    :param phenotype_f: the file containing all of phenotype values
    :param genotype_f: the file containing all of genotype values
    :return: lists with the phenotypes and genotype values
    """
    phenoList = []
    genoList = []
    try:
        with open(phenotype_f, 'r') as pFile:
            print("Reading Phenotypes")
            for line in pFile:
                phenoList.append(float(line.strip()))
    except IOError:
        print("Could not read file: ", phenotype_f)
        exit(1)

    try:
        with open(genotype_f, 'r') as gFile:
            print("Reading Genotypes")
            for line in gFile:
                ends = [float(x) for x in line.strip().split(' ')]
                genoList.append(ends)
    except IOError:
        print("Could not read file: ", genotype_f)
        exit(1)

    return phenoList, genoList 
    
def testStat(N, Y, G):
    T = N*pow(pearsonr(Y,G)[0], 2)
    return T

def permutationTest(SNP, Pheno, permutations, prin = True):
    T_permuted = []
    T_observed = testStat(250, Pheno, SNP)

    for i in range(0, permutations):
        shuffled = Pheno.copy()
        random.shuffle(shuffled)
        T_permuted.append(testStat(250, shuffled, SNP))
    
    if prin == True:
        plt.title('Histogram of Test Statistic with Permuted Data', fontsize=14)
        plt.hist(T_permuted, bins =20, color = 'red', edgecolor= 'black')
        plt.axvline(T_observed, color='black', linestyle='dashed', linewidth=1)
        plt.ylabel('Frequency')
        plt.xlabel('Distribution of Test Statistic')
        plt.savefig('q5i.png')

        plt.clf()

        p_count = 0
        for i in T_permuted:
            if i >=  T_observed:
                p_count += 1
        print("T1 value: %f" % T_observed)
        print("Number of more extreme test statistics for SNP 1: %d" % p_count)

        print("P-value of T1 based on chi-square approximation: %f" % (1 - chi2.cdf(T_observed , 1)))
    else:
        return 1 - chi2.cdf(T_observed , 1)

def multipleTesting(SNPs, Pheno, permutations):
    p_val_permuted = []

    for i in range(0, permutations):
        shuffled = Pheno.copy()
        random.shuffle(shuffled)
        SNP_ps = []
        for k in range(0, 10):
            SNP_k = [row[k] for row in SNPs]
            T_val = testStat(250, shuffled, SNP_k)
            p_val = 1 - chi2.cdf(T_val , 1)
            SNP_ps.append(p_val)

        p_val_permuted.append(min(SNP_ps))

    p_threshold = np.quantile(p_val_permuted, 0.05) #quantile sorts the data itself in ascending order, so 0.05 is the 95th percentile for descending data
    print("Controlled p-value threshold: %f" % p_threshold)

    for k in range(0, 10):
        T_observed = testStat(250, Pheno, [row[k] for row in SNPs])
        p = 1 - chi2.cdf(T_observed , 1)
        if p <= p_threshold:
            print("Adjusted p-value threshold\tReject the null hypothesis for SNP %d (observed p-value of %f)" % (k+1, p))
        if p <= 0.005:
            print("Bonferrioni Procedure threshold\tReject the null hypothesis for SNP %d (observed p-value of %f)" % (k+1, p))

def main():
    p = 'ps1.phenos'
    g = 'ps1.genos'

    pList, gList = loadData(p,g)

    random.seed(12345)

    permutationTest([row[0] for row in gList], pList, 100000, True)
    multipleTesting(gList, pList, 10000)

if __name__ == "__main__":
    main()