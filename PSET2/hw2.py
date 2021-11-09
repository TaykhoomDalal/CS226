import datetime, os, pprint, re, sys, time, matplotlib, random, math
import pandas as pd
import numpy as np
import seaborn as sns
import operator
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress, norm
import statsmodels.api as sm 

def sigmoid(x):
  return 1.0/ (1.0 + np.exp(-x))

def loadData(genotype_f, phenotype_f):
    
    """
    :param genotype_f: the file containing all of the genotype values
    :param phenotype_f: the file containing all of the phenotype values
    :return: lists with the phenotypes and genotype values
    """
    phenoList = []
    genoList = []
    try:
        with open(phenotype_f, 'r') as pFile:
            print("Reading Phenotypes")
            for line in pFile:
                phenos = [float(x) for x in line.strip().split(' ')]
                if len(phenos) == 1:
                    phenos = phenos[0]
                phenoList.append(phenos)
    except IOError:
        print("Could not read file: ", phenotype_f)
        exit(1)

    try:
        with open(genotype_f, 'r') as gFile:
            print("Reading Genotypes")
            for line in gFile:
                genos = [int(x) for x in line.strip()]
                genoList.append(genos)
    except IOError:
        print("Could not read file: ", genotype_f)
        exit(1)

    return np.transpose(np.array(genoList)), np.array(phenoList)

def closedFormRidge(X,y, regularizer, Xt, yt):

    _, m = X.shape

    firstTerm = (X.T @ X) + regularizer*np.identity(m)
    secondTerm = X.T @ y
    train_betas = np.linalg.solve(firstTerm, secondTerm)

    y_train_preds = X @ train_betas
    y_test_preds = Xt @ train_betas

    return mean_squared_error(y, y_train_preds), mean_squared_error(yt, y_test_preds)

def ridgeRegression(X, y, regularizer, Xt, yt):
    ridge_clf = Ridge(fit_intercept = False, alpha = regularizer, random_state = 42)
    ridge_clf.fit(X, y)
    y_train_preds = ridge_clf.predict(X)
    y_test_preds = ridge_clf.predict(Xt)

    return mean_squared_error(y, y_train_preds), mean_squared_error(yt, y_test_preds)

def q1(X1_train, y1_train, X1_test, y1_test):
    train_mses = []
    test_mses = []
    lambdas = [0, 2, 5, 8]

    X_rows = X1_train.shape[0]
    Xt_rows = X1_test.shape[0]
    
    # add the row of ones corresponding to the intercept
    X1_train_ex = np.append(X1_train, np.ones((X_rows,1)), axis =1)
    X1_test_ex = np.append(X1_test, np.ones((Xt_rows,1)), axis =1)

    for i in lambdas:
        train, test = closedFormRidge(X1_train, y1_train, i, X1_test, y1_test)
        
        # if i == 0:
        #     print(ridgeRegression(X1_train_ex, y1_train, i, X1_test_ex, y1_test))
        #     print(train, test)
        
        train_mses.append(train)
        test_mses.append(test)

    plt.title('Q1: Training Data', fontsize=14)
    plt.plot(lambdas, train_mses,'ro-')
    index = 0
    for i in train_mses:
        plt.annotate(str(round(i, 3)), xy=(lambdas[index], i), xytext=(-5,9), textcoords='offset points')
        index += 1
    plt.ylabel('MSE')
    plt.ylim(0,2)
    plt.xlabel('Lambda Value')
    plt.savefig('q1c_train.png')
    plt.clf()

    plt.title('Q1: Test Data', fontsize=14)
    plt.plot(lambdas, test_mses,'ro-')
    index = 0
    for i in test_mses:
        if index == 0:
            xytext = (6, 0)
        elif index == 3:
            xytext = (-8,5) 
        else:
            xytext = (4, 5)
        plt.annotate(str(round(i)), xy=(lambdas[index], i), xytext=xytext, textcoords='offset points')
        index += 1
    plt.ylabel('MSE')
    plt.xlabel('Lambda Value')
    plt.savefig('q1c_test.png')
    plt.clf()

def NLL(X, y, B):
    n = X.shape[0]
    summation = 0

    for i in range(n):
        summation += y[i]*np.log(sigmoid(B.T @ X[i])) + (1-y[i])*np.log(1 - sigmoid(B.T @ X[i]))
        
    return -1*summation

def betaDeriv(X, y, B):
    n, m = X.shape
    summation = np.zeros(m)

    for i in range(n):
        summation += sigmoid(B @ X[i]) * X[i] - y[i]*X[i]

    return summation

def gradientDescent(stepSize, X, y, iterations = 100):
    n, m = X.shape
    prevBeta = np.zeros(m)
    nll_list = []

    for i in range(iterations):
        newBeta = prevBeta - stepSize*betaDeriv(X, y, prevBeta)
        if i < 50:
            nll_list.append(NLL(X,y, newBeta))
        prevBeta = newBeta

    nll = np.array(nll_list)
    x = [i for i in range(1,51)]

    #indexes where nll has nan value
    indexes = np.argwhere(np.isnan(nll)).flatten()

    nll = nll[~np.isnan(nll)]

    #remove these indexes from x
    x = np.delete(x, indexes)

    return nll, x

def logRegHessian(X,y, B):
    n, m = X.shape
    summation = np.zeros((n,n))

    diag = [sigmoid(B @ X[i])*(1 - sigmoid(B @ X[i])) for i in range(n)]
    np.fill_diagonal(summation, diag)

    return X.T @ summation @ X

def newtonsMethod(X, y, iterations = 100):
    n, m = X.shape
    prevBeta = np.zeros(m)
    nll_list = []

    for i in range(iterations):
        newBeta = prevBeta - np.linalg.inv(logRegHessian(X, y, prevBeta)) @ betaDeriv(X, y, prevBeta)
        if i < 50:
            nll_list.append(NLL(X,y, newBeta))
        prevBeta = newBeta

    print("Beta estimate for SNP 1: %f\nBeta estimate for SNP 7: %f" % (prevBeta[0], prevBeta[6]))

    nll = np.array(nll_list)
    x = [i for i in range(1,51)]

    #indexes where nll has nan value
    indexes = np.argwhere(np.isnan(nll)).flatten()

    nll = nll[~np.isnan(nll)]

    #remove these indexes from x
    x = np.delete(x, indexes)

    return nll, x

def q2(X, y):
    
    # add the row of ones corresponding to the intercept
    n = X.shape[0]
    X_ones = np.append(X, np.ones((n,1)), axis =1)

    plt.title('Q2d: Gradient Descent and Newton\'s Method', fontsize=14)
    plt.ylabel('Negative Log Likelihood')
    plt.xlabel('Iteration')
    

    colors = ['r', 'g', 'b', 'c', 'm']

    for i in range(5):
        nll, x = gradientDescent(10**(-1*(i+1)), X_ones, y)
        plt.plot(x, nll,colors[i]+'o-', label='GD: stepsize of %f'%10**(-1*(i+1)), linewidth=0.5,markersize=3)
    
    nll, x = newtonsMethod(X_ones, y)
    plt.plot(x, nll,'ko-', label='Newton\'s Method', linewidth=0.5,markersize=3)
    
    plt.legend()
    plt.yscale('log') 
    plt.ylim(bottom=1)
    plt.savefig('q2d.png')
    plt.clf()

def linRegression(X, y):
    slope, intercept, r, p, se = linregress(X, y)
    return p, y - slope*X + intercept

def q3(X, y):
    for i in range(4):
        p_list = []
        for j in range(382):
            # if j == 118 and i == 2:
            #     newX = np.delete(X[:, j], np.argmax(y[:, i]))
            #     newY = np.delete(y[:, i], np.argmax(y[:, i]))
            #     p_val, resids = linRegression(newX, newY)
            #     print(p_val, p_val < 0.05/(382*4))
                
            # elif j == 42 and i == 3:
            #     newX = X[:, j]
            #     newY = y[:, i]
            #     for wwrg in range(3):
            #         print(np.argmin(newY), newY[np.argmin(newY)])
            #         newX = np.delete(newX, np.argmin(newY))
            #         newY = np.delete(newY, np.argmin(newY))

            #     p_val, resids = linRegression(newX, newY)
            #     print(p_val, p_val < 0.05/(382*4))
            p_val, resids = linRegression(X[:, j], y[:, i])
            p_list.append(p_val)
            if p_val < 0.05/(382):
                
                theoretical_points = [(_-0.5)/len(resids) for _ in range(1, 1+len(resids))]
                empirical_cdf = sorted(resids)
                theoretical_cdf = [norm.ppf(_) for _ in theoretical_points]
                
                # normalize the data to 0-1 range
                newt = (theoretical_cdf - min(theoretical_cdf))/(max(theoretical_cdf) - min(theoretical_cdf))
                newe = (empirical_cdf - min(empirical_cdf))/(max(empirical_cdf) - min(empirical_cdf))
                
                plt.title('Q-Q Plot of Residuals for SNP %d and Phenotype %d' % (j+1, i+1), fontsize=14)
                plt.plot([min(newe),max(newe)], [min(newe), max(newe)], 'r-')
                plt.scatter(newt[1:498], newe[1:498], color = 'b')
                plt.ylabel('Observed Quantiles')
                plt.xlabel('Theoretical Quantiles')
                plt.savefig('q3d%d_resids.png'%(i+1))
                plt.clf()

                # sm.qqplot(resids,fit = "true")
                # plt.plot([min(resids),max(resids)], [min(resids), max(resids)], 'b-')
                # plt.savefig('q3d_pheno%d_resids.png'%(i+1))
                # plt.clf()
                
                plt.title('Boxplot for SNP %d and Phenotype %d' % (j+1, i+1), fontsize=14)
                sns.boxplot(x=X[:, j], y=y[:, i])
                plt.ylabel('Phenotype Value')
                plt.xlabel('P-values')
                plt.savefig('q3d_pheno%d.png'%(i+1))

                plt.clf()
                print("Phenotype %d: SNP %d p-value = %.9f" % (i+1, j+1, p_val))

        ax = plt.gca()
        plt.title('Histogram of P-Values for Phenotype %d' % (i+1), fontsize=14)
        plt.hist(p_list, bins =10, edgecolor= 'black')
        plt.ylabel('Frequency')
        plt.xlabel('P-values')
        plt.savefig('q3c_pheno%d.png'%(i+1))

        plt.clf()



def main():
    # X1_train, y1_train = loadData("Q1/Q1.training.geno", "Q1/Q1.training.pheno")
    # X1_test, y1_test = loadData("Q1/Q1.test.geno", "Q1/Q1.test.pheno")
    # q1(X1_train, y1_train, X1_test, y1_test)
    
    # X2, y2 = loadData("Q2/hw.2-1.geno", "Q2/hw.2-1.pheno")
    # q2(X2, y2)

    X3, y3 = loadData("Q3/hw.2-2.geno", "Q3/hw.2-2.pheno")
    q3(X3, y3)





if __name__ =="__main__":
    main()