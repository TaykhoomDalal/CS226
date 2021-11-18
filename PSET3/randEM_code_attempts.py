# def hiddenVarProba(pi, f, x, r, m, n, K):

#     numerator = np.zeros(K)

#     for i in range(n):
#         xi = x[i]

#         for k in range(K):
#             numerator[k] = np.log(pi[k])

#             for j in range(m):
#                 numerator[k] *= xi[j]*np.log(f[j][k]) + (1-xi[j])*np.log(1 - f[j][k])
#             if np.exp(numerator[k]) == 0:
#                 numerator[k] = 0.00001
#         r[i, :] = numerator / sum(numerator)
#     return r


# def mixtureComponentAndFValue(r, x, K, n, m):
#     numer = np.array([sum(r[:, i]) for i in range(K)])
#     pi = numer/n
    
#     f = np.empty((m, K))
#     for k in range(K):
#         num = np.zeros(m)
#         for i in range(n):
#             num += r[i, k]*x[i]
#         f[: , k] = num / numer[k]
#     return pi, f

# pi, f = mixtureComponentAndFValue(r, X, K, n, m)
    
# def hiddenVarProba_L(f, x, m, k, i):
#     prod = np.empty(m)

#     for j in range(m):
#         prod[j] = x[i][j]*np.log(f[j][k]) + (1-x[i][j])*np.log(1 - f[j][k])
    
#     return prod


# def completeDataLL(pi, x, f, n,m, K, r):
#     firstTerm = secondTerm = 0

#     for i in range(n):
#         for k in range(K):
#             firstTerm += r[i][k]*np.log(pi[k])
#             secondTerm += r[i][k]*sum(x[i][j]*np.log(f[j][k]) + (1-x[i][j])*np.log(1-f[j][k]) for j in range(m))

#     return firstTerm+secondTerm

# newLL = completeDataLL(pi, X, f, n, m, K, r)


# def hiddenVarProbaAndLL(pi, f, x, r, m, n, K):

#     logLikelihood = 0

#     # loop over all i individuals
#     for i in range(n):
#         d_sum = 0
#         lstar = 0
#         denom = np.zeros(K)

#         # Here we will calculate the value of l_i* 
    
#         # for each cluster
#         for k_ in range(K):
            
#             # calculate the value of l_ik' over all m snps
#             for j in range(m):
#                 denom[k_] += x[i][j]*math.log(f[j][k_]) + (1-x[i][j])*math.log(1 - f[j][k_])

#             # add the log of pi_k' to the value of l_ik'
#             denom[k_] += math.log(pi[k_])    
        
#         # since l_i* = max (over k') of l_ik', we take the max value in the denom array
#         lstar = denom.max()

#         # the full value of the denominator is thus: exp(l_ik' - lstar) summed together (values in denom are l_ik)
#         d_sum = np.exp(denom - lstar).sum()

#         # now for each individual, we have k r values, so we need to calculate each one
#         for k in range(K):

#             # the value of the numerator for r[i][k] is the same as the l_ik calculated for the denominator
#             s = np.exp(denom[k] - lstar)

#             #divide the numerator by the sum over all k
#             r[i][k] = s/d_sum
        
#             # also we can calculate the loglikelihood using the r[i][k] we just found
#             logLikelihood += r[i][k]*math.log(pi[k]) + r[i][k]*sum(x[i][j]*math.log(f[j][k]) + (1-x[i][j])*math.log(1-f[j][k]) for j in range(m))


#     return r, logLikelihood