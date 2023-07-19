import numpy as np
from scipy.stats import triang, lognorm, pareto

def Main(a, b, c, point1, point2, data, mu, sigma, xm, alpha, num, point3, point4, point5):
    #1
    # Calculates the triangular distribution 
    dist = triang(c=(c-a)/(b-a), loc=a, scale=b-a)
    #Finds the probability 𝐩𝐫𝐨𝐛𝟏 that the AV is no greater than 𝐩𝐨𝐢𝐧𝐭1
    prob1 = dist.cdf(point1)
    #Find the probability 𝐩𝐫𝐨𝐛𝟐 that the AV is greater than 𝐩𝐨𝐢𝐧𝐭𝟐 
    prob2 = 1 - dist.cdf(point2)
    # Calculates the mean of the triangular distribution
    MEAN_t = dist.mean()
    # Calculates the median of the triangular distribution
    MEDIAN_t = dist.median()
    
    #2
    #Calculates mean of the dataset using np.mean
    MEAN_d = np.mean(data)
    #Calculates variance of the dataset using np.var
    VARIANCE_d = np.var(data)
    
    # 3
    #Uses the Monte Carlo method using 500000 random samples to calcualte Impact A and B using log distribution
    impact_A = lognorm(sigma, scale=np.exp(mu)).rvs(num)
    impact_B = pareto(alpha, loc=xm).rvs(num)
    #Calcualtes total impact by using addition on both Impact A and B
    total_impact = impact_A + impact_B
    #uses np.mean to calculate the probability 𝐩𝐫𝐨𝐛𝟑 that the total impact is greater than 𝐩𝐨𝐢𝐧𝐭3
    prob3 = np.mean(total_impact > point3)
    #uses np.mean to calculate the probability 𝐩𝐫𝐨𝐛𝟒 that the total impact is between 𝐩𝐨𝐢𝐧𝐭𝟒 and 𝐩𝐨𝐢𝐧𝐭𝟓 
    prob4 = np.mean((total_impact > point4) & (total_impact < point5))
    
    # 4 Calculates the ALE using Mean_d * Median_t * prob 3
    EF = prob3
    SLE = MEDIAN_t * EF
    ARO = MEAN_d
    ALE = ARO * SLE 
    
    return (prob1, prob2, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d, prob3, prob4, ALE)


a, b, c, point1, point2, mu, sigma, xm, alpha, num, point3, point4, point5 = 10000, 35000, 18000, 12000, 25000, 0, 3, 1, 4, 500000, 30, 50, 100
data = [11, 15, 9, 5, 3, 14, 16, 15, 12, 10, 11, 4, 7, 12, 6]

results = Main(a, b, c, point1, point2, data, mu, sigma, xm, alpha, num, point3, point4, point5)

print("prob1 =", results[0])
print("prob2 =", results[1])
print("MEAN_t =", results[2])
print("MEDIAN_t =", results[3])
print("MEAN_d =", results[4])
print("VARIANCE_d =", results[5])
print("prob3 =", results[6])
print("prob4 =", results[7])
print("ALE =", results[8])
