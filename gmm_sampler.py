import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multinomial, dirichlet, lognorm, bernoulli

#x : 1000 x 2
#z : 1000 x 1
#pi : 1 x 3
#mu : 3 x 2
#lamb : 3 x 1
#v : 3 x 2
#cov : 3 x 2 x 2


def compute_cov(lamb, v):
    lamb_I = np.expand_dims(lamb, 1) * np.expand_dims(np.identity(2), 0)
    vvT = np.expand_dims(v, 1) * np.expand_dims(v, 2) 
    cov = lamb_I + vvT
    return cov

def pi_times_normal_pdf(x, pi, mu, cov):
    for k in range(3):
        tmp = pi[:,k] * multivariate_normal.pdf(x, mu[k], cov[k])
        if k == 0:
            numerator = np.expand_dims(tmp, 1)
        else:
            numerator = np.concatenate((numerator, np.expand_dims(tmp, 1)), axis=1)
    return numerator

def gibbs_z_sample(numerator):
    denominator = np.log(np.prod(np.exp(numerator), 1, keepdims=True))
    prob = numerator / denominator#np.sum(numerator, 1, keepdims=True)
    for i in range(1000):
        try:
            zi = multinomial.rvs(1, prob[i])
        except ValueError:
            # concentrated probability on certain index
            zi = np.zeros(3)
            index = np.argmax(prob[i])
            zi[index] = 1
        if i == 0:
            z = np.expand_dims(zi, 0)
        else:
            z = np.concatenate((z, np.expand_dims(zi, 0)), axis=0)
    return z

def gibbs_pi_sample(z):
    count = np.sum(z, 0)
    pi = dirichlet.rvs(1+count)
    return pi

def mcmc_sample(x, pi, mu, lamb, v, sigma_q):
    mu_prime, lamb_prime, v_prime = mu.copy(), lamb.copy(), v.copy()
    for k in range(3):
        mu_prime[k] = multivariate_normal.rvs(mu[k], sigma_q**2 * np.identity(2), size=(1,1))
        lamb_prime[k] = lognorm.rvs(sigma_q, scale=lamb[k], size=(1,1))
        v_prime[k] = multivariate_normal.rvs(v[k], sigma_q**2 * np.identity(2), size=(1,1))
        
        cov = compute_cov(lamb, v)
        cov_prime = compute_cov(lamb_prime, v_prime)
    
        numerator = pi_times_normal_pdf(x, pi, mu, cov)
        numerator_prime = pi_times_normal_pdf(x, pi, mu_prime, cov_prime)

        first = np.prod(np.sum(numerator_prime, 1) / np.sum(numerator, 1), 0)
        second_1 = - 0.1 * (np.dot(mu_prime[k], mu_prime[k]) - np.dot(mu[k], mu[k]))
        second_2 = - 50 * ((np.log(lamb_prime[k]) - 0.1) ** 2 - (np.log(lamb[k]) - 0.1) ** 2)
        second_3 = - 2 * (np.dot(v_prime[k], v_prime[k]) - np.dot(v[k], v[k]))
        second = np.exp(second_1 + second_2 + second_3)
        
        accept_prob = min([1, first * second])
        binary = bernoulli.rvs(accept_prob)
        #print(binary)
        
        if binary == 1:
            pass
        else:
            mu_prime[k] = mu[k].copy()
            lamb_prime[k] = lamb[k].copy()
            v_prime[k] = v[k].copy()
        mu = mu_prime.copy()
        lamb = lamb_prime.copy()
        v = v_prime.copy()
    return mu, lamb, v

def compute_log_likelihood(x, pi, mu, lamb, v):
    cov = compute_cov(lamb, v)
    numerator = pi_times_normal_pdf(x, pi, mu, cov)
    
    log_likelihood = np.sum(np.log(np.sum(numerator, 1)), 0)
    return log_likelihood
  
if __name__ == "__main__":
    x = np.loadtxt('X.txt')
    
    sigma_q = 0.5
    
    pi = dirichlet.rvs(np.ones(3))
    mu = multivariate_normal.rvs(np.zeros(2), 5 * np.identity(2), size=(3,1))
    lamb = lognorm.rvs(0.1, scale=np.exp(0.1), size=(3,1))
    v = multivariate_normal.rvs(np.zeros(2), 0.25 * np.identity(2), size=(3,1))
    
    log_likelihood_list = []
    cnt = 0
    while True:
        cnt += 1
        
        cov = compute_cov(lamb, v)
        numerator = pi_times_normal_pdf(x, pi, mu, cov)
        z = gibbs_z_sample(numerator)
        pi = gibbs_pi_sample(z)
        mu, lamb, v = mcmc_sample(x, pi, mu, lamb, v, sigma_q)
        log_likelihood = compute_log_likelihood(x, pi, mu, lamb, v)
        log_likelihood_list.append(log_likelihood)
        
        if cnt % 50 == 0:
            print(cnt)
            estimated_param = {'z':z, 'pi':pi, 'mu':mu, 'lamb':lamb, 'v':v}
            with open('estimated_param.pickle', 'wb') as f:
                pickle.dump(estimated_param, f)
                f.close()

            with open('log_likelihood.pickle', 'wb') as f:
                pickle.dump(log_likelihood_list, f)
                f.close()
                
            plt.plot(log_likelihood_list)
            plt.savefig('trace_plot.png')
            plt.close()
            
            for k in range(3):
                plt.scatter(x[z[:,k] == 1][:,0], x[z[:,k] == 1][:,1], color=['r', 'g', 'b'][k])
            plt.savefig('clustered' + str(cnt) + '.png')
            plt.close()
            
        