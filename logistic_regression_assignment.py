import numpy as np
import numpy.random as rn
from scipy import optimize, stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt



# ##############################################################################
# load_data generates a binary dataset for visualisation and testing using two
# parameters:
# * A **jitter** parameter that controls how noisy the data are; and
# * An **offset** parameter that controls the separation between the two classes.
#
# Do not change this function!
# ##############################################################################
def load_data(N=50, jitter=0.7, offset=1.2):
    # Generate the data
    x = np.vstack([rn.normal(0, jitter, (N // 2, 1)),
                   rn.normal(offset, jitter, (N // 2, 1))])
    y = np.vstack([np.zeros((N // 2, 1)), np.ones((N // 2, 1))])
    x_test = np.linspace(-2, offset + 2).reshape(-1, 1)

    # Make the augmented data matrix by adding a column of ones
    x_train = np.hstack([np.ones((N, 1)), x])
    x_test = np.hstack([np.ones((N, 1)), x_test])
    return x_train, y, x_test


# ##############################################################################
# predict takes a input matrix X and parameters of the logistic regression theta
# and predicts the output of the logistic regression.
# ##############################################################################
def predict(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: prediction of f(X); K x 1 vector
    prediction = np.zeros((X.shape[0], 1))

    # Task 1:
    # TODO: Implement the prediction of a logistic regression here.
    prediction = 1/(1+np.exp(-np.dot(X, theta)))

    return prediction


def predict_binary(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: binary prediction of f(X); K x 1 vector; should be 0 or 1

    prediction = 1. * (predict(X, theta) > 0.5)

    return prediction


# ##############################################################################
# log_likelihood takes data matrices x and y and parameters of the logistic
# regression theta and returns the log likelihood of the data given the logistic
# regression.
# ##############################################################################
def log_likelihood(X, y, theta):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # theta: parameters (D x 1)
    # returns: log likelihood, scalar

    L = 0

    # Task 2:
    # TODO: Calculate the log-likelihood of a dataset
    # given a value of theta.
    N = len(y)
    one_vec = np.ones((N,1))
    mu = predict(X, theta)
    L = y.T.dot(np.log(mu)) + (one_vec - y).T.dot(np.log(one_vec - mu))
    return L.item()


# ##############################################################################
# max_lik_estimate takes data matrices x and y ands return the maximum
# likelihood parameters of a logistic regression.
# ##############################################################################
def max_lik_estimate(X, y):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # returns: maximum likelihood parameters (D x 1)

    N, D = X.shape

    theta_init = rn.rand(D, 1)
    theta_ml = theta_init

    # Task 3:
    # TODO: Optimize the log-likelihood function you've
    # written above an obtain a maximum likelihood estimate

    ####Using scipy.minimize########
    def f(theta):
        theta = theta.reshape(D,1)
        return -log_likelihood(X, y, theta)
    # def gradient(theta):
    #     return ((predict(X,theta)-y).T.dot(X)).T.flatten()

    min = optimize.minimize(fun = f, x0=theta_init, method='BFGS')
    theta_ml = min.x

    return theta_ml


# ##############################################################################
# neg_log_posterior takes data matrices x and y and parameters of the logistic
# regression theta as well as a prior mean m and covariance S and returns the
# negative log posterior of the data given the logistic regression.
# ##############################################################################
def neg_log_posterior(theta, X, y, m, S):
    # theta: D x 1 matrix of parameters
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: scalar negative log posterior

    negative_log_posterior = 0

    # Task 4:
    # TODO: Calculate the log-posterior
    neg_log_prior = 0.5*(np.log(np.linalg.det(S))+(theta-m).T.dot(np.linalg.inv(S)).dot(theta-m))
    negative_log_posterior = neg_log_prior - log_likelihood(X, y, theta)

    return negative_log_posterior.item()


# ##############################################################################
# map_estimate takes data matrices x and y as well as a prior mean m and
# covariance  and returns the maximum a posteriori parameters of a logistic
# regression.
# ##############################################################################
def map_estimate(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)

    N, D = X.shape

    theta_init = rn.rand(D, 1)
    theta_map = theta_init

    # Task 5:
    # TODO: Optimize the log-posterior function you've
    # written above an obtain a maximum a posteriori estimate

    def f(theta):
        theta = theta.reshape(D,1)
        return neg_log_posterior(theta, X, y, m, S)

    min = optimize.minimize(fun = f,x0 = theta_init,method='BFGS')
    theta_map = min.x
    return theta_map


# ##############################################################################
# laplace_q takes an array of points z and returns an array with Laplace
# approximation q evaluated at all points in z.
# ##############################################################################
def laplace_q(z):
    # z: double array of size (T,)
    # returns: array with Laplace approximation q evaluated
    #          at all points in z

    q = np.zeros_like(z)

    # Task 6:
    # TODO: Evaluate the Laplace approximation $q(z)$.
    T = len(z)
    mean = 2
    variance = 4
    for i in range(T):
        q[i] = 1/np.sqrt(2*np.pi*variance) * np.exp(-0.5*(z[i]-mean)**2/variance)

    return q


# ##############################################################################
# get_posterior takes data matrices x and y as well as a prior mean m and
# covariance and returns the maximum a posteriori solution to parameters
# of a logistic regression as well as the covariance approximated with the
# Laplace approximation.
# ##############################################################################
def get_posterior(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)
    #          covariance of Laplace approximation (D x D)

    mu_post = np.zeros_like(m)
    S_post = np.zeros_like(S)

    # Task 7:
    # TODO: Calculate the Laplace approximation of p(theta | X, y)

    D = len(X[0])
    N = len(X)
    mu_post = map_estimate(X, y, m, S).reshape(-1,1)
    sigm = predict(X, mu_post)
    one_vec = np.ones_like(y)
    H_NNL = X.T.dot(sigm*(one_vec-sigm)*X)
    H_prior = np.linalg.inv(S).T
    S_post = np.linalg.inv(H_NNL + H_prior)

    return mu_post, S_post


# ##############################################################################
# metropolis_hastings_sample takes data matrices x and y as well as a prior mean
# m and covariance and the number of iterations of a sampling process.
# It returns the sampling chain of the parameters of the logistic regression
# using the Metropolis algorithm.
# ##############################################################################
def metropolis_hastings_sample(X, y, m, S, nb_iter):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: nb_iter x D matrix of posterior samples

    D = X.shape[1]
    samples = np.zeros((nb_iter, D))

    # Task 8:
    # TODO: Write a function to sample from the posterior of the
    # parameters of the logistic regression p(theta | X, y) using the
    # Metropolis algorithm.

    mu_post, S_post = get_posterior(X, y, m, S)
    sample_cur = max_lik_estimate(X, y).reshape(D,1)

    #theta_map = map_estimate(X, y, m, S)

    i = 0
    while(i<nb_iter):
        sample_tmp = np.random.multivariate_normal(np.mean(sample_cur,axis=1), S_post).reshape(D,1)
        p_tmp = (2*np.pi)**(-0.5*D)*np.linalg.det(S_post)**(-0.5)*np.exp(-0.5*((sample_tmp - mu_post).T.dot(np.linalg.inv(S_post)).dot(sample_tmp - mu_post)))
        p_cur = (2 * np.pi) ** (-0.5 * D) * np.linalg.det(S_post) ** (-0.5) * np.exp(
            -0.5 * ((sample_cur - mu_post).T.dot(np.linalg.inv(S_post)).dot(sample_cur - mu_post)))
        ratio = p_tmp.item()/p_cur.item()
        u = np.random.uniform(0,1)
        if ratio>=u:
            samples[i] = sample_tmp.squeeze()
            sample_cur = sample_tmp
        else:
            samples[i] = sample_cur.squeeze()
        i = i+1

    return samples


#####PLOT######
# x, y, x_test = load_data()
# D = x.shape[1]
# nb_iter = 10000
# m = np.zeros((D, 1))
# S = 5 * np.eye(D)
# samples = metropolis_hastings_sample(x, y, m, S, nb_iter)
#
# _,_,h1 = plt.hist(samples[:,1], bins=50, density=True)
# theta_map, S_post = get_posterior(x, y, m, S)
#
# h2, = plt.plot(np.linspace(0,10),
#          stats.multivariate_normal.pdf(np.linspace(0,10), theta_map[1], S_post[1, 1]), 'r')
# plt.legend([h1[0], h2], ['Metropolis samples', 'Laplace posterior'])
# plt.xlabel('theta')
# plt.show()
###############