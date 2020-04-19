import numpy as np
import matplotlib.pyplot as plt
import math


def train_backtracking(function, weight, g_traget, g_search, alpha=1, gama=0.8, cls=1):
    """

    :param weight:
    :param g_traget:
    :param g_search:
    :param alpha: initial step back track rate
    :param gama: step back track rate
    :param cls: initial learning rate
    :return:
    """
    print("!!!!")
    f_decent = alpha * cls * np.dot(g_traget.T, g_search)
    weight_n = weight + alpha * g_traget[:-1]
    f_next = function(weight_n)
    f_current = function(weight)

    flag = f_next - f_current - f_decent
    times = 0
    while (flag > 0):
        alpha = gama * alpha
        f_decent = alpha * cls * np.dot(g_traget.T, g_search)
        f_next = function(weight + alpha * g_traget[:-1])
        f_current = function(weight)

        flag = f_next - f_current - f_decent
        times += 1
    print("search %d times\n" % times)
    return alpha

def lagrange_loss(loss_f,constrain_f,lambda_lag):
    return lambda param: loss_f(param) + np.dot(lambda_lag.T, constrain_f(param))

def sigmiod(s_in):
    return np.exp(s_in) / (1 + np.exp(s_in))


nba_data = np.loadtxt("nbadata.txt")
print(nba_data.shape)

NBA_champion = nba_data[nba_data[:, -1] == 1]
NBA_champion_balanced = np.repeat(NBA_champion, 20, axis=0)
print(NBA_champion.shape, NBA_champion_balanced.shape)
nba_data_balanced = np.concatenate((NBA_champion_balanced, nba_data), axis=0)
print(nba_data_balanced.shape)

x = nba_data_balanced[:, 0:22].reshape(nba_data_balanced.shape[0], -1)
y = nba_data_balanced[:, -1].reshape(1, -1).T
c = np.ones(nba_data_balanced.shape[0]).reshape(1, -1).T
print(x.shape, c.shape)
x_hom = np.concatenate((c, x), axis=1).T
print(x_hom.shape)

# w = np.random.normal(loc=0.0, scale=1, size=(nba_data.shape[1],1)) # print initial weights
w = np.zeros((nba_data.shape[1], 1))  # print initial weights
a = 1 + np.exp(np.dot(x_hom.T, w))

linear_regression = np.dot(x_hom.T, w)
logistic_regression = sigmiod(linear_regression)
grad_logistic_regression = np.dot(x_hom, (logistic_regression - y))
gradient_norm = np.linalg.norm(grad_logistic_regression)

# np.diag([1,2,3,4])
Hessian_logistic_regression = np.dot(x_hom, np.diag(logistic_regression) * x_hom.T)
# print(logistic_regression.shape,grad_logistic_regression.shape,Hessian_logistic_regression.shape)


def loss_function(weight):
    lin_reg = np.dot(x_hom.T, weight)
    loss_temp = np.ones_like(lin_reg)
    loss_temp[lin_reg < 3] = np.log(1 + np.exp(lin_reg[lin_reg < 3]))
    # loss = -np.sum(y * lin_reg - np.log(1 + np.exp(lin_reg)))
    loss = -np.sum(y * lin_reg - loss_temp)
    return loss


# constrains
num_A_cons = 1
A_cons = np.ones([num_A_cons, nba_data.shape[1]])
b_cons = np.ones([num_A_cons, 1])
linear_constrain = lambda param: np.dot(A_cons, param) - b_cons
lambda_lagrange = np.ones([num_A_cons, 1])

import time
t_s = time.monotonic()

lr = 1
epoch = 0
gradient_cache = [gradient_norm,]
loss_cache = [loss_function(w).item()]
while  gradient_norm>= 1e-5:
    grad_lagrange = np.concatenate((grad_logistic_regression, -b_cons + np.dot(A_cons, w)))
    # delta = lr * np.dot(inv_Hessian_lagrange, -grad_lagrange)

    Hessian_lagrange_t1 = np.concatenate((Hessian_logistic_regression, A_cons.T), axis=1)
    Hessian_lagrange_t2 = np.concatenate((A_cons,np.zeros((num_A_cons,num_A_cons))), axis=1)
    Hessian_lagrange = np.concatenate((Hessian_lagrange_t1, Hessian_lagrange_t2), axis=0)
    delta = np.linalg.solve(Hessian_lagrange,-grad_lagrange)

    delta_w = delta[: nba_data.shape[1]].reshape((1, -1)).T
    delta_lambda_lagrange = delta[nba_data.shape[1]:].reshape((1, -1)).T - lambda_lagrange

    alpha = train_backtracking(lagrange_loss(loss_function,linear_constrain,lambda_lagrange), w, grad_lagrange, delta)
    # alpha = 1
    w = w + alpha * delta_w
    lambda_lagrange = lambda_lagrange + alpha * delta_lambda_lagrange

    linear_regression = np.dot(x_hom.T, w)
    logistic_regression = sigmiod(linear_regression)
    grad_logistic_regression_k = grad_logistic_regression.copy()
    grad_logistic_regression = np.dot(x_hom, (logistic_regression - y))
    Hessian_logistic_regression = np.dot(x_hom, np.diag(logistic_regression) * x_hom.T)

    stop_cond = np.abs(np.dot(grad_logistic_regression.T, delta_w)) + np.sum(
        np.dot(np.abs(lambda_lagrange.T), np.abs(b_cons - np.dot(A_cons, w))))
    gradient_norm = stop_cond
    epoch = epoch + 1
    if epoch % 1 == 0:
        print("epoch = %6f, gradient_norm = %6f" % (epoch, np.linalg.norm(grad_logistic_regression)))
    if np.isnan(gradient_norm):
        print("NaN!!!")
        break
    gradient_cache.append(gradient_norm.item())
    loss_cache.append(loss_function(w).item())

t_e = time.monotonic()

s, ms = divmod((t_e - t_s) * 1000, 1000)
m, s = divmod(s, 60)
h, m = divmod(m, 60)
print("%d:%02d:%02d:%02d" % (h, m, s, ms))

# test
print(w)
print("constrain result G(w) = %e"% linear_constrain(w).item())
print("target function result F(w) = %e"% loss_function(w).item())
print("terminate metric-norm of the lagrange function gradient = %e"% gradient_cache[-1])
result = np.exp(np.dot(x_hom.T, w)) / (1 + np.exp(np.dot(x_hom.T, w)))
# print(result >= 0.5)
error_in = (result >= 0.5) ^ (y == 1)
train_error = sum(error_in) / len(error_in)
print("initial train_error at threshold=0.5: ", train_error)

fig = plt.figure(figsize=(10, 10),tight_layout=True)
plt.subplot(2,1,1)
plt.plot(np.arange(epoch + 1), np.log(gradient_cache) / math.log(10), label='Validation')
plt.title("""Newton method on Logistic Regression with balanced data
 $log(|g^T \Delta x^{(k)}|+\sum_{i=0}^{r}|\lambda_i||A_ix^{(k)}+b_i|$ vs Epochs""")
plt.xlabel('Epochs')
plt.ylabel('log(||g||')
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(epoch + 1), np.log(loss_cache) / math.log(10), label='Validation')
plt.title("""Newton method on Logistic Regression with balanced data
 loss vs Epochs""")
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.savefig('Newton.png')
plt.show()
# # sweep
# error_in = (y == 1)
# who_num = sum(error_in)
# print("who_num: ", who_num)
# train_error = sum(error_in) / len(error_in)
# train_error_min = train_error
# threshold_best = 0
# print("""Sweep start naive predict all NBA team will not win champions
# train_error_min = %4f threshold_best = 0""" % train_error_min)
# for threshold in np.arange(0.25, 0.75, 1 / 256):
#     error_in = (result >= threshold) ^ (y == 1)
#     train_error = sum(error_in) / len(error_in)
#     # print("threshold: %4f ,train_error: %4f" % (threshold, train_error))
#     if train_error < train_error_min:
#         train_error_min = train_error
#         threshold_best = threshold
# print("threshold_best: %4f ,train_error_min: %4f" % (threshold_best, train_error_min))


# q,r = np.linalg.qr(A) # reduced
# q,r = np.linalg.qr(A,mode="complete") # full
