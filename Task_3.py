import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import integrate

# manual input ----------------------
alpha = 0.9  # alpha = k_1 / k_0
gamma = 0.6  # gamma = a_3 / a
lambda_lambda = 200
# -----------------------------------

I = np.eye(3)
k_0 = 1
k_1 = alpha


def effective_k(p, lambda_p):
    def g(gam):
        if gam <= 1:
            y = 1 / (gam * (1 - gam ** 2) ** 0.5) * np.arctan(((1 - gam ** 2) ** 0.5) / gam)
            return y
        else:
            y = 1 / (2 * gam * (gam ** 2 - 1) ** 0.5) * np.log(
                (gam + (gam ** 2 - 1) ** 0.5) / (gam - (gam ** 2 - 1) ** 0.5))
            return y

    def f_0(gam):
        return (1 - g(gam)) / (2 * (1 - gam ** -2))

    f_gam = f_0(gamma)

    def P_c(fi, theta):
        m_1 = np.cos(theta) * np.sin(fi)
        m_2 = np.sin(theta) * np.sin(fi)
        m_3 = np.cos(fi)

        mm = np.array([[m_1 ** 2, m_1 * m_2, m_1 * m_3],
                       [m_2 * m_1, m_2 ** 2, m_2 * m_3],
                       [m_3 * m_1, m_3 * m_2, m_3 ** 2]]
                      )
        p_el = f_gam * (I - mm) + (1 - 2 * f_gam) * mm
        p_el = p_el / k_0
        return p_el

    def psi(fi):
        y = 1 / (2 * np.pi) * (
                (lambda_p ** 2 + 1) * np.exp(-lambda_p * fi) + lambda_p * np.exp(-lambda_p * np.pi / 2))
        y = p * y
        return y

    def under_integrate_function(fi, theta, i_, j_):
        y = inv(inv(k_1 * I - k_0 * I) + P_c(fi, theta))
        y = y * psi(fi) * np.sin(fi)
        return y[i_][j_]

    k = np.array([])

    for i in range(0, 3):
        for j in range(0, 3):
            def func(fi, theta):
                return under_integrate_function(fi, theta, i, j)

            k = np.append(k, integrate.dblquad(func, 0, 2 * np.pi, 0, np.pi / 2)[0])
    k = k.reshape(3, 3)
    k = k_0 * I + k
    return k


heterogeneity_concentration = np.linspace(0, 1, 2)

for comp_num in range(0, 3):
    K = []
    for _p_ in heterogeneity_concentration:
        K.append(effective_k(_p_, lambda_lambda)[comp_num][comp_num])
    plt.plot(heterogeneity_concentration, K, label=f"компонента {comp_num + 1}")
plt.legend()
plt.grid()
plt.show()
