import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def probability_density_function(lambda_p, fi):
    psi = 1 / (2 * np.pi) * ((lambda_p ** 2 + 1) * np.exp(-lambda_p * fi) + lambda_p * np.exp(-lambda_p * np.pi / 2))
    return psi


def function_to_integrate(lambda_p):
    def real_function_to_integrate(fi):
        f = 2 * np.pi * probability_density_function(lambda_p, fi) * np.sin(fi)
        return f

    return real_function_to_integrate


integral = []
lamda_array = np.linspace(0, 100, 1000)
for lam in lamda_array:
    result = integrate.quad(function_to_integrate(lam), 0.0, np.pi / 2)[0]
    integral.append(result)

# result = round(result[0], 4)
# print(result)

plt.plot(lamda_array, integral)
plt.xlabel("lamda")
plt.ylabel("integral")
plt.show()
