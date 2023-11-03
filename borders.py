import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5


def voigt(p):
    y = 1 + (alpha - 1) * p
    return y


def reis(p):
    y = 1 + (1 / alpha - 1) * p
    y = 1 / y
    return y


def hashin_shtrikman_bottom(p):
    y = alpha + (1 - p) / (p / (3 * alpha) - 1 / (alpha - 1))
    return y


def hashin_shtrikman_top(p):
    y = 1 + p / ((1 - p) / 3 + 1 / (alpha - 1))
    return y


heterogeneity_concentration = np.linspace(0, 1, 1000)
effective_k_over_k_0_voigt = voigt(heterogeneity_concentration)
effective_k_over_k_0_reis = reis(heterogeneity_concentration)
effective_k_over_k_0_hashin_shtrikman_bottom = hashin_shtrikman_bottom(heterogeneity_concentration)
effective_k_over_k_0_hashin_shtrikman_top = hashin_shtrikman_top(heterogeneity_concentration)

plt.plot(heterogeneity_concentration, effective_k_over_k_0_voigt, "--", label="граница Фойгта", color="black")
plt.plot(heterogeneity_concentration, effective_k_over_k_0_reis, "-.", label="граница Рейса", color="black")
plt.plot(heterogeneity_concentration, effective_k_over_k_0_hashin_shtrikman_bottom,
         label="нижняя граница Хашина-Штрикмана")
plt.plot(heterogeneity_concentration, effective_k_over_k_0_hashin_shtrikman_top,
         label="верхняя граница Хашина-Штрикмана")
plt.xlabel("объемная доля неоднородности")
plt.ylabel("$k^*$/$k_0$")
plt.grid()
plt.legend()
plt.text(0.85, alpha + 0.65 * (1 - alpha), f"α = {alpha}")
# plt.text(0.05, alpha + 0.34*(1 - alpha), f"α = {alpha}")
plt.show()
