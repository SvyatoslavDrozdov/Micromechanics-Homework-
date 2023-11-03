import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# manual input ----------------------
alpha = 0.9  # alpha = k_1/k_0
gamma = 0.6
# -----------------------------------


I = np.eye(3)
mm = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
k_0 = 1
k_1 = alpha
P_sp = 1 / (3 * k_0) * I


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
P_el = f_gam * (I - mm) + (1 - 2 * f_gam) * mm
P_el = P_el / k_0

first_part = inv(inv(k_1 * I - k_0 * I) + P_sp)
second_part = inv(inv(k_1 * I - k_0 * I) + P_el)


# I made matrix just for test
def p_sf_and_p_el(p, comp_num):
    y = []
    matrix = []
    for p_i in p:
        y.append((I + p_i / (2 * k_0) * first_part + p_i / (2 * k_0) * second_part)[comp_num][comp_num])
        matrix.append(I + p_i / (2 * k_0) * first_part + p_i / (2 * k_0) * second_part)
    return [y, matrix]


def p_sp(p, comp_num):
    y = []
    matrix = []
    for p_i in p:
        y.append((I + p_i / k_0 * first_part)[comp_num][comp_num])
        matrix.append(I + p_i / k_0 * first_part)
    return [y, matrix]


def p_el(p, comp_num):
    y = []
    matrix = []
    for p_i in p:
        y.append((I + p_i / k_0 * second_part)[comp_num][comp_num])
        matrix.append(I + p_i / k_0 * second_part)
    return [y, matrix]


heterogeneity_concentration = np.linspace(0, 1, 100)
do = 1
if do:
    for i in range(0, 3):
        plt.figure(figsize=(7, 5))
        plt.title(f"компонента номер {i + 1}")
        conductivity_p_sf_and_p_el = p_sf_and_p_el(heterogeneity_concentration, i)[0]
        conductivity_p_sp = p_sp(heterogeneity_concentration, i)[0]
        conductivity_p_el = p_el(heterogeneity_concentration, i)[0]

        plt.plot(heterogeneity_concentration, conductivity_p_sf_and_p_el,
                 label="сферические и эллиптические неоднородности")
        plt.plot(heterogeneity_concentration, conductivity_p_sp, label="сферические неоднородности")
        plt.plot(heterogeneity_concentration, conductivity_p_el, label="эллиптические неоднородности")
        plt.text(0.85, alpha + 0.65 * (1 - alpha), f"α = {alpha}")
        plt.grid()
        plt.legend()

do_2 = 0
if do_2:
    for i in range(0, 3):
        plt.title(f"компонента номер {i + 1}")
        conductivity_p_el = p_el(heterogeneity_concentration, i)[0]
        plt.plot(heterogeneity_concentration, conductivity_p_el, label="эллиптические неоднородности", color="b")
        plt.text(0.85, alpha + 0.65 * (1 - alpha), f"α = {alpha}")
        plt.grid()
        plt.legend()

plt.show()
