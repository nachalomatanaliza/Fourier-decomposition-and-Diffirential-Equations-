from sympy import sin, pi, integrate, lambdify, pprint, nsimplify, symbols, factor
from sympy.abc import x
import matplotlib.pyplot as plt
import numpy as np


def partial_sum(n: int): #расчет функции частичной суммы порядка n
    part_sum_function = 0
    for i in range(1,n+1):
        part_sum_function += b_сoef_func(i)*sin(pi*i*x/2)
    return lambdify(x, part_sum_function)


def mae(func: np.array, partial_sum: np.array) -> float: #средняя абсолютная ошибка
    return  np.mean(abs(func-partial_sum))


def rmse(func: np.array, partial_sum: np.array) -> float:#корень из средней квадратической ошибки
    return np.sqrt(np.mean((func-partial_sum)**2))


l = 2  ##полупериод разложения
f = lambda \
    x: 3 * x + 1 if x < l and x > 0 else 3 * x - 1 if x > -l and x < 0 else np.nan  ##ф-я продолженная нечетным образом

n = symbols('n', integer=True, positive = True)

b_coef = integrate((2/l)*(3 * x + 1) * sin(pi * n * x / 2),(x,0,l))
b_сoef_func = lambdify(n, b_coef) #вычисление коэффициента Bn и преобразование в функцию для вычисления числовых значений при разных n

print('Коэффициент Bn:')
pprint(nsimplify(b_coef))

S2 = partial_sum(2)
S4 = partial_sum(4)
S6 = partial_sum(6)

x = np.linspace(-l+0.001, l-0.001, 1000)
y = [f(element) for element in x]
S2_y = [S2(element) for element in x]
S4_y = [S4(element) for element in x]
S6_y = [S6(element) for element in x]

print('Среднее абсолютное отклонение: ')
print(f'f(x)-S\u2082: {mae(np.array(y),np.array(S2_y))}')
print(f'f(x)-S\u2084: {mae(np.array(y),np.array(S4_y))}')
print(f'f(x)-S\u2086: {mae(np.array(y),np.array(S6_y))}\n')
print('Корень среднеквадратической ошибки: ')
print(f'f(x)-S\u2082: {rmse(np.array(y),np.array(S2_y))}')
print(f'f(x)-S\u2084: {rmse(np.array(y),np.array(S4_y))}')
print(f'f(x)-S\u2086: {rmse(np.array(y),np.array(S6_y))}')



##построение графиков
#plt.style.use(['dark_background'])
plt.plot(x, y, ls='', ms=1, marker='.', label = "Исходная функция")
plt.plot(x, S2_y, c='pink', ls='', ms=0.9, marker='.', label = "S\u2082")
plt.plot(x, S4_y, c='violet', ls='', ms=0.9, marker='.', label = "S\u2084")
plt.plot(x,S6_y, c='silver',ls='', ms=0.9, marker='.', label = "S\u2086")
plt.axhline(0, 0, lw=0.3, color='white')
plt.axvline(0, 0, lw=0.3, color='white')
plt.legend(loc = 2,frameon = False, markerscale = 10)
plt.title("Приближение f(x) на интервале (-2;2)")
plt.show()