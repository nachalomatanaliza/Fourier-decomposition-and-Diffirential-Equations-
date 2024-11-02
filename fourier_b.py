from sympy import cos, sin, pi, integrate, lambdify, symbols, pprint, nsimplify
from sympy.abc import x
import matplotlib.pyplot as plt
import numpy as np


def partial_sum(n: int): #расчет функции частичной суммы порядка n
    part_sum_function = a_0/2
    for i in range(1,n+1):
        part_sum_function += a_coef_func(i)*cos(pi*i*x/l) + b_coef_func(i)*sin(pi*i*x/l)
    return lambdify(x, part_sum_function)


def mae(func: np.array, partial_sum: np.array) -> float: #средняя абсолютная ошибка
    return  np.mean(abs(func-partial_sum))


def rmse(func: np.array, partial_sum: np.array):#корень из средней квадратической ошибки
    return np.sqrt(np.mean((func-partial_sum)**2))

l = 2  ##полупериод разложения
f = lambda \
    x: 3 * x + 1 if x < l and x > 0 else 0 if x > -l and x <= 0 else np.nan

n = symbols('n', integer=True, positive = True)

a_0 = integrate(1/l*(3*x+1),(x,0,2)) #вычисление Ао
a_сoef = integrate(1/l*(3 * x + 1) * cos(pi * n * x / 2),(x, 0, l))#вычисление коэффициента An
a_coef_func = lambdify(n, a_сoef) #преобразование An в функцию для вычисления числовых значений при разных n

b_сoef = integrate(1/l*(3 * x + 1) * sin(pi * n * x / 2), (x, 0, l)) #вычисление коэффициента Bn
b_coef_func = lambdify(n,b_сoef) #gреобразование в функцию для вычисления числовых значений при разных n

print('Коэффициент А0:')
print(int(a_0))
print('Коэффициент An:')
pprint(nsimplify(a_сoef))
print('\nКоэффициент Bn:')
pprint(nsimplify(b_сoef))

S3 = partial_sum(3)
S5 = partial_sum(5)

x = np.linspace(-l+0.001, l-0.001, 1000)
y = [f(element) for element in x]
S3_y = [S3(element) for element in x]
S5_y = [S5(element) for element in x]

print('Среднее абсолютное отклонение: ')
print(f'f(x)-S\u2083: {mae(np.array(y),np.array(S3_y))}')
print(f'f(x)-S\u2085: {mae(np.array(y),np.array(S5_y))}')
print('Корень среднеквадратической ошибки: ')
print(f'f(x)-S\u2083: {rmse(np.array(y),np.array(S3_y))}')
print(f'f(x)-S\u2085: {rmse(np.array(y),np.array(S5_y))}')

#plt.style.use(['dark_background'])
plt.axhline(0, 0, lw=0.3, color='white')
plt.axvline(0, 0, lw=0.3, color='white')
plt.plot(x, y, ls='', ms=1, marker='.', label = "Исходная функция")
plt.plot(x, S3_y, c='violet', ls='', ms=0.9, marker='.', label="S\u2083")
plt.plot(x, S5_y, c='silver', ls='', ms=0.9, marker='.', label="S\u2085")
plt.legend(loc=2, frameon=False, markerscale=10)
plt.title("Приближение f(x) на интервале (-2;2)")
plt.show()