from sympy.abc import x, y, v, u
import sympy as sp
import matplotlib.pyplot as plt
import phaseportrait


def replace_coords(system:list[2], point: list)-> list: #замена координат
    new_system = []
    for function in system:
        new_func = function.replace(x, (u + point[0]))
        new_func = new_func.replace(y, (v + point[1]))
        new_system.append(new_func)
    return new_system


def taylor_coefficient(func, point:list[2])-> list:  #разложение в ряд Тейлора до линейных членов
    u0 = point[0]                             # возвращает коэффициенты при u и v
    v0 = point[1]
    matrix_row = [sp.diff(func,u).subs([(u,u0),(v,v0)]), sp.diff(func, v).subs([(u,u0),(v,v0)]) ]
    return matrix_row


def point_type(lamda1, lamda2) -> str:
    if sp.im(lambda1) != 0:
        if sp.re(lambda1) > 0:
            return "Неусточивый фокус"
        elif sp.re(lambda1) < 0:
            return "Устойчивый фокус"
        else:
            return "Центр"
    else:
        if lamda1 > 0 and lamda2 > 0:
            return "Неустойчивый узел"
        elif lambda1 < 0 and lamda2 < 0:
            return "Устойчивый узел"
        else:
            return "Седло (неустойчиво)"


def dF(x, y):
    return [(x + y) ** 2 - 1, -(y) ** 2 + x + 1]


system = dF(x,y)  #система уравнений
stationary_points = sp.solvers.solve(system, (x, y)) #поиск стаицонарных точек

for point in stationary_points:
    replaced_system = replace_coords(system, point) #перенос начала координат в данную особую точку
    print(f'Стационарная точка: {point}')
    print(f"Система после замены переменных:\n [u' = {replaced_system[0]}\n [v' = {replaced_system[1]}")
    row1 = taylor_coefficient(replaced_system[0],[0,0])
    row2 = taylor_coefficient(replaced_system[1], [0,0])
    print(f"Линеаризованная разложением по Тейлору система:\n "
          f"[u' = {row1[0]}u{"%+d" % (row1[1])}v\n [v' = {row2[0]}u{"%+d" % (row2[1])}v")
    A = sp.Matrix((row1,row2))  #матрица линейной системы
    eigen_vectors = A.eigenvects()
    lambda1, lambda2 = eigen_vectors[0][0], eigen_vectors[1][0] #собственные значения
    print("Матрица А:")
    sp.pprint(A)
    print("Собственные значения:")
    sp.pprint([lambda1,lambda2])
    print("Собственные векторы:")
    print(str(eigen_vectors[0][2][0]).replace("Matrix", ""))
    print(str(eigen_vectors[1][2][0]).replace("Matrix", ""))
    print("Тип точки: ", point_type(lambda1, lambda2), "\n\n")


portrait = phaseportrait.PhasePortrait2D(dF,[[-1.5,3.5], [-2.5,2]],xlabel='X', ylabel='Y', Density=2, odeint_method='euler')
portrait.plot()
for point in stationary_points:
    plt.scatter(point[0], point[1], color='black', label=f'({point[0]}, {point[1]})')
plt.legend()
plt.show()