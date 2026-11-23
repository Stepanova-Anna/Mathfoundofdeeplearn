# Отчёт по лабораторной работе «Математический анализ и матричное исчисление»

Выполнила: Степанова Анна

---

## Общие данные

```python
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)

X_data = np.array([
    [1, 0, -1],
    [0, 1, 1],
    [1, 1, 0],
    [-1, 2, 1],
], dtype=float)

Y_data = np.array([
    [0.2, 0.8],
    [0.7, 0.1],
    [0.6, 0.4],
    [0.1, 0.9],
], dtype=float)

W_initial = np.array([
    [0.2, -0.3],
    [0.4, 0.1],
    [-0.5, 0.2],
], dtype=float)

b_initial = np.array([0.1, -0.2], dtype=float)
regularization = 0.05

def sigmoid(z):
    z_safe = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_safe))

print("X:", X_data.shape, "Y:", Y_data.shape)
print("W:", W_initial.shape, "b:", b_initial.shape)
```

---

## 1. Суммы, индексы и матричная запись

### 1.1. От суммы к матрице

```python
A = np.array([[1, 2, -1], [0, 3, 1]], dtype=float)
B = np.array([[2, 1], [1, -1], [0, 2]], dtype=float)

print("A.shape =", A.shape, "B.shape =", B.shape)

C_matmul = A @ B

C_loops = np.zeros((A.shape[0], B.shape[1]))
for i in range(A.shape[0]):
    for j in range(B.shape[1]):
        for k in range(A.shape[1]):
            C_loops[i, j] += A[i, k] * B[k, j]

print("C_matmul =")
print(C_matmul)
print("C_loops =")
print(C_loops)
print("совпадают:", np.allclose(C_matmul, C_loops))
```

Промежуточные суммы элементов:
```python
print("C11 =", 1*2 + 2*1 + (-1)*0)
print("C12 =", 1*1 + 2*(-1) + (-1)*2)
print("C21 =", 0*2 + 3*1 + 1*0)
print("C22 =", 0*1 + 3*(-1) + 1*2)
```

Получаем C = [[4, -3], [3, -1]]. Свободные индексы i, j остаются у результата; k — суммируемый, исчезает после суммирования. Менять местами j и k нельзя: изменится смысл и размерность.

---

### 1.2. Три индекса и две свёртки

```python
S_tensor = np.array([
    [[1, 0, 0], [0, 1, 0]],
    [[0, 1, 0], [0, 0, 1]],
], dtype=float)
W_tensor = np.array([[1, 0], [0, 2], [1, -1]], dtype=float)
V_tensor = np.array([[1, -1], [2, 1]], dtype=float)

print("S:", S_tensor.shape, "W:", W_tensor.shape, "V:", V_tensor.shape)
```

Размеры: S — (2,2,3), W — (3,2), V — (2,2). Результат — (2,2), свободны n, c; суммируются t, k, d.

Слагаемые для L_11 до удаления нулей:
```python
terms = []
for t in range(2):
    for k in range(3):
        for d in range(2):
            term = S_tensor[0, t, k] * W_tensor[k, d] * V_tensor[d, 0]
            terms.append((t, k, d, term))
for t, k, d, val in terms:
    print(f"S[0,{t},{k}]*W[{k},{d}]*V[{d},0] = {val}")
print("L11 =", sum(v for _, _, _, v in terms))
```

Промежуточная матрица E:
```python
E_matrix = np.einsum('ntk,kd->nd', S_tensor, W_tensor)
print("E =")
print(E_matrix)
```

Затем L:
```python
L_matrix = E_matrix @ V_tensor
print("L =")
print(L_matrix)
```

Получаем E = [[1, 2], [1, 1]], L = [[5, 1], [3, 0]]. Индексная запись в Эйнштейне: суммирование по t, k, d; свободные — n, c.

---

### 1.3. Четыре способа вычисления

```python
L_loops = np.zeros((2, 2))
for n in range(2):
    for c in range(2):
        for t in range(2):
            for k in range(3):
                for d in range(2):
                    L_loops[n, c] += S_tensor[n, t, k] * W_tensor[k, d] * V_tensor[d, c]

E_inter = np.einsum('ntk,kd->nd', S_tensor, W_tensor)
L_matrix = E_inter @ V_tensor

L_once = np.einsum('ntk,kd,dc->nc', S_tensor, W_tensor, V_tensor)

E_twice = np.einsum('ntk,kd->nd', S_tensor, W_tensor)
L_twice = np.einsum('nd,dc->nc', E_twice, V_tensor)

print(L_loops)
print(L_matrix)
print(L_once)
print(L_twice)
print("все равны:",
      np.allclose(L_loops, L_matrix)
      and np.allclose(L_loops, L_once)
      and np.allclose(L_loops, L_twice))
```

Соответствие индексов:

| Вызов | Смысл |
|---|---|
| `'ntk,kd->nd'` | свёртка S по t,k с W по k; свободны n,d |
| `'ntk,kd,dc->nc'` | одна свёртка по t,k,d; свободны n,c |
| `'nd,dc->nc'` | свёртка промежуточной E по d с V по d; свободны n,c |

Все четыре реализации дают одинаковый результат, но размерности нужно контролировать до запуска: `einsum` иногда «молча» работает и при неверных индексах, если размеры случайно совпали.

---

## 2. Функции одной переменной

### 2.1. Производная как предел и как оператор

```python
x0 = 1.0
f = np.exp
exact = np.exp(x0)

h_values = np.logspace(-1, -16, 16)

D_plus  = (f(x0 + h_values) - f(x0)) / h_values
D_cent  = (f(x0 + h_values/2) - f(x0 - h_values/2)) / h_values
D_minus = (f(x0) - f(x0 - h_values)) / h_values

err_plus  = np.abs(D_plus  - exact)
err_cent  = np.abs(D_cent  - exact)
err_minus = np.abs(D_minus - exact)

for h, e1, e2, e3 in zip(h_values, err_plus, err_cent, err_minus):
    print(f"h={h:.1e}  err+= {e1:.2e}  err0= {e2:.2e}  err-= {e3:.2e}")

plt.figure(figsize=(7, 4))
plt.loglog(h_values, err_plus,  'o-', label='прямая')
plt.loglog(h_values, err_cent,  's-', label='центральная')
plt.loglog(h_values, err_minus, '^-', label='обратная')
plt.xlabel('h'); plt.ylabel('|D - e|'); plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

Касательная и функция:
```python
x_grid = np.linspace(0.5, 1.5, 100)
tangent = exact * (x_grid - x0) + exact
plt.figure(figsize=(5, 4))
plt.plot(x_grid, f(x_grid), label='e^x')
plt.plot(x_grid, tangent, '--', label='касательная в x0=1')
plt.axvline(x0, color='gray', lw=0.5)
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

Лучшая точность достигается при h ≈ 10⁻⁵: компромисс между ошибкой усечения (падает с уменьшением h) и ошибкой округления (растёт из-за вычитания близких чисел).

<img width="787" height="465" alt="image" src="https://github.com/user-attachments/assets/afad2bba-be5a-47ad-9d6b-1282aae5c0bc" />


---

### 2.2. Правила дифференцирования

```python
x0 = 1.0
h = 1e-5

funcs = [
    lambda x: 2*x**3 - 5*x**2 + 7*x + 4,
    lambda x: x**2 * np.exp(x),
    lambda x: (2*x**2 + 5)**2,
    lambda x: np.log(2*x**3 + 3*x**2),
]

analytic = [
    6*x0**2 - 10*x0 + 7,
    np.exp(x0) * (x0**2 + 2*x0),
    8*x0 * (2*x0**2 + 5),
    (6*x0**2 + 6*x0) / (2*x0**3 + 3*x0**2),
]

for i, (fn, a) in enumerate(zip(funcs, analytic), 1):
    num = (fn(x0 + h) - fn(x0 - h)) / (2*h)
    print(f"y{i}: аналитика = {a:.6f}, численно = {num:.6f}")
```

В пунктах 3 и 4 внешняя функция — квадрат и логарифм, внутренняя — многочлен. Значения при x=1: 3, 3e ≈ 8.1548, 56, 2.4.

<img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/dec17964-a55d-4155-b25d-6a029e9222bd" />


---

### 2.3. Дифференциал и граница дифференцируемости

```python
x0, dx = 1.0, 0.02
y0  = np.log(1 + x0**2)
y1  = np.log(1 + (x0+dx)**2)
exact_change  = y1 - y0
linear_change = 2*x0/(1 + x0**2) * dx

print("y0 =", y0, "y1 =", y1)
print("точное изменение =", exact_change)
print("линейное приближение =", linear_change)
```

Исследование r(x) = max(0, x):
```python
relu = lambda x: np.maximum(0.0, x)
h = 1e-5
left  = (relu(0) - relu(-h)) / h
right = (relu(h) - relu(0)) / h
cent  = (relu(h) - relu(-h)) / (2*h)
print("левая =", left, "правая =", right, "центральная =", cent)

xs = np.linspace(-1, 1, 201)
plt.figure(figsize=(5, 4))
plt.plot(xs, relu(xs), label='r(x)=max(0,x)')
plt.plot(xs[xs<0], 0*xs[xs<0], '--', label='левая ветвь, k=0')
plt.plot(xs[xs>0], xs[xs>0], '--', label='правая ветвь, k=1')
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

Левая производная 0, правая 1 — не совпадают, обычной производной в нуле нет. Центральная разность возвращает 0.5 — усреднение двух односторонних пределов.

<img width="549" height="435" alt="image" src="https://github.com/user-attachments/assets/f728e7a9-2a50-4c03-8f71-0332c62e87a4" />


---

## 3. Частные производные, градиент и дифференциал

### 3.1. Полный дифференциал многочлена

```python
def z_practice(p):
    x, y = p
    return 5*x**3 + 2*x**2*y**4 - 6*x + 9*y

def grad_z(p):
    x, y = p
    return np.array([15*x**2 + 4*x*y**4 - 6,
                     8*x**2*y**3 + 9])

point = np.array([1.0, -1.0])
print("grad_z(1,-1) =", grad_z(point))

h = 1e-5
e1 = np.array([h, 0]); e2 = np.array([0, h])
num_dx = (z_practice(point + e1) - z_practice(point - e1)) / (2*h)
num_dy = (z_practice(point + e2) - z_practice(point - e2)) / (2*h)
print("численный градиент =", np.array([num_dx, num_dy]))
```

Аналитический градиент в (1,−1): (13, 1). Коэффициент при dx совпадает с ∂z/∂x, при dy — с ∂z/∂y.

---

### 3.2. Локальная плоскость

```python
def f_quadratic(p):
    x, y = p
    return x**2 + x*y + 2*y**2 - 3*x + 1

def grad_f_quad(p):
    x, y = p
    return np.array([2*x + y - 3, x + 4*y])

x_base = np.array([1.0, -1.0])
delta  = np.array([0.01, -0.02])

f0 = f_quadratic(x_base)
g0 = grad_f_quad(x_base)
print("f(x0) =", f0)
print("grad_f(x0) =", g0)

linear_change = g0 @ delta
exact_change  = f_quadratic(x_base + delta) - f0
print("линейный прогноз =", linear_change)
print("точное изменение =", exact_change)
```

Для двумерного случая касательная — плоскость z = f(x0) + ∇f(x0)ᵀ(x − x0).

```python
xs = np.linspace(-0.5, 2.5, 200); ys = np.linspace(-2.5, 0.5, 200)
XX, YY = np.meshgrid(xs, ys)
ZZ = XX**2 + XX*YY + 2*YY**2 - 3*XX + 1
plt.figure(figsize=(5, 4))
plt.contour(XX, YY, ZZ, levels=15)
plt.quiver(*x_base, *g0, color='red', scale=10)
plt.quiver(*x_base, *delta, color='green', scale=0.2)
plt.grid(alpha=0.3); plt.show()
```

<img width="575" height="445" alt="image" src="https://github.com/user-attachments/assets/5c8d3bbd-3855-4a3b-93ff-b20ea31c2097" />


---

### 3.3. Численный градиент

```python
def central_gradient(fn, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (fn(x + e) - fn(x - e)) / (2*h)
    return g

x_base = np.array([1.0, -1.0])
g_exact = grad_f_quad(x_base)

for h in [1e-2, 1e-3, 1e-5, 1e-7]:
    g_num = central_gradient(f_quadratic, x_base, h)
    print(f"h={h:.0e}  ||g_num - g_exact|| = {np.linalg.norm(g_num - g_exact):.3e}")
```

Для центральной разности в n измерениях — 2n обращений к f; цена линейно растёт с размерностью.

---

## 4. Производная по направлению и полная производная

### 4.1. Изменение вдоль направления

```python
x_base = np.array([1.0, -1.0])
g = grad_f_quad(x_base)
v = np.array([3.0, 4.0]) / 5.0

D_v = g @ v
norm_g = np.linalg.norm(g)
max_up   = g / norm_g
max_down = -g / norm_g

print("D_v f =", D_v)
print("||grad|| =", norm_g)
print("макс. рост:", max_up, "скорость",  norm_g)
print("макс. убывание:", max_down, "скорость", -norm_g)

for h in [1e-2, 1e-4, 1e-6]:
    num = (f_quadratic(x_base + h*v) - f_quadratic(x_base - h*v)) / (2*h)
    print(f"h={h:.0e}  численно D_v = {num:.6f}")
```

D_v f = −3.6, функция убывает вдоль v. Скорость максимальна вдоль ∇f/||∇f||; нормировка нужна, чтобы сравнивать именно скорость на единицу пути, а не зависящую от длины v.

---

### 4.2. Все пути влияния переменной

```python
def F_substituted(t):
    return t*t + t**2 + t*t**2 + 2*t**4

def dF_substituted(t):
    return 2*t + 2*t + 2*t + 8*t**3

def dF_total(t):
    x, y = t, t**2
    dFdt_direct = x
    dFdx = t + 2*x + y
    dFdy = x + 4*y
    dxdt, dydt = 1.0, 2*t
    return dFdt_direct + dFdx*dxdt + dFdy*dydt

t0, h = 1.0, 1e-5
num = (F_substituted(t0 + h) - F_substituted(t0 - h)) / (2*h)
print("аналитика (подстановка) =", dF_substituted(t0))
print("аналитика (полная)      =", dF_total(t0))
print("центральная разность    =", num)
```

При t=1 все три значения = 15. Прямой путь t→F даёт слагаемое ∂F/∂t; косвенные — через ∂F/∂x·dx/dt и ∂F/∂y·dy/dt. Пропуск любого пути — неполная производная.

---

## 5. Якобиан, JVP, VJP и цепное правило

### 5.1. Якобиан векторной функции

```python
def vector_function(p):
    x1, x2 = p
    return np.array([x1**2 + x2, np.exp(x1 - x2), x1*x2])

def jacobian_exact(p):
    x1, x2 = p
    return np.array([
        [2*x1, 1.0],
        [np.exp(x1 - x2), -np.exp(x1 - x2)],
        [x2, x1],
    ])

x_star = np.array([1.0, 0.0])
print("f(x*) =", vector_function(x_star))
print("J(x*) =")
print(jacobian_exact(x_star))

h = 1e-5
J_num = np.zeros((3, 2))
for j in range(2):
    e = np.zeros(2); e[j] = h
    J_num[:, j] = (vector_function(x_star + e) - vector_function(x_star - e)) / (2*h)
print("J численно =")
print(J_num)
```

Размерности: вход (2,), выход (3,), якобиан (3,2). Элемент J_{ij} = ∂f_i/∂x_j — чувствительность i-й компоненты выхода к j-й входной координате.

---

### 5.2. JVP без полного якобиана

```python
v = np.array([1.0, -2.0])
J = jacobian_exact(x_star)
exact_jvp = J @ v
print("J @ v =", exact_jvp)

errors = []
h_list = np.logspace(-1, -12, 12)
for h in h_list:
    num_jvp = (vector_function(x_star + h*v/2) - vector_function(x_star - h*v/2)) / h
    errors.append(np.linalg.norm(num_jvp - exact_jvp))

for h, e in zip(h_list, errors):
    print(f"h={h:.1e}  ||err|| = {e:.2e}")

plt.figure(figsize=(6, 3.5))
plt.loglog(h_list, errors, 'o-')
plt.xlabel('h'); plt.ylabel('||JVP_num - JVP_exact||')
plt.grid(alpha=0.3); plt.show()
```

Численный JVP — 2 обращения к f, независимо от n, потому что сдвиг делается сразу вдоль всего v.

<img width="687" height="417" alt="image" src="https://github.com/user-attachments/assets/f0bfae29-6f52-48d7-831a-4492eba24794" />


---

### 5.3. VJP и производная композиции

```python
u = np.array([1.0, 2.0, -1.0])
uJ = u @ jacobian_exact(x_star)
print("u^T J =", uJ)

c_target = np.array([0.0, 1.0, 1.0])

def g(z):
    return 0.5 * np.sum((z - c_target)**2)

def grad_g(z):
    return z - c_target

def h_comp(x):
    return g(vector_function(x))

def grad_h_analytic(x):
    Jx = jacobian_exact(x)
    gz = grad_g(vector_function(x))
    return Jx.T @ gz

x_star = np.array([1.0, 0.0])
print("grad g в f(x*) =", grad_g(vector_function(x_star)))
print("grad h аналитически =", grad_h_analytic(x_star))

def central_gradient(fn, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (fn(x + e) - fn(x - e)) / (2*h)
    return g

print("grad h численно    =", central_gradient(h_comp, x_star))
```

Размерности: ∇_z g — (3,), J — (3,2), Jᵀ∇_z g — (2,). VJP переносит чувствительность скаляра на все входные координаты одним матрично-векторным произведением.

---

## 6. Гессиан и кривизна

### 6.1. Положительная кривизна

```python
def z_min(p):
    x, y = p
    return 2*x**2 + y**2 + x - 3*y + 2*x*y

def grad_min(p):
    x, y = p
    return np.array([4*x + 2*y + 1, 2*y - 3 + 2*x])

A_sys = np.array([[4, 2], [2, 2]], dtype=float)
rhs   = np.array([-1.0, 3.0])
stationary = np.linalg.solve(A_sys, rhs)
print("stationary =", stationary)
print("gradient там:", grad_min(stationary))
print("f(stationary) =", z_min(stationary))

H = np.array([[4, 2], [2, 2]], dtype=float)
print("det H =", np.linalg.det(H))
print("eigvals H =", np.linalg.eigvalsh(H))
```

Точка (−2, 3.5), f = −6.25. det H = 4 > 0, H_11 = 4 > 0, оба собственных значения положительны → локальный минимум.

```python
xs = np.linspace(-4, 0, 200); ys = np.linspace(1, 6, 200)
XX, YY = np.meshgrid(xs, ys)
ZZ = 2*XX**2 + YY**2 + XX - 3*YY + 2*XX*YY
plt.figure(figsize=(5, 4))
plt.contour(XX, YY, ZZ, levels=20)
plt.plot(*stationary, 'r*', markersize=12)
plt.grid(alpha=0.3); plt.show()
```

<img width="545" height="444" alt="image" src="https://github.com/user-attachments/assets/163bc8dc-4c90-4db1-80a3-c91461b8841b" />


---

### 6.2. Неопределённая кривизна и седло

```python
def z_saddle(p):
    x, y = p
    return x**2 - 2*y**2 - 4*x*y + 3*x

def grad_saddle(p):
    x, y = p
    return np.array([2*x - 4*y + 3, -4*x - 4*y])

A_sys = np.array([[2, -4], [-4, -4]], dtype=float)
rhs   = np.array([-3.0, 0.0])
stat_saddle = np.linalg.solve(A_sys, rhs)
print("stationary =", stat_saddle)
print("gradient там:", grad_saddle(stat_saddle))

H_s = np.array([[2, -4], [-4, -4]], dtype=float)
print("det H =", np.linalg.det(H_s))
print("eigvals H =", np.linalg.eigvalsh(H_s))
```

Точка (−0.5, 0.5), det H = −24 < 0, собственные значения 4 и −6 разных знаков → седло. Вдоль одного собственного направления функция растёт, вдоль другого — убывает.

```python
xs = np.linspace(-2, 1, 200); ys = np.linspace(-1, 2, 200)
XX, YY = np.meshgrid(xs, ys)
ZZ = XX**2 - 2*YY**2 - 4*XX*YY + 3*XX
plt.figure(figsize=(5, 4))
plt.contour(XX, YY, ZZ, levels=20)
plt.plot(*stat_saddle, 'r*', markersize=12)
plt.grid(alpha=0.3); plt.show()
```

<img width="577" height="439" alt="image" src="https://github.com/user-attachments/assets/4ffd5d59-e627-4054-8337-6dd23d4cdd24" />


---

### 6.3. Квадратичная модель и HVP

```python
def phi(p):
    x, y = p
    return np.exp(x + 2*y) + x**2

def phi_grad(p):
    x, y = p
    return np.array([np.exp(x + 2*y) + 2*x, 2*np.exp(x + 2*y)])

def phi_hess(p):
    x, y = p
    s = np.exp(x + 2*y)
    return np.array([[s + 2, 2*s], [2*s, 4*s]])

x0 = np.array([0.0, 0.0])
delta = np.array([0.1, 0.1])
print("phi(x0) =", phi(x0))
print("phi_grad(x0) =", phi_grad(x0))
print("phi_hess(x0) =")
print(phi_hess(x0))

linear_part = phi(x0) + phi_grad(x0) @ delta
quad_part   = 0.5 * delta @ phi_hess(x0) @ delta
print("линейное приближение =", linear_part)
print("квадратичный член =", quad_part)
print("квадратичное приближение =", linear_part + quad_part)
print("точное значение =", phi(x0 + delta))

v_hvp = np.array([1.0, -1.0])
print("H @ v =", phi_hess(x0) @ v_hvp)

h = 1e-5
Hv_num = (phi_grad(x0 + h*v_hvp) - phi_grad(x0 - h*v_hvp)) / (2*h)
print("H @ v численно =", Hv_num)
```

Квадратичный член даёт информацию о кривизне: насколько функция отклонится от линейного приближения при шаге Δ.

---

## 7. Матричное дифференцирование

### 7.1. Линейно-квадратичная форма

```python
A_quad = np.array([[2, -1], [3, 1]], dtype=float)
a_quad = np.array([1.0, -2.0])
x_quad = np.array([2.0, -1.0])

f_val = a_quad @ x_quad + x_quad @ A_quad @ x_quad
print("f(x) =", f_val)

grad_exact = a_quad + (A_quad + A_quad.T) @ x_quad
print("grad f =", grad_exact)

H_exact = A_quad + A_quad.T
print("Hessian =")
print(H_exact)

A_skew = 0.5 * (A_quad - A_quad.T)
print("x^T A_skew x =", x_quad @ A_skew @ x_quad)
```

Квадратичная форма xᵀAx зависит только от симметричной части A; поэтому в производной появляется (A + Aᵀ)x.

---

### 7.2. Производная по матрице

```python
a_matrix = np.array([1.0, -2.0])
b_matrix = np.array([3.0, 1.0, -1.0])
M_parameter = np.array([[1, 2, 0], [-1, 3, 4]], dtype=float)

s_val = a_matrix @ M_parameter @ b_matrix
print("s =", s_val)

G_exact = np.outer(a_matrix, b_matrix)
print("a b^T =")
print(G_exact)

h = 1e-6
G_num = np.zeros_like(M_parameter)
for i in range(M_parameter.shape[0]):
    for j in range(M_parameter.shape[1]):
        Mp = M_parameter.copy(); Mp[i, j] += h
        Mm = M_parameter.copy(); Mm[i, j] -= h
        G_num[i, j] = (a_matrix @ Mp @ b_matrix - a_matrix @ Mm @ b_matrix) / (2*h)

print("G_num =")
print(G_num)
print("max |G_exact - G_num| =", np.max(np.abs(G_exact - G_num)))
print("s = tr(M b a^T) =", np.trace(M_parameter @ np.outer(b_matrix, a_matrix)))
```

Значение s = 13, градиент a·bᵀ = [[3,1,−1],[−6,−2,2]]. Форма градиента совпадает с формой M.

---

### 7.3. Сумма потерь, след и матричный градиент

```python
def Q_loss(W):
    residual = X_data @ W - Y_data
    return 0.5 * np.sum(residual**2) / len(X_data)

def Q_grad(W):
    return X_data.T @ (X_data @ W - Y_data) / len(X_data)

def Q_hess_action(W, dW):
    return X_data.T @ (X_data @ dW) / len(X_data)

print("Q(W0) =", Q_loss(W_initial))
G = Q_grad(W_initial)
print("grad Q =")
print(G)

h = 1e-6
G_num = np.zeros_like(W_initial)
for d in range(W_initial.shape[0]):
    for k in range(W_initial.shape[1]):
        Wp = W_initial.copy(); Wp[d, k] += h
        Wm = W_initial.copy(); Wm[d, k] -= h
        G_num[d, k] = (Q_loss(Wp) - Q_loss(Wm)) / (2*h)
print("G_num =")
print(G_num)
print("max |G - G_num| =", np.max(np.abs(G - G_num)))
```

Индексы d, k градиента соответствуют индексам W_{dk}; производная скалярной функции по матрице имеет ту же форму, что и сама матрица.

---

## 8. Сквозная составная функция

### 8.1. Прямое вычисление

```python
N, D, K = X_data.shape[0], X_data.shape[1], W_initial.shape[1]

def forward(W, b):
    Z = X_data @ W + b
    H = sigmoid(Z)
    R = H - Y_data
    data_loss = 0.5 * np.sum(R**2) / N
    reg_loss  = 0.5 * regularization * np.sum(W**2)
    return Z, H, data_loss, reg_loss, data_loss + reg_loss

Z, H, data_loss, reg_loss, L = forward(W_initial, b_initial)

print("X:", X_data.shape, "W:", W_initial.shape, "b:", b_initial.shape,
      "-> Z:", Z.shape, "H:", H.shape)
print("Z =")
print(Z)
print("H =")
print(H)
print("data_loss =", data_loss, "reg_loss =", reg_loss, "L =", L)

n = 0
manual_Z = [sum(X_data[n, d] * W_initial[d, k] for d in range(D)) + b_initial[k]
            for k in range(K)]
print("Z[0] вручную =", manual_Z, "; из forward =", Z[0])
```

b добавляется к каждой строке Z — это расширение по строкам (broadcast), а не произведение двух матриц: b не имеет строковой размерности.

---

### 8.2. Распространение производных

```python
def objective_and_grad(W, b):
    N = X_data.shape[0]
    Z = X_data @ W + b
    H = sigmoid(Z)
    R = H - Y_data

    data_loss = 0.5 * np.sum(R**2) / N
    reg_loss  = 0.5 * regularization * np.sum(W**2)
    loss = data_loss + reg_loss

    G_H = R / N
    G_Z = G_H * H * (1 - H)
    grad_W = X_data.T @ G_Z + regularization * W
    grad_b = G_Z.sum(axis=0)

    return loss, grad_W, grad_b

loss, gW, gb = objective_and_grad(W_initial, b_initial)
print("loss =", loss)
print("grad_W =")
print(gW)
print("grad_b =", gb)
print("формы:", gW.shape, gb.shape)
```

Цепное правило идёт от скаляра L назад к W и b. Слагаемое λW — производная L2-регуляризации.

---

### 8.3. Проверка каждого параметра

```python
def numerical_parameter_gradients(W, b, h=1e-6):
    gW = np.zeros_like(W)
    gb = np.zeros_like(b)
    for d in range(W.shape[0]):
        for k in range(W.shape[1]):
            Wp = W.copy(); Wp[d, k] += h
            Wm = W.copy(); Wm[d, k] -= h
            gW[d, k] = (objective_and_grad(Wp, b)[0] - objective_and_grad(Wm, b)[0]) / (2*h)
    for k in range(b.shape[0]):
        bp = b.copy(); bp[k] += h
        bm = b.copy(); bm[k] -= h
        gb[k] = (objective_and_grad(W, bp)[0] - objective_and_grad(W, bm)[0]) / (2*h)
    return gW, gb

_, gW_an, gb_an = objective_and_grad(W_initial, b_initial)
gW_num, gb_num = numerical_parameter_gradients(W_initial, b_initial)

print("max |ΔW| =", np.max(np.abs(gW_an - gW_num)))
print("max |Δb| =", np.max(np.abs(gb_an - gb_num)))
print("max отн. ошибка W =", np.max(np.abs(gW_an - gW_num) / (np.abs(gW_num) + 1e-12)))
print("max отн. ошибка b =", np.max(np.abs(gb_an - gb_num) / (np.abs(gb_num) + 1e-12)))
```

Порядок диагностики при неудаче: коэффициент 1/N, знак остатка (H − Y, а не Y − H), производная сигмоиды, транспонирование X, суммирование по строкам для b, регуляризация.

---

## 9. Численная оптимизация

### 9.1. Шаг, траектория и устойчивость

```python
learning_rates = [0.1, 1.0, 20.0]
iterations = 100

histories = {}
for lr in learning_rates:
    W = W_initial.copy()
    b = b_initial.copy()
    losses = [objective_and_grad(W, b)[0]]
    grad_norms = []
    for step in range(iterations):
        _, gW, gb = objective_and_grad(W, b)
        grad_norms.append(np.sqrt(np.sum(gW**2) + np.sum(gb**2)))
        W = W - lr * gW
        b = b - lr * gb
        losses.append(objective_and_grad(W, b)[0])
    histories[lr] = {
        "losses": np.array(losses),
        "gnorms": np.array(grad_norms),
        "start": losses[0],
        "final": losses[-1],
        "min":   min(losses),
        "argmin_iter": int(np.argmin(losses)),
        "increases": int(np.sum(np.diff(losses) > 0)),
    }

for lr, info in histories.items():
    print(f"lr={lr}: start={info['start']:.6f}  final={info['final']:.6f}  "
          f"min={info['min']:.6f}  argmin_iter={info['argmin_iter']}  "
          f"increases={info['increases']}")

plt.figure(figsize=(7, 4))
for lr, info in histories.items():
    plt.semilogy(info["losses"], label=f"η={lr}")
plt.xlabel("iteration"); plt.ylabel("L (log)")
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

Ожидаемо: при η=0.1 — медленное устойчивое убывание, при η=1 — быстрое устойчивое, при η=20 — возможны колебания или рост. Оценка должна опираться на всю историю, а не на последнюю точку.

<img width="812" height="460" alt="image" src="https://github.com/user-attachments/assets/af5a6456-dd75-4b4e-8650-88dd17b09dbd" />


---

## 10. Итоговая карта понятий

### 10.1. Резюме

Предел отношения приращений задаёт производную — мгновенную скорость изменения. Дифференциал переносит эту скорость на малое приращение. Для функции многих переменных частные производные собираются в градиент, а он задаёт производную по направлению и полную производную сложной зависимости. Для векторного выхода градиенты компонентов складываются в якобиан, а цепное правило становится произведением якобианов (или JVP/VJP, если якобиан не строится явно). Вторые производные образуют гессиан и описывают локальную кривизну: знак собственных значений отличает минимум от седла. Матричный градиент — та же производная, записанная в форме параметра; через след и индексные тождества она сворачивается в компактные выражения. Конечная разность — универсальный численный тест: подтверждает формулу в точке и обнаруживает ошибку знака или транспонирования. Проверенные градиенты используются в численной оптимизации: шаг слишком мал — медленно, шаг слишком велик — неустойчиво.

### 10.2. Таблица операторов

| Оператор | Вход | Выход | Размерность | Что измеряет | Проверка |
|---|---|---|---|---|---|
| Производная | x ∈ ℝ | скаляр | 1 | локальная скорость | центральная разность |
| Частная производная | x ∈ ℝⁿ | скаляр | 1 | скорость по одной координате | разность по e_i |
| Градиент | x ∈ ℝⁿ | вектор | (n,) | направление и скорость роста | покоординатная разность |
| Дифференциал | x, Δx | скаляр | — | линейное изменение | сравнение с точным при малом Δx |
| Производная по направлению | x, v | скаляр | — | скорость вдоль v | разность вдоль v |
| Полная производная | t | скаляр | — | все пути через t | разность по t |
| Якобиан | x ∈ ℝⁿ | матрица m×n | (m, n) | чувствительность всех компонент | построчная проверка |
| JVP | x, v | вектор | (m,) | изменение выхода вдоль v | разность f(x±hv/2)/h |
| VJP | x, u | вектор | (n,) | чувствительность скаляра к x | сравнение с цепным правилом |
| Гессиан | x | матрица n×n | (n, n) | кривизна / изменение градиента | разность градиентов |
| HVP | x, v | вектор | (n,) | действие гессиана на v | разность градиентов вдоль v |
| Матричный градиент | W | матрица | как W | чувствительность по элементам | поэлементная разность |
| Оптимизация | W, b | обновлённые W, b | как параметры | уменьшение L | история L и ||∇L|| |

### 10.3. Что где в сквозной функции

- Суммы по индексам — в Z = XW + b: каждая Z_{nk} — сумма по d.
- Поэлементное умножение — в σ(Z) и в G_Z = G_H ⊙ H ⊙ (1−H).
- Матричное произведение — в XW и XᵀG_Z: индексы суммируются, свободные остаются.
- Производная по W имеет размер (3,2), потому что W — матрица (3,2): у скалярной функции производная по матрице той же формы.

