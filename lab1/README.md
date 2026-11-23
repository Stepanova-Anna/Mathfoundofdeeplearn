# Отчёт по лабораторной работе «Линейная алгебра для Data Science»

Выполнила: Степанова Анна

---

## 1. Векторы

### 1.1. Координаты и длина

```python
import numpy as np

A = np.array([1, 2, 3], dtype=float)
B = np.array([3, -4, 6], dtype=float)
AB = B - A
length_AB = np.linalg.norm(AB)
print("AB =", AB)
print("length_AB =", length_AB)
```

Результат: AB = (2, -6, 3), length_AB = 7.0.

---

### 1.2. Угол между векторами

```python
a = np.array([-1, 1, 0], dtype=float)
b = np.array([1, -2, 2], dtype=float)
dot = a @ b
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
cos_phi = dot / (norm_a * norm_b)
phi_deg = np.degrees(np.arccos(np.clip(cos_phi, -1.0, 1.0)))
print("dot =", dot)
print("norm_a =", norm_a)
print("norm_b =", norm_b)
print("cos_phi =", cos_phi)
print("phi_deg =", phi_deg)
```

Результат: dot = -3.0, norm_a = 1.4142, norm_b = 3.0, cos_phi = -0.7071, phi_deg = 135.0. Угол тупой.

---

### 1.3. Ортогональность

```python
a = np.array([3, 2, -1], dtype=float)
x = -2.0
b = np.array([2, x, 2], dtype=float)
print("a @ b =", a @ b)
```

Результат: a @ b = 0.0. Значит x = -2.

---

### 1.4. Векторы объектов в данных

```python
X = np.array([
    [1, 2],
    [2, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 7],
    [6, 8],
    [7, 10],
], dtype=float)

delta = X[-1] - X[0]
dist = np.linalg.norm(delta)
print("delta =", delta)
print("dist =", dist)
```

Результат: delta = (6, 8), dist = 10.0. Координаты имеют разные единицы, поэтому евклидова норма смешивает шкалы; перед сравнением нужна стандартизация.

---

## 2. Матрица данных и матричные операции

### 2.1. Сложение матриц

```python
A = np.array([[3, 5, 7], [2, -1, 0], [4, 3, 2]], dtype=float)
B = np.array([[1, 2, 4], [2, 3, -2], [-1, 0, 1]], dtype=float)
S = A + B
print(S)
```

Результат:
```
[[ 4.  7. 11.]
 [ 4.  2. -2.]
 [ 3.  3.  3.]]
```

---

### 2.2. Линейная комбинация матриц

```python
A = np.array([[3, 5], [4, 1]], dtype=float)
B = np.array([[2, 3], [1, -2]], dtype=float)
R = 2 * A + 5 * B
print(R)
```

Результат:
```
[[16. 25.]
 [13. -8.]]
```

---

### 2.3. Произведение матриц

```python
A = np.array([[1, 3, 1], [2, 0, 4], [1, 2, 3]], dtype=float)
B = np.array([[2, 1, 0], [1, -1, 2], [3, 2, 1]], dtype=float)
AB = A @ B
BA = B @ A
print("AB =")
print(AB)
print("BA =")
print(BA)
print("AB == BA:", np.array_equal(AB, BA))
```

Результат: AB = [[8,0,7],[16,10,4],[13,5,7]], BA = [[4,6,6],[1,7,3],[8,11,14]]. Матрицы не перестановочны.

---

### 2.4. Размерность матрицы данных

```python
print("X.shape =", X.shape)
print("y.shape =", np.array([48,52,55,63,70,78,85,94]).shape)
print("Средние =", X.mean(axis=0))
print("X.T @ X =")
print(X.T @ X)
```

Результат: X.shape = (8, 2), средние = [3.75, 5.125], X.T @ X = [[144,197],[197,271]].

---

## 3. Линейные отображения

### 3.1. Матрица преобразования

```python
T = np.array([
    [0, 1, -1],
    [1, 0, 0],
    [1, 0, 1]
], dtype=float)
for e in np.eye(3):
    print("T @ e =", T @ e)
```

Результат: T @ e1 = (0,1,1), T @ e2 = (1,0,0), T @ e3 = (-1,0,1). Матрица T собрана верно.

---

### 3.2. Образ вектора

```python
A = np.array([[1, -1, 0], [3, 0, 1], [1, 1, -2]], dtype=float)
a = np.array([1, -2, 3], dtype=float)
print("A @ a =", A @ a)
```

Результат: A @ a = (3, 6, -7).

---

## 4. Определитель и ранг

### 4.1. Определитель

```python
A = np.array([[5, 2, 3], [-4, 3, 5], [2, 3, 4]], dtype=float)
print("det A =", np.linalg.det(A))
```

Результат: det A ≈ -17.0. Матрица обратима.

---

### 4.2. Ранги

```python
A1 = np.array([[1, 2], [0, 5], [-1, 4], [1, 1]], dtype=float)
A2 = np.array([[3, -5, -1, -2], [8, -6, 3, -7], [2, 4, 5, -3]], dtype=float)
print("rank A1 =", np.linalg.matrix_rank(A1))
print("rank A2 =", np.linalg.matrix_rank(A2))
```

Результат: rank A1 = 2, rank A2 = 2.

---

## 5. Базис и координаты

### 5.1. Проверка базиса

```python
a = np.array([0, 1, 2], dtype=float)
b = np.array([1, 0, 1], dtype=float)
c = np.array([-1, 2, 4], dtype=float)
B = np.column_stack([a, b, c])
print("det B =", np.linalg.det(B))
print("rank B =", np.linalg.matrix_rank(B))
```

Результат: det B = -1.0, rank B = 3. Векторы образуют базис.

---

### 5.2. Координаты в новом базисе

```python
p = np.array([2, 1, -1], dtype=float)
r = np.array([1, -1, 1], dtype=float)
q = np.array([0, 3, 2], dtype=float)
x = np.array([1, -4, 4], dtype=float)
B = np.column_stack([p, r, q])
coords = np.linalg.solve(B, x)
print("coords =", coords)
print("B @ coords =", B @ coords)
```

Результат: coords = [-1, 3, 0], B @ coords = [1, -4, 4].

---

## 6. Системы линейных уравнений

### 6.1. Квадратная система

```python
A = np.array([[2, -1, 1], [3, 2, 2], [1, -2, 1]], dtype=float)
b = np.array([2, -2, 1], dtype=float)
x = np.linalg.solve(A, b)
print("x =", x)
print("residual =", A @ x - b)
```

Результат: x = [2, -1, -3], residual = [0, 0, 0].

---

### 6.2. Недоопределённая система

Параметрическое решение: x1 = 13/23 - 2t, x2 = 50/23, x3 = t.

```python
A = np.array([[3, -4, 6], [2, 5, 4]], dtype=float)
b = np.array([-7, 12], dtype=float)
for t in [-2.0, 0.0, 3.0]:
    x = np.array([13/23 - 2*t, 50/23, t], dtype=float)
    print("t =", t, "residual =", A @ x - b)
```

Результат: для всех t невязка близка к нулю.

---

## 7. Проекция на прямую

```python
y = np.array([3.0, 1.0])
a = np.array([1.0, 1.0])
c = (a @ y) / (a @ a)
y_proj = c * a
res = y - y_proj
print("c =", c)
print("y_proj =", y_proj)
print("res =", res)
print("a @ res =", a @ res)
```

Результат: c = 2.0, y_proj = [2, 2], res = [1, -1], a @ res = 0.0.

<img width="572" height="538" alt="image" src="https://github.com/user-attachments/assets/cc0b5f08-62a3-43a2-ae75-368109246d88" />


---

## 8. Метод наименьших квадратов

### 8.1. Матричная запись регрессии

```python
X = np.array([
    [1, 2], [2, 2], [2, 3], [3, 4], [4, 5], [5, 7], [6, 8], [7, 10]
], dtype=float)
y = np.array([48, 52, 55, 63, 70, 78, 85, 94], dtype=float)
D = np.column_stack([np.ones(len(X)), X])
beta, *_ = np.linalg.lstsq(D, y, rcond=None)
y_hat = D @ beta
print("beta =", beta)
print("y_hat =", y_hat)
```

Результат: beta ≈ [38.9787, 4.8936, 2.1064]. Оба коэффициента положительны.

---

### 8.2. Невязка и качество приближения

```python
residuals = y - y_hat
SSE = np.sum(residuals**2)
RMSE = np.sqrt(np.mean(residuals**2))
R2 = 1 - SSE / np.sum((y - y.mean())**2)
print("residuals =", residuals)
print("SSE =", SSE)
print("RMSE =", RMSE)
print("R2 =", R2)
```

Результат: SSE ≈ 2.8085, RMSE ≈ 0.5925, R2 ≈ 0.9985.

<img width="1360" height="485" alt="image" src="https://github.com/user-attachments/assets/09be452c-53f9-4e47-894e-4b051023ad32" />


---

### 8.3. Регрессия как проекция

```python
print("D.T @ (y - y_hat) =", D.T @ (y - y_hat))
```

Результат: вектор, близкий к [0, 0, 0].

---

## 9. PCA

### 9.1. Результаты готового алгоритма

```python
mean_X = X.mean(axis=0)
scale_X = X.std(axis=0, ddof=0)
Z = (X - mean_X) / scale_X
C = Z.T @ Z / len(Z)
eigenvalues, eigenvectors = np.linalg.eigh(C)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
explained_ratio = eigenvalues / eigenvalues.sum()
print("C =")
print(C)
print("eigenvalues =", eigenvalues)
print("explained_ratio =", explained_ratio)
print("first component =", eigenvectors[:, 0])
```

Результат: C ≈ [[1, 0.9877], [0.9877, 1]], eigenvalues ≈ [1.9877, 0.0123], explained_ratio ≈ [0.9938, 0.0062], first component ≈ [0.7071, 0.7071].

<img width="1362" height="548" alt="image" src="https://github.com/user-attachments/assets/9e023887-246e-4e16-8d30-1c7b5ac21064" />


---

### 9.2. Польза и ограничения

Сжатие двух коррелирующих признаков до одной координаты полезно для визуализации и уменьшения размерности. Ограничения: PCA чувствителен к масштабу, ищет только линейные направления, не использует y при выборе компонент.

---

## 10. SVD

### 10.1. Малоранговое приближение изображения

Для фрагмента m=126, n=263, mn=33138. Число хранимых значений при ранге k: k(m+n+1)=390k.

```python
m, n = 126, 263
for k in [2, 5, 20, 50]:
    storage = k * (m + n + 1)
    ratio = storage / (m * n)
    print(f"k={k}: storage={storage}, ratio={ratio:.4f}")
```

Результат: k=2: 780 (2.35%), k=5: 1950 (5.88%), k=20: 7800 (23.54%), k=50: 19500 (58.85%).

---

### 10.2. Связь PCA и SVD

```python
U_data, s_data, Vt_data = np.linalg.svd(Z, full_matrices=False)
pca_dir = eigenvectors[:, 0]
svd_dir = Vt_data[0]
print("PCA dir =", pca_dir)
print("SVD dir =", svd_dir)
print("|dot| =", abs(pca_dir @ svd_dir))
```

Результат: |dot| ≈ 1.0 — направления совпадают с точностью до знака.

---

## 11. Итоговая интерпретация

**Карта понятий.** Вектор наблюдения описывает одного студента. Матрица данных собирает наблюдения в таблицу. Линейное отображение переводит признаки в новое представление. Ранг показывает число независимых направлений. Система уравнений восстанавливает параметры. Проекция и метод наименьших квадратов дают лучшее приближение, когда точного решения нет. Главная компонента оставляет наиболее информативное направление. Сингулярное число показывает вклад направления. Малоранговое приближение даёт компромисс между точностью и объёмом хранения.

**Три результата:**

| Результат | Вход | Выход | Польза | Потери |
|---|---|---|---|---|
| Регрессия | D, y | beta, y_hat | Прогноз, описание связи | Переобучение, не причинный вывод |
| PCA | Z | Компоненты, координаты | Сжатие, визуализация | Игнорирует y, теряет часть дисперсии |
| SVD | A | U, S, V^T, A_k | Малоранговое сжатие | При малом k теряются детали |
