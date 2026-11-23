# Отчёт по лабораторной работе «Языковая модель на Elman RNN»

Выполнила: Степанова Анна

---

## 1. Цель и выбранные константы

Цель — проследить полный путь данных в языковой модели: от строк русского текста и индексов слов до распределения следующего слова, функции потерь и обновления параметров. Реализация Elman RNN выполнена вручную, без `nn.RNN`; используются только `nn.Parameter`, `nn.Embedding`, `nn.Linear`.

**Корпус:** `maxzt/RuHeritage-Corpus`, ревизия `b2d01eaad3ddb119fe10c4cb17a86e47415a77cb`, каталог `data`, поле `text`. Данные читаются потоково, в память не загружается весь корпус.

**Зафиксированные константы:**

| Константа | Значение | Роль |
|---|---|---|
| `SEED` | 42 | воспроизводимость |
| `MAX_DOCUMENTS` | 1200 | объём учебной выборки |
| `MAX_WORDS_PER_DOCUMENT` | 160 | ограничение длины документа |
| `MAX_ORDINARY_WORDS` | 6000 | размер обычной части словаря |
| `MAX_SEQUENCE_LENGTH` | 32 | длина окна после сдвига |
| `EMBEDDING_DIM` | 128 | размер эмбеддинга |
| `HIDDEN_DIM` | 128 | размер скрытого состояния |
| `NUM_LAYERS` | 2 | число рекуррентных слоёв |
| `BATCH_SIZE` | 64 | размер мини-пакета |
| `LEARNING_RATE` | 2e-3 | шаг AdamW |
| `WEIGHT_DECAY` | 1e-4 | L2-регуляризация |
| `MAX_EPOCHS` | 12 | верхний предел эпох |
| `PATIENCE` | 3 | порог ранней остановки |

**Специальные индексы:**

```python
SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = range(4)
print(PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX)
```

Получаем `0 1 2 3` — этот контракт используется дальше в маске потерь и в сдвиге.

---

## 2. Получение корпуса и нормализация

Нормализация приводит текст к нижнему регистру, заменяет «ё» на «е» и оставляет только русские словарные токены. Каждый непустой документ после нормализации проверяется на минимальную длину и ограничивается 160 словами. Каждый шестой непустой документ направляется в валидацию, остальные — в обучение.

```python
WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)

def normalize_text(text):
    text = text.lower().replace("ё", "е")
    return WORD_RE.findall(text)

ds = load_dataset(
    DATASET_NAME,
    data_dir=DATASET_DATA_DIR,
    revision=DATASET_REVISION,
    split="train",
    streaming=True,
)

train_documents, val_documents = [], []
non_empty_index = 0

for example in ds:
    text = example.get(TEXT_FIELD, "")
    if not text or not text.strip():
        continue
    tokens = normalize_text(text)[:MAX_WORDS_PER_DOCUMENT]
    if len(tokens) < 1:
        continue
    non_empty_index += 1
    if non_empty_index % 6 == 0:
        val_documents.append(tokens)
    else:
        train_documents.append(tokens)
    if non_empty_index >= MAX_DOCUMENTS:
        break

print(f"train docs: {len(train_documents)}, val docs: {len(val_documents)}")
print("Пример train:", train_documents[0][:15])
print("Пример val:  ", val_documents[0][:15] if val_documents else "нет")
```

**Результат.** Документов в обучении — 1000, в валидации — 200 (при `MAX_DOCUMENTS=1200`). Пример нормализованных токенов из обучающей части: `['иван', 'и', 'марья', 'не', 'могли', 'понять', 'что', ...]`. Валидационная часть — литературные тексты, которых не было в обучении.

Поле `text` использовано именно у RuHeritage-Corpus, ревизия зафиксирована — вход этапа воспроизводим.

---

## 3. Словарь и сдвинутые пары

Словарь строится **только** по обучающим документам: сначала зарезервированы четыре специальных токена с индексами 0–3, затем добавляются самые частые обычные слова (не более 6000). Всё, что не попало в словарь, отображается в `<unk>`.

```python
counter = Counter()
for doc in train_documents:
    counter.update(doc)

stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
for word, _freq in counter.most_common(MAX_ORDINARY_WORDS):
    if word not in stoi:
        stoi[word] = len(stoi)
itos = {i: w for w, i in stoi.items()}
VOCAB_SIZE = len(stoi)

assert stoi["<pad>"] == 0 and stoi["<unk>"] == 1
assert stoi["<bos>"] == 2 and stoi["<eos>"] == 3
print(f"VOCAB_SIZE = {VOCAB_SIZE}")
```

Размер словаря — около 6 004 (4 специальных + обычные слова, реально встретившиеся в обучении). Если слово из валидации отсутствует в `stoi`, функция `encode` возвращает `UNK_IDX`, словарь при этом не расширяется:

```python
def encode(tokens):
    return [stoi.get(w, UNK_IDX) for w in tokens]
```

**Сдвинутые пары.** Для каждого документа строится индексная последовательность `[BOS] + doc + [EOS]`, режется на окна не длиннее 33, и для каждого окна вход — все токены кроме последнего, цель — все токены кроме первого.

```python
def make_pairs(documents):
    pairs = []
    window = MAX_SEQUENCE_LENGTH + 1
    for doc in documents:
        idx = [BOS_IDX] + encode(doc) + [EOS_IDX]
        start = 0
        while start < len(idx) - 1:
            chunk = idx[start:start + window]
            if len(chunk) < 2:
                break
            pairs.append((chunk[:-1], chunk[1:]))
            if len(chunk) < window:
                break
            start += MAX_SEQUENCE_LENGTH
    return pairs

train_pairs = make_pairs(train_documents)
val_pairs = make_pairs(val_documents)
print(f"train pairs = {len(train_pairs)}, val pairs = {len(val_pairs)}")
```

**Пример пары на коротком фрагменте `[bos, a, b, eos]`:**

```python
sample_inp, sample_tgt = train_pairs[0]
print(" input :", [itos[i] for i in sample_inp])
print(" target:", [itos[i] for i in sample_tgt])
```

Вход = `[bos, a, b]`, цель = `[a, b, eos]`. Каждая цель сдвинута на один токен влево. Это не метки классов документа — модель на каждой позиции решает классификацию на `V` классов.

---

## 4. Dataset, collate и DataLoader

Одна пара имеет переменную длину `(L)` с `1 ≤ L ≤ 32`. Функция `collate_fn` дополняет короткие строки справа индексом 0 и возвращает два тензора `(B, T)` с `T ≤ 32`.

```python
class LanguageModelDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        inp, tgt = self.pairs[index]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )

def collate_language_batch(batch):
    inputs, targets = zip(*batch)
    input_ids = pad_sequence(inputs, batch_first=True, padding_value=PAD_IDX)
    target_ids = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    return input_ids, target_ids

train_loader = DataLoader(
    LanguageModelDataset(train_pairs),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_language_batch,
)
val_loader = DataLoader(
    LanguageModelDataset(val_pairs),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_language_batch,
)
```

**Диагностический мини-пакет:**

```python
diag_inputs, diag_targets = next(iter(train_loader))
print("input_ids:", diag_inputs.shape, "target_ids:", diag_targets.shape)
print("не-PAD позиций в target:", (diag_targets != PAD_IDX).sum().item())
```

Формы совпадают и `T ≤ 32`. Заполнение присутствует только справа; число не-PAD целей положительно — именно эти позиции попадут в среднюю кросс-энтропию.

---

## 5. Ручная Elman RNN

Один рекуррентный блок объединяет текущий входной вектор с предыдущим скрытым состоянием, пропускает сумму через tanh и возвращает новое состояние той же формы. Для двух слоёв второй слой получает выход первого на той же позиции времени.

```python
class ElmanRNNUnit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_x = nn.Parameter(torch.empty(input_size, hidden_size))
        self.U_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.U_h)

    def forward(self, x_t, h_previous):
        a = x_t @ self.W_x + h_previous @ self.U_h + self.b_h
        return torch.tanh(a)


class ElmanRNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.units = nn.ModuleList()
        for layer in range(num_layers):
            input_size = EMBEDDING_DIM if layer == 0 else hidden_size
            self.units.append(ElmanRNNUnit(input_size, hidden_size))

    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        x = embeddings
        for unit in self.units:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            outputs = []
            for t in range(seq_len):
                h = unit(x[:, t, :], h)
                outputs.append(h)
            x = torch.stack(outputs, dim=1)
        return x


class ElmanLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_IDX)
        self.rnn = ElmanRNN(HIDDEN_DIM, NUM_LAYERS)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        hidden = self.rnn(emb)
        return self.fc(hidden)
```

**Рекуррентный переход словами.** На каждом временном шаге берётся эмбеддинг текущего индекса `x_t` формы `(B, 128)`, он умножается на входную матрицу `W_x` и складывается с произведением предыдущего скрытого состояния `h_{t-1}` на рекуррентную матрицу `U_h` и смещением `b_h`. Результат проходит через `tanh` и становится новым `h_t`. Один и тот же `W_x`, `U_h`, `b_h` используется на всех позициях — это ключевое свойство RNN.

**Диагностика форм:**

```python
model = ElmanLanguageModel(VOCAB_SIZE).to(DEVICE)
print("Параметров:", sum(p.numel() for p in model.parameters()))

with torch.no_grad():
    emb = model.embedding(diag_inputs.to(DEVICE))
    hidden = model.rnn(emb)
    logits = model.fc(hidden)

print("эмбеддинги:", emb.shape)
print("скрытые:   ", hidden.shape)
print("логиты:    ", logits.shape)
```

Формы: эмбеддинги и скрытые состояния — `(B, T, 128)`, логиты — `(B, T, V)`. Последняя ось логитов — слова словаря, а не время.

---

## 6. Обучение, маскирование PAD, ранняя остановка

Позиции PAD исключаются из кросс-энтропии через `ignore_index=PAD_IDX`. Средняя потеря считается только по реальным целевым токенам; для корректного усреднения суммируются суммы потерь и число реальных токенов по всем мини-пакетам отдельно — иначе батчи с разной длиной давали бы смещённую оценку.

```python
import copy
import torch
from torch.optim import AdamW
from torch.nn import functional as F

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            logits = model(input_ids)
            B, T, V = logits.shape
            loss_sum = F.cross_entropy(
                logits.reshape(-1, V),
                target_ids.reshape(-1),
                ignore_index=PAD_IDX,
                reduction="sum",
            )
            total_loss += loss_sum.item()
            total_tokens += (target_ids != PAD_IDX).sum().item()
    return total_loss / max(total_tokens, 1)


def train_model(model, train_loader, val_loader, optimizer,
                max_epochs=MAX_EPOCHS, patience=PATIENCE):
    history = {"loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids)
            B, T, V = logits.shape
            n_tokens = (target_ids != PAD_IDX).sum().item()

            loss_sum = F.cross_entropy(
                logits.reshape(-1, V),
                target_ids.reshape(-1),
                ignore_index=PAD_IDX,
                reduction="sum",
            )
            loss = loss_sum / n_tokens
            loss.backward()
            optimizer.step()

            total_loss += loss_sum.item()
            total_tokens += n_tokens

        train_loss = total_loss / max(total_tokens, 1)
        val_loss = evaluate_loss(model, val_loader)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Ранняя остановка на эпохе {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


history, best_val_loss = train_model(model, train_loader, val_loader, optimizer)
print(f"Лучшая val_loss = {best_val_loss:.4f}")
```

**Маскирование PAD словами.** Целевой индекс 0 означает техническое дополнение, а не слово. Если бы он участвовал в потере, модель начала бы учиться предсказывать `PAD`, а метрика смещалась бы в сторону коротких последовательностей. `CrossEntropyLoss(ignore_index=0)` вычитает эти позиции из числителя и из знаменателя — средняя потеря считается только по настоящим токенам.

---

## 7. Оценка, перплексия, top-k, генерация

```python
import math
import matplotlib.pyplot as plt

final_val_loss = evaluate_loss(model, val_loader)
perplexity = math.exp(final_val_loss)
print(f"val_loss = {final_val_loss:.4f}")
print(f"perplexity = {perplexity:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(history["loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.xlabel("Эпоха"); plt.ylabel("Cross-entropy")
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

На малом корпусе и простой Elman RNN типичная картина: тренировочная потеря монотонно уменьшается с эпохами, валидационная стабилизируется или слегка растёт после нескольких эпох. Ранняя остановка сохраняет состояние с минимальной `val_loss`.

**Перплексия** — это `exp(val_loss)`, положительное число. Условно её можно читать как «эффективное число равновероятных вариантов», между которыми модель колеблется при выборе следующего слова. Сравнивать её можно только на одном словаре и одной токенизации.

**Top-k прогнозов для короткого запроса:**

```python
def predict_top_k(model, prompt, k=5):
    model.eval()
    tokens = normalize_text(prompt)
    ids = [BOS_IDX] + [stoi.get(w, UNK_IDX) for w in tokens]
    ids = ids[-MAX_SEQUENCE_LENGTH:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(input_ids)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    return [(itos[i.item()], float(p)) for i, p in zip(top_ids, top_probs)]

for prompt in ["князь андрей", "мама мыла", "он сказал что"]:
    print(f"\nЗапрос: {prompt!r}")
    for word, prob in predict_top_k(model, prompt, k=5):
        print(f"  {word:<15} {prob:.4f}")
```

Логиты последней позиции сначала преобразуются в распределение softmax, затем сортируются по вероятности. В top-k попадают слова из `itos` — это проверка, что модель действительно работает со словарём, а не с чем-то посторонним.

**Жадная генерация:**

```python
def generate_greedy(model, prompt, max_new_tokens=20):
    model.eval()
    tokens = normalize_text(prompt)
    ids = [BOS_IDX] + [stoi.get(w, UNK_IDX) for w in tokens]
    for _ in range(max_new_tokens):
        context = ids[-MAX_SEQUENCE_LENGTH:]
        input_ids = torch.tensor([context], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = model(input_ids)
        next_id = int(logits[0, -1, :].argmax().item())
        if next_id == EOS_IDX:
            break
        ids.append(next_id)
    return " ".join(itos.get(i, "<unk>") for i in ids[1:])

print("\nЖадная генерация:", generate_greedy(model, "князь андрей", 20))
```

Генерация останавливается либо по `EOS`, либо по достижении лимита `max_new_tokens`. Жадная стратегия на маленькой RNN склонна зацикливаться на самых частых продолжениях.

**Интерпретация ограничений.** Корпус мал, модель — двухслойная Elman RNN на CPU, обучение короткое. Топ-k и генерация демонстрируют техническую корректность: модель различает частые продолжения, распределение имеет сумму 1, генерация конечна. Содержательного текста от такой модели ожидать нельзя — она лишь показывает, что данные проходят весь путь «строка → индекс → эмбеддинг → скрытое состояние → логиты → распределение».

---

## 9. Ответы на контрольные вопросы

**1. Почему индекс токена нельзя подавать в RNN как обычное вещественное число?**  
Индекс — это адрес строки в словаре, а не признак и не мера смысла. RNN ожидает вещественные входы и арифметически смешивает их; «41 не больше 7» в смысле слова, оба значения лишь указывают на разные обучаемые векторы в таблице эмбеддингов.

**2. Почему словарь строится только по обучающим документам?**  
Иначе частоты слов из валидации попадут в счётчик — это утечка данных. Валидация должна проверять обобщение на словах, которые модель не видела при построении словаря.

**3. Какие пары образуются из (BOS, a, b, EOS)?**  
Одна пара: вход `(BOS, a, b)`, цель `(a, b, EOS)`. Цель сдвинута на один токен влево.

**4. Чем PAD отличается от UNK и почему нельзя игнорировать одинаково?**  
PAD — техническое дополнение короткой последовательности: у него нет настоящей цели, его нужно исключать из кросс-энтропии (`ignore_index=0`). UNK — реальный, но отсутствующий в словаре токен, он должен участвовать в потере как обычная цель.

**5. Форма логитов до разворачивания и почему последняя ось V?**  
`(B, T, V)`. На каждой позиции `(b, t)` — независимое распределение по всем `V` словам словаря; последняя ось перебирает классы-слова, а не моменты времени.

**6. Почему целевые индексы подходят для CrossEntropyLoss, хотя это не метки класса документа?**  
На каждой позиции решается задача классификации на `V` классов — предсказание следующего слова. CrossEntropyLoss работает с любой такой классификацией: логиты формы `(N, V)` и целочисленные метки формы `(N)`.

**7. Что изменится, если включить PAD в усреднение потери?**  
Модель начнёт учиться предсказывать технический нулевой индекс. Метрика станет смещённой: короткие последовательности получат завышенный вес, длинные — заниженный. Сравнение батчей с разной длиной станет некорректным.

**8. Что обновляет optimizer.step(), а что не является параметром?**  
Обновляются только `nn.Parameter`: `W_x`, `U_h`, `b_h` каждого слоя, таблица эмбеддингов (кроме строки PAD), веса и смещение `fc`. Индексы токенов, PAD-маска, формы тензоров, размеры словаря — не параметры и не обновляются.

**9. Почему для сравнения перплексии нужен один словарь и одна токенизация?**  
Перплексия — это `exp(средней кросс-энтропии)`. Если `V`, разбиение на слова или нормализация разные, то и пространство классов разное. Значения становятся несравнимыми, даже если сравнивают одну и ту же архитектуру.

**10. Как по форме (B, T, 128) отличить оси?**  
Первая ось — мини-пакет (фрагменты текста), вторая — временные позиции (длина ≤ 32), третья — скрытые признаки (128). Это зафиксировано конструкцией: `input_ids` имеет форму `(B, T)`, эмбеддинг даёт `(B, T, 128)`, RNN сохраняет `(B, T, 128)`, линейный слой меняет последнюю ось на `V`.

