# Машинное обучение. Лабораторная работа №3
## Задание 1
*Реализуйте нейронную сеть с двумя сверточными слоями, и одним полносвязным с нейронами с кусочно-линейной функцией активации. Какова точность построенное модели?*

Модель нейронной сети:
```python
conv_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu',
        input_shape=(28, 28, 1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
])
```

Для борьбы с переобучением использовались регуляризация L2 и случайное "отключение" части нейронов в слоях на каждом этапе обучения (Dropout).

Также во время обучения модели иcпользовался один из популярных оптимизаторов `Adam` с динамически изменяемой скоростью обучения.

Использовались два варианта динамического изменения learning rate:
* С помощью `ExponentialDecay` (экспоненциальное затухание), где через `DECAY_STEPS` шагов (20000) `learning_rate` уменьшается на `1 - DECAY_RATE` (10%). 
```python
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=True
)
```
* С помощью `callback`'а `ReduceLROnPlateau`, который автоматически уменьшает `learning_rate` в 10 раз, когда показатель потерь на валидационном наборе данных перестает улучшаться в течение 6 эпох. 
```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=6,
    verbose=1,
    min_lr=MIN_LEARNING_RATE
)
```

Вышеперечисленные техники использовались и для последующих заданий.

**Точность данной модели на тестовой выборке составила: 91,1%.**

## Задание 2
*Замените сверточные слои на слои, реализующие операцию пулинга (Pooling) с функцией максимума или среднего. Как это повлияло на точность классификатора?*

Модель нейронной сети:
```python
pooling_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.MaxPooling2D((2, 2), input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
])
```

**Точность данной модели на тестовой выборке составила: 81,6%.**

Точность данной модели по сравнению с предыдущей намного меньше, так как модель, состоящая только из pooling-слоёв не способна извлекать признаки из изображения и обрабатывать их, а только уменьшать размерность данных. 

## Задание 3
*Реализуйте классическую архитектуру сверточных сетей LeNet-5*

Модель нейронной сети:
```python
lenet_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(
        6, (5, 5), activation='relu',
        input_shape=(28, 28, 1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001)
    ),
    tf.keras.layers.AveragePooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.AveragePooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
])
```

**Точность данной модели на тестовой выборке составила: 89,7%.**

Кривые обучения вышеперечисленных моделей:
![Learning curves](https://user-images.githubusercontent.com/55394253/228273222-3a340a17-1ca5-47af-b124-658b323b890f.png)

Максимальное значение точности было получено с помощью первой модели, минимальное значение функции потерь - с помощью LeNet-5.

## Задание 4
*Сравните максимальные точности моделей, построенных в лабораторных работах 1-3. Как можно объяснить полученные различия?*

**Логистическая модель (из ЛР 1):** точность модели, обученной на выборке из 200000 изображений - **82%**.
![Learning curve](https://user-images.githubusercontent.com/55394253/228247374-1c15c192-b1fc-4b16-ac42-1f2be8dabdcd.png)

**Полносвязная нейронная сеть (из ЛР 2):** точность модели, обученной на выборке из 200000 изображений в течение 50 эпох - **88,3%**.
```python
simple_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
])
```

**Полносвязная нейронная сеть с регуляризацией и методом сброса нейронов (из ЛР 2):** точность модели - **89,3%**.
```python
regularized_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
])
```

**Полносвязная нейронная сеть с динамически изменяемой скоростью обучения (из ЛР 2):** точность модели - **88,7%**.

Кривые обучения вышеперечисленных моделей из ЛР 2:
![Learning curves](https://user-images.githubusercontent.com/55394253/228275370-32031807-d64c-4cd8-80c1-16e42e5eb74a.png)

Свёрточные нейронные сети показали большую точность и меньшее значение функции потерь по сравнению с другими моделями.

Если сравнивать нейронные сети с логистической регрессией, то основное преимущество нейронной сети заключается в ее способности к извлечению высокоуровневых признаков из данных и способности к нелинейному моделированию сложных зависимостей в данных. С другой стороны, логистическая регрессия является более простой и понятной моделью, особенно если данные имеют простую структуру и зависимости в данных линейные.

При сравнении свёрточной нейронной с полносвязной нейронной сетью, свёрточные нейронные сети имеют несколько свойств, которые делают их более эффективными, чем полносвязные нейронные сети:
* CNN способны автоматически извлекать иерархию признаков из изображений. Они могут обнаруживать простые признаки, такие как края и текстуры, а затем использовать их для обнаружения более сложных признаков, таких как объекты.
* CNN имеют меньшее количество параметров, чем полносвязные нейронные сети, что делает их более эффективными для обучения на больших наборах данных. Сверточные нейронные сети используют общие веса для свертки, что позволяет им обобщать изображения лучше, чем полносвязные нейронные сети.
* CNN используют pooling-слои, которые позволяют уменьшать размерность данных, сохраняя важную информацию о признаках. Это помогает предотвратить переобучение и снизить риск обработки шума.

В целом, свёрточные нейронные сети обычно показывают более высокую точность, чем полносвязные нейронные сети, при работе с изображениями и другими типами данных, такими как аудио и видео.
