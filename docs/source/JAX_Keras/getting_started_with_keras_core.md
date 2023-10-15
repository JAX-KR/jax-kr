# 케라스 코어 시작하기 (Getting started with Keras Core)
**저자:** [fchollet](https://twitter.com/fchollet)<br>
**역자:** [조용은](https://www.github.com/gdakate)<br>
**생성 날짜:** 2023/07/10<br>
**마지막 수정:** 2023/07/10<br>
**번역 일자:** 2023/10/15<br>
**설명:** 새로운 멀티 백엔드 케라스와의 첫 만남.

#소개 (Introduction)
케라스 코어는 TensorFlow, JAX, 그리고 PyTorch와 상호교환할 수 있게 작동하는 Keras API의 완전한 구현입니다. 이 노트북은 여러분에게 케라스 코어의 주요 작업 흐름을 안내할 것입니다.

먼저, 케라스 코어를 설치해봅시다:


```python
!pip install -q keras-core
```

#설정 (Setup)
우리는 여기에서 JAX 백엔드를 사용할 것입니다 -- 하지만 아래의 문자열을 `"tensorflow"` 또는 `"torch"`로 수정하고 "Restart runtime"을 누르면, 전체 노트북은 동일하게 실행될 것입니다! 이 전체 가이드는 백엔드에 구애받지 않습니다.


```python
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

# 백엔드가 구성된 후에만 keras_core를 가져와야 합니다.
# 패키지를 가져온 후에는 백엔드를 변경할 수 없습니다.

import keras_core as keras
```

#첫 번째 예제: MNIST 컨볼루션 네트워크 (A first example: A MNIST convnet)
ML의 Hello World부터 시작해보겠습니다: 컨볼루션 네트워크을 훈련시켜 MNIST 숫자를 분류합니다.

다음은 데이터입니다:


```python
# 데이터를 로드하고 훈련 세트와 테스트 세트로 나눕니다
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 이미지를 [0, 1] 범위로 스케일링합니다
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# 이미지가 (28, 28, 1)의 형태를 가지고 있는지 확인합니다
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
```

다음은 우리의 모델입니다.

케라스가 제공하는 다양한 모델 빌드 옵션은 다음과 같습니다:
- [The Sequential API](https://keras.io/keras_core/guides/sequential_model/) (아래에서 사용하는 방법)
- [The Functional API](https://keras.io/keras_core/guides/functional_api/) (가장 일반적으로 사용)
- [Writing your own models yourself via subclassing](https://keras.io/keras_core/guides/making_new_layers_and_models_via_subclassing/) (고급 사용 사례를 위한)


```python
# 모델 매개변수들
num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
```

다음은 우리 모델의 요약입니다:


```python
model.summary()
```

`compile()` 메서드를 사용하여 옵티마이저, 손실 함수 및 모니터링할 평가지표들을 지정합니다. JAX 및 TensorFlow 백엔드에서 XLA 컴파일이 기본적으로 활성화되어 있음을 주의하세요.







```python
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
```

모델을 학습하고 평가해 봅시다.
훈련 중에 보지 않은 데이터에 대한 일반화를 모니터링하기 위해 데이터의 15%를 검증 분할로 따로 떼어 놓겠습니다.


```python
batch_size = 128
epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
score = model.evaluate(x_test, y_test, verbose=0)
```

훈련 중에 매 에폭의 끝에서 모델을 저장했습니다. 다음과 같이 모델을 가장 최근 상태로도 저장할 수 있습니다:


```python
model.save("final_model.keras")
```

그리고 다음과 같이 다시 불러올 수 있습니다.


```python
model = keras.saving.load_model("final_model.keras")
```

다음으로, `predict()`를 사용하여 클래스 확률의 예측을 조회할 수 있습니다:


```python
predictions = model.predict(x_test)
```

기본 사항은 이것으로 끝입니다!

# 크로스-프레임워크 커스텀 컴포넌트 작성(Writing cross-framework custom components)
케라스 코어는 동일한 코드베이스로 TensorFlow, JAX, PyTorch에서 작동하는 커스텀 레이어, 모델, 평가지표, 손실함수, 옵티마이저를 작성할 수 있게 해줍니다. 먼저 커스텀 레이어를 살펴봅시다.

`tf.keras`에서 커스텀 레이어를 작성하는 방법을 이미 알고 있다면 — 좋습니다, 아무것도
수정할 필요 없습니다. 하나만 제외하고: `tf` 네임스페이스에서 함수를 사용하는 대신 `keras.ops.*`에서 함수를 사용해야 합니다.

`keras.ops` 네임스페이스는 다음을 포함합니다.:

- NumPy API의 구현, 예를 들어 `keras.ops.stack` 또는 `keras.ops.matmul.`
- NumPy에 없는 신경망 특정 작업의 집합, 예를 들어 `keras.ops.conv` 또는 `keras.ops.binary_crossentropy.`

모든 백엔드에서 작동하는 커스텀 `Dense` 레이어를 만들어 봅시다:







```python

class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, name=None):
        super().__init__(name=name)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=keras.initializers.GlorotNormal(),
            name="kernel",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer=keras.initializers.Zeros(),
            name="bias",
            trainable=True,
        )

    def call(self, inputs):
        # 케라스 연산을 사용하여 백엔드에 구애받지 않는 레이어/평가지표/등을 생성합니다.
        x = keras.ops.matmul(inputs, self.w) + self.b
        return self.activation(x)

```


다음으로, `keras.random` 네임스페이스에 의존하는 커스텀 `Dropout` 레이어를 만들어 봅시다:


```python

class MyDropout(keras.layers.Layer):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        self.rate = rate
        # RNG 상태 관리를 위해 seed_generator를 사용합니다.
        # 이것은 상태 요소이며 seed 변수는
        # `layer.variables`의 일부로 추적됩니다.
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        # 랜덤 연산을 위해 keras_core.random을 사용합니다.
        return keras.random.dropout(inputs, self.rate, seed=self.seed_generator)

```

다음으로, 두 커스텀 레이어를 사용하는 커스텀 서브클래스 모델을 작성해봅시다:


```python

class MyModel(keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_base = keras.Sequential(
            [
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                keras.layers.GlobalAveragePooling2D(),
            ]
        )
        self.dp = MyDropout(0.5)
        self.dense = MyDense(num_classes, activation="softmax")

    def call(self, x):
        x = self.conv_base(x)
        x = self.dp(x)
        return self.dense(x)

```


이를 컴파일하고 학습시켜봅시다:


```python
model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=1,  # 빠르게 처리하기 위해
    validation_split=0.15,
)
```

# 임의의 데이터 소스에서 모델 학습하기 (Training models on arbitrary data sources)
모든 케라스 모델은 사용하는 백엔드에 상관없이 다양한 데이터 소스에서 훈련 및 평가할 수 있습니다. 이에는 다음이 포함됩니다:

- NumPy 배열
- Pandas 데이터프레임
- TensorFlow `tf.data.Dataset` 객체
- PyTorch `DataLoader` 객체
- Keras `PyDataset` 객체

이들은 TensorFlow, JAX, 또는 PyTorch를 케라스 백엔드로 사용하든 상관없이 모두 작동합니다.

PyTorch `DataLoaders`로 시도해봅시다:


```python
import torch

# TensorDataset 생성
train_torch_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train), torch.from_numpy(y_train)
)
val_torch_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_test), torch.from_numpy(y_test)
)

# DataLoader 생성
train_dataloader = torch.utils.data.DataLoader(
    train_torch_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_torch_dataset, batch_size=batch_size, shuffle=False
)

model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
model.fit(train_dataloader, epochs=1, validation_data=val_dataloader)

```

이제 `tf.data`로도 시도해 봅시다:


```python
import tensorflow as tf

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
model.fit(train_dataset, epochs=1, validation_data=test_dataset)
```

## 추가 학습 자료 (Further reading)
이것으로 케라스 코어의 새로운 멀티 백엔드 기능에 대한 짧은 개요를 마칩니다. 다음으로 이것들을 배울 수 있을 것입니다 :

### `fit()`에서 일어나는 것을 어떻게 커스터마이징할까요?
비표준 훈련 알고리즘을 직접 구현하려고 하지만 (예: GAN 훈련 루틴) 여전히 `fit()`의 힘과 사용성을 누리고 싶다면, `fit()`을 임의의 사용 사례를 지원하도록 커스터마이징하는 것이 정말 쉽습니다.

- [TensorFlow에서 `fit()`에서 일어나는 것을 커스터마이징하는 방법](http://keras.io/keras_core/guides/custom_train_step_in_tensorflow/)
-[JAX에서 `fit()`에서 일어나는 것을 커스터마이징하는 방법](http://keras.io/keras_core/guides/custom_train_step_in_jax/)
-[PyTorch에서 `fit()`에서 일어나는 것을 커스터마이징하는 방법](http://keras.io/keras_core/guides/custom_train_step_in_pytorch/)

## 커스텀 훈련 루프 작성 방법 (How to write custom training loops)
- [TensorFlow에서 처음부터 훈련 루프 작성하는 방법](http://keras.io/keras_core/guides/writing_a_custom_training_loop_in_tensorflow/)
-[JAX에서 처음부터 훈련 루프 작성하는 방법](http://keras.io/keras_core/guides/writing_a_custom_training_loop_in_jax/)
-[PyTorch에서 처음부터 훈련 루프 작성하는 방법](http://keras.io/keras_core/guides/writing_a_custom_training_loop_in_torch/)

## 훈련 분배 방법 (How to distribute training)
- [TensorFlow와 함께 분산 훈련 가이드](http://keras.io/keras_core/guides/distributed_training_with_tensorflow/)
- [JAX 분산 훈련 예제](https://github.com/keras-team/keras-core/blob/main/examples/demo_jax_distributed.py)
- [PyTorch 분산 훈련 예제](https://github.com/keras-team/keras-core/blob/main/examples/demo_torch_multi_gpu.py)

이 라이브러리를 즐기세요! 🚀
