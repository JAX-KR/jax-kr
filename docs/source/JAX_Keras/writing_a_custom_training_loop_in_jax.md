# JAX로 스크래치부터 학습 루프 만들어보기 (Writing a training loop from scratch in JAX)

**저자:** [fchollet](https://twitter.com/fchollet)<br>
**역자:** [Junghyun Park](https://github.com/parkjh688)<br>
**생성 날짜:** 2023/06/25<br>
**마지막 수정 날짜:** 2023/06/25<br>
**설명:** JAX로 낮은 레벨의 학습 & 평가 루프 코드 만들기<br>
**번역 일자:** 2023/10/15

## 셋업하기 (Setup)


```python
import os

# 이 가이드에서는 jax backend로만 실행합니다.
os.environ["KERAS_BACKEND"] = "jax"

import jax

# tf.data를 사용하기 위해서 TF를 임포트합니다.
import tensorflow as tf
import keras_core as keras
import numpy as np
```

## 소개 (Introduction)

케라스는 기본적으로 학습 및 평가 루프를 제공합니다, `fit()`과 `evaluate()` 입니다. 두 가 지에 대한 사용 방법은 가이드인 [Training & evaluation with the built-in methods](/keras_core/guides/training_with_built_in_methods/)를 참고하면 됩니다.


만약 모델의 학습 알고리즘을 커스터마이즈 하고 싶을 때, `fit()`의 편리성은(예를 들어, GAN에서 `fit()`을 사용한다면), `Model` 클래스의 서브 클래스에 나만의 `train_step()` 메소드를 구현하면 된다는 것에 있습니다.
`train_step()` 메소드는 `fit()`이 반복적으로 호출할 것이기 때문입니다.


이제, 만약 굉장히 낮은 레벨의 컨트롤을 학습과 평가 과정에서 하고 싶다면, 나만의 학습 및 평가 루프 코드를 스크래치부터 만들어야 합니다. 이 가이드가 바로 그 내용에 관련한 것입니다.

## 엔드-투-엔드 예제 (A first end-to-end example)

커스텀 학습 루프를 만들기 위해서, 아래의 재료가 필요합니다:
- 당연하게도, 학습할 모델.
- 옵티마이저. `keras_core.optimizers`에서 가져다 쓸 수 있고, `optax` 패키지에서도 사용할 수 있습니다.
- 손실 함수.
- 데이터 셋. JAX 에코시스템이 사용하는 데이터 로드 방식은 `tf.data`를 사용하는 것이기 떄문에 이것을 사용할 것 입니다.


차례대로 보겠습니다.

가장 먼저, 모델과 MNIST 데이터셋을 봅시다.


```python

def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model()

# 학습 데이터셋을 준비합니다.
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)).astype("float32")
x_test = np.reshape(x_test, (-1, 784)).astype("float32")
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# 10,000개의 샘플을 검증(validation)용으로 사용합니다.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 학습 데이터 준비.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# 검증 데이터 준비.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
```

다음은, 손실 함수와 옵티마이저입니다. 케라스 옵티마이저를 사용할 것 입니다.


```python
# 손실함수를 준비합니다.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# 옵티마이저를 준비합니다.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
```

### JAX에서 그라디언트 가져오기 (Getting gradients in JAX)

커스텀 학습 루프에서 미니 배치 그라디언트를 이용해 모델을 학습해봅시다.

JAX에서는, 그라디언트는 *메타프로그래밍(metaprogramming)*에 의해 계산됩니다: 첫 번째 함수에 대한 그라디언트 계산 함수를 생성하기 위해 함수에서`jax.grad`(혹은 `jax.value_and_grad`)를 호출합니다.

그래서 제일 먼저 손실값을 반환할 함수가 필요합니다. 이 함수는 그라디언트를 만들어낼 함수입니다. 아래와 같을 것입니다:
```python
def compute_loss(x, y):
    ...
    return loss
```

이런 함수를 가지고 있다면, 메타프로그래밍로 그라디언트를 계산할 수 있습니다. 다음과 같이 말이죠
```python
grad_fn = jax.grad(compute_loss)
grads = grad_fn(x, y)
```

일반적으로, 그라디언트값 말고도 손실값도 원할 것입니다. `jax.grad` 대신 `jax.value_and_grad`를 사용하면 원하는 대로 손실값과 그라디언트 모두를 얻을 수 있습니다:
```python
grad_fn = jax.value_and_grad(compute_loss)
loss, grads = grad_fn(x, y)
```

### JAX 계산은 순수한 무상태 (JAX computation is purely stateless)

JAX는, 모든 것이 무상태 함수여야 합니다 -- 따라서 손실을 계산하는 함수도 무상태여야 합니다. 무상태는 모든 케라스 변수들(예를 들어 가중치 텐서들)은 함수 입력으로 전달되어야 하며, 포워드 패스 중에 업데이트된 모든 변수는 함수 출력으로 반환되어야 합니다.


순방향 패스 중에, Keras 모델의 훈련 불가능한 변수가 업데이트될 수 있습니다. 이러한 변수는 예를 들어 RNG 시드 상태 변수나 BatchNormalization 통계 정보일 수 있습니다. 이러한 변수들을 반환해야 할 필요가 있습니다. 그래서 다음과 같은 것이 필요합니다:
```python
def compute_loss_and_updates(trainable_variables, non_trainable_variables, x, y):
    ...
    return loss, non_trainable_variables
```

이러한 함수를 갖고 있다면, `value_and_grad`에서 `hax_aux`를 지정하여 그라디언트 함수를 얻을 수 있습니다. 이것은 JAX에게 손실 계산 함수가 손실 이외의 출력값을 반환한다는 것을 알려줍니다. 또한 주의해야 할 점은 손실은 항상 첫 번째 출력값이어야 한다는 것입니다.
```python
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
(loss, non_trainable_variables), grads = grad_fn(
    trainable_variables, non_trainable_variables, x, y
)
```

이제 기본 적인 것들을 해결했으니,`compute_loss_and_updates`함수를 구현해봅시다. Keras 모델에는 `stateless_call` 메소드가 있는데, 이 메소드가 유용할 것입니다. 이 메소드는 `model.__call__`처럼 동작하는데, 그러나 이는 모델 내의 모든 변수의 값을 명시적으로 전달하도록 요구하며, `__call__`의 출력뿐만 아니라 (잠재적으로 업데이트된) 훈련되지 않는 변수도 반환합니다.


```python

def compute_loss_and_updates(trainable_variables, non_trainable_variables, x, y):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    return loss, non_trainable_variables

```

Let's get the gradient function:


```python
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
```

### 학습 스텝 함수 (The training step function)


다음은, 엔드-투-엔트 학습 스텝을 구현하는 것입니다. 이 함수는 순방향 패스를 실행하고 손실을 계산하며 그래디언트를 계산하는 것 뿐만 아니라, 옵티마이저를 사용하여 훈련 가능한 변수를 업데이트하는 것도 해야합니다. 또한 이 함수는 상태를 가지지 않아야 하므로, 사용할 모든 상태 요소를 포함하는 `state` 튜플을 입력으로 받을 것입니다:

- `trainable_variables`과 `non_trainable_variables`: 모델의 변수
- `optimizer_variables`: 옵티마이저의 상태 변수, 예를 들면 모멘텀 어큐멀레이터(momentum accumulators)와 같은 것들  



학습 가능한 변수들을 업데이트 하기 위해, 옵티마이저의 무상태 (stateless) 메소드인 `stateless_apply`를 사용합니다. 이 메소드는 `optimizer.apply()`와 동일한 역할을 수행하지만 항상 `trainable_variables`와 `optimizer_variables`를 전달해야 합니다. 이 메소드는 업데이트된 학습 가능한 변수와 업데이트된 optimizer_variables를 모두 반환합니다.


```python

def train_step(state, data):
    trainable_variables, non_trainable_variables, optimizer_variables = state
    x, y = data
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        grads, trainable_variables, optimizer_variables
    )
    # 업데이트된 상태 반환
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )

```

### `jax.jit`으로 더 빠르게 만들기 (Make it fast with `jax.jit`)


기본적으로, JAX 작업은 즉시 실행(eagerly)됩니다. 이는 TensorFlow의 eager 모드나 PyTorch의 eager 모드와 마찬가지로 동작합니다. 그리고 마찬가지로 TensorFlow나 PyTorch의 eager 모드처럼, 꽤 느립니다 -- eager 모드는 실제 작업을 수행하는 방법으로 사용하기보다 디버깅 환경으로서 더 적합합니다. 따라서 train_step 함수를 빠르게 만들기 위해 컴파일해보겠습니다.


상태가 없는 JAX 함수를 가지고 있다면, `@jax.jit` 데코레이터를 이용해 XLA로 그 함수를 컴파일 할 수 있습니다. . 이 함수는 첫 번째 실행 시 추적(traced)되며, 이후의 실행에서는 추적된 그래프를 실행하게 됩니다 (이것은 @tf.function(jit_compile=True)와 유사합니다).


한 번 해볼까요:


```python

@jax.jit
def train_step(state, data):
    trainable_variables, non_trainable_variables, optimizer_variables = state
    x, y = data
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # 업데이트된 상태 반환
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )

```

이제 모델을 학습할 준비가 되었습니다. 훈련 루프 자체는 간단합니다: 단순히 loss, state = train_step(state, data)를 반복적으로 호출합니다.

노트:
- tf.data.Dataset에서 제공하는 TF 텐서를 JAX 함수로 전달하기 전에 NumPy로 변환합니다.
- 모든 변수는 미리 빌드되어야 합니다. 모델과 옵티마이저도 미리 빌드 되어있어야 합니다. Functional API 모델은 이미 빌드되어 있지만, 만약 서브클래스 모델이라면 데이터 배치로 모델을 빌드해야 합니다.


```python
# 옵티마이저 변수 빌드하기
optimizer.build(model.trainable_variables)

trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
state = trainable_variables, non_trainable_variables, optimizer_variables

# 학습 루프
for step, data in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = train_step(state, data)
    # 100 배치마다 로깅
    if step % 100 == 0:
        print(f"Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")
```

여기서 주목해야 할 중요한 점은 이 루프가 완전히 상태를 가지지 않는다는 것입니다 -- 모델에 연결된 변수인 (`model.weights`)는 루프 중에 업데이트되지 않습니다. 그들의 새로운 값은 오직 `state` 튜플에 저장됩니다. 따라서 모델을 저장하기 전에 어느 시점에서든 새로운 변수 값을 모델에 다시 연결시켜 주어야 합니다.

모델을 업데이트하려면 각 모델 변수에 대해 `variable.assign(new_value)`를 호출하면 됩니다:


```python
trainable_variables, non_trainable_variables, optimizer_variables = state
for variable, value in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for variable, value in zip(model.non_trainable_variables, non_trainable_variables):
    variable.assign(value)
```

## 낮은 레벨의 메트릭 사용하기 (Low-level handling of metrics)

이 기본적인 학습 루프에 메트릭 모니터링을 추가해봅시다.

스크래치부터 작성한 학습한 루프에서는 내장 Keras 메트릭(또는 직접 작성한 사용자 정의 메트릭)을 쉽게 재사용할 수 있습니다. 이러한 방법으로요:

- 루프 시작 부분에서 메트릭을 인스턴스화합니다
- `train_step` 인수 및 `compute_loss_and_updates` 인수에 `metric_variables`를 포함합니다.
- `compute_loss_and_updates` 함수에서 `metric.stateless_update_state()`를 호출합니다. 이것은 `update_state()`와 동일하지만 상태를 가지지 않습니다.
- 메트릭의 현재 값을 표시해야 하는 경우, `train_step` 외부(eager scope 내에서), 새 메트릭 변수 값을 메트릭 객체에 연결하고 `metric.result()`를 호출합니다.
- 메트릭의 상태를 지우려면(일반적으로 에폭 끝에), `metric.reset_state()`를 호출합니다.


이런 지식을 사용해 학습 종료 시 학습 및 검증 데이터에 대한 `CategoricalAccuracy`를 계산해봅시다.


```python
# Get a fresh model
model = get_model()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()


def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, metric_variables, x, y
):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metric_variables = train_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, (non_trainable_variables, metric_variables)


grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)


@jax.jit
def train_step(state, data):
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    ) = state
    x, y = data
    (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(
        trainable_variables, non_trainable_variables, metric_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # 업데이트된 상태 반환
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    )

```

다음과 같이 평가 스텝 함수도 준비해야합니다:


```python

@jax.jit
def eval_step(state, data):
    trainable_variables, non_trainable_variables, metric_variables = state
    x, y = data
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metric_variables = val_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, (
        trainable_variables,
        non_trainable_variables,
        metric_variables,
    )

```

지금껏 만들어온 루프입니다:


```python
# 옵티마이저 변수 빌드
optimizer.build(model.trainable_variables)

trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
metric_variables = train_acc_metric.variables
state = (
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metric_variables,
)

# 학습 루프
for step, data in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = train_step(state, data)
    # 100 배치마다 로깅
    if step % 100 == 0:
        print(f"Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
        _, _, _, metric_variables = state
        for variable, value in zip(train_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f"Training accuracy: {train_acc_metric.result()}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")

metric_variables = val_acc_metric.variables
(
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metric_variables,
) = state
state = trainable_variables, non_trainable_variables, metric_variables

# 평가 루프
for step, data in enumerate(val_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = eval_step(state, data)
    # 100 배치마다 로깅
    if step % 100 == 0:
        print(f"Validation loss (for 1 batch) at step {step}: {float(loss):.4f}")
        _, _, metric_variables = state
        for variable, value in zip(val_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f"Validation accuracy: {val_acc_metric.result()}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")
```

## 모델이 추적하는 손실의 낮은 레벨 핸들링 (Low-level handling of losses tracked by the model)


레이어 및 모델은 순방향 패스 중에 `self.add_loss(value)`를 호출하는 레이어에 의해 생성된 모든 손실을 재귀적으로 추적합니다. 결과적으로 나오는 스칼라 손실 값 리스트는 순방향 패스 종료 시에 `model.losses` 속성을 통해 사용할 수 있습니다.

이러한 손실 구성요소를 사용하려면 이들을 합산하고 학습 단계에서 최종 손실에 추가해야 합니다.

다음은 activity regularization 손실을 생성하는 레이어를 살펴봅시다:


```python

class ActivityRegularizationLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * jax.numpy.sum(inputs))
        return inputs

```

Activity regularization을 사용하는 굉장히 간단한 모델을 빌드해봅시다:


```python
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu")(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

이제 compute_loss_and_updates 함수는 다음과 같아야 합니다:

- model.stateless_call()에 return_losses=True를 전달합니다
- 결과로 나오는 losses를 합산하고 최종 손실에 추가합니다


```python

def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, metric_variables, x, y
):
    y_pred, non_trainable_variables, losses = model.stateless_call(
        trainable_variables, non_trainable_variables, x, return_losses=True
    )
    loss = loss_fn(y, y_pred)
    if losses:
        loss += jax.numpy.sum(losses)
    metric_variables = train_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, non_trainable_variables, metric_variables

```

끝입니다!
