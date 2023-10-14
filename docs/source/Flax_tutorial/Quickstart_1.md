# Quick Start

번역: 김한빈, 박정현

Flax에 오신 것을 환영합니다!

Flax는 JAX 위에 구축된 오픈 소스 Python 신경망 라이브러리입니다. 이 튜토리얼은 Flax Linen API를 사용하여 간단한 합성곱 신경망(CNN)을 구축하고 MNIST 데이터셋에서 이미지 분류를 위해 해당 신경망을 훈련하는 방법을 보여줍니다.

# 1. Flax 설치하기


```python
!pip install -q flax
```

# 2. 데이터 로드하기

Flax는 어떤 데이터 로딩 파이프라인이든 사용할 수 있으며, 이 예제에서는 TFDS를 활용하는 방법을 보여줍니다. MNIST 데이터셋을 로드하고 준비하는 함수를 정의하고, 샘플을 부동 소수점 숫자로 변환하는 함수입니다.


```python
import tensorflow_datasets as tfds  # TFDS for MNIST
import tensorflow as tf             # TensorFlow operations

def get_datasets(num_epochs, batch_size):
  """Load MNIST train and test datasets into memory."""
  train_ds = tfds.load('mnist', split='train')
  test_ds = tfds.load('mnist', split='test')

  train_ds = train_ds.map(lambda sample: {'image': tf.cast(sample['image'],
                                                           tf.float32) / 255.,
                                          'label': sample['label']}) # normalize train set
  test_ds = test_ds.map(lambda sample: {'image': tf.cast(sample['image'],
                                                         tf.float32) / 255.,
                                        'label': sample['label']}) # normalize test set

  train_ds = train_ds.repeat(num_epochs).shuffle(1024) # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
  train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1) # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
  test_ds = test_ds.shuffle(1024) # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
  test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1) # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency

  return train_ds, test_ds
```

#3. 네트워크 정의하기

Flax Linen API를 사용하여 Flax Module을 서브클래싱하여 합성곱 신경망을 생성합니다. 이 예제에서 사용하는 아키텍처는 비교적 간단하므로-레이어를 단순히 쌓는 것- __call__ 메소드 내에서 인라인 서브모듈을 직접 정의하고 @compact 데코레이터로 감싸는 방식으로 구현할 수 있습니다. Flax Linen @compact 데코레이터에 대해 자세히 알아보려면 "Setup vs Compact 가이드"를 참조하시기 바랍니다.


```python
from flax import linen as nn  # Linen API

class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x

```

## View model layers

Flax Module의 인스턴스를 생성하고, Module.tabulate 메서드를 사용하여 모델 레이어의 테이블을 시각화합니다. 이를 위해 RNG 키와 템플릿 이미지 입력을 전달합니다.


```python
import jax
import jax.numpy as jnp  # JAX NumPy

cnn = CNN()
print(cnn.tabulate(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1))))
```

# 4. Create a TrainState

Flax에서 일반적인 패턴은 step number, parameters,
optimizer state를 포함한 전체 훈련 상태를 나타내는 단일 데이터 클래스를 생성하는 것입니다.

이러한 패턴은 매우 일반적이므로, Flax는 대부분의 기본 사용 사례를 지원하는 flax.training.train_state.TrainState 클래스를 제공합니다.


```python
!pip install -q clu
```


```python
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers
```

메트릭을 계산하기 위해 clu 라이브러리를 사용할 것입니다. clu에 대한 자세한 내용은 레포지토리와 노트북을 참조하세요.


```python
@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

```

그런 다음 metrics를 포함하는 train_state.TrainState의 서브클래스를 작성하여야 합니다. 이렇게 하면 train_step()(조금 더 아래에 코드가 있습니다)과 같은 함수에 단일 인수를 전달하여 손실을 계산하고 매개변수를 업데이트하며 동시에 메트릭을 계산할 수 있는 이점이 있습니다.


```python
class TrainState(train_state.TrainState):
  metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.ones([1, 28, 28, 1]))['params'] # initialize parameters by passing a template image
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())
```

# 5. Training step

아래와 같은 기능을 수행하는 함수입니다:

* TrainState.apply_fn (Module.apply 메소드(forward pass)를 포함하는)을 사용하여 매개변수와 일괄적인 입력 이미지로 신경망을 평가합니다.
* 미리 정의된 optax.softmax_cross_entropy_with_integer_labels()를 사용하여 교차 엔트로피 손실을 계산합니다. 이 함수는 정수 레이블을 예상하므로 레이블을 원핫 인코딩으로 변환할 필요가 없습니다.
* jax.grad를 사용하여 손실 함수의 기울기를 계산합니다.
* 파라미터를 업데이트하기 위해 그래디언트의 pytree를 옵티마이저에 적용합니다.

JAX의 @jit 데코레이터를 사용하여 train_step 함수 전체를 추적하고 XLA로 JIT 컴파일하여 하드웨어 가속기에서 더 빠르고 효율적으로 실행되는 fused device 연산으로 변환합니다.


```python
@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state

```

# 6. Metric computation

손실과 정확도 메트릭을 위한 별도의 함수를 작성합니다. 손실은 optax.softmax_cross_entropy_with_integer_labels 함수를 사용하여 계산하고, 정확도는 clu.metrics를 사용하여 계산합니다.


```python
@jax.jit
def compute_metrics(*, state, batch):
  logits = state.apply_fn({'params': state.params}, batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
  metric_updates = state.metrics.single_from_model_output(
    logits=logits, labels=batch['label'], loss=loss)
  metrics = state.metrics.merge(metric_updates)
  state = state.replace(metrics=metrics)
  return state
```

# 7. 데이터 다운로드




```python
num_epochs = 10
batch_size = 32

train_ds, test_ds = get_datasets(num_epochs, batch_size)
```

# 8. Seed randomness

- 데이터셋 셔플을 재현할 수 있도록 TF random seed를 설정합니다.(`tf.data.Dataset.shuffle` 사용)

- `PRNGKey`를 사용해 매개변수를 초기화 합니다.( [`JAX PRNG 디자인`](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html) 및 [`PRNG chains`](https://flax.readthedocs.io/en/latest/philosophy.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables)에 대해 자세히 알아보기).


```python
tf.random.set_seed(0)
```


```python
init_rng = jax.random.PRNGKey(0)
```

# 9. TrainState 초기화


`create_train_state` 함수는 모델 매개변수, 옵티마이저 및 메트릭을 초기화 합니다. 이는 학습 상태(training state) 데이터 클래스에 입력되고, 해당 데이터 클래스가 함수의 출력으로 반환됩니다.


```python
learning_rate = 0.01
momentum = 0.9
```


```python
state = create_train_state(cnn, init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.
```

# 10. 학습 및 평가

"셔플된" 데이터셋을 생성합니다.
- 데이터셋은 학습 에폭 수만큼 반복됩니다.
- 무작위 배치를 샘플링할 1,024 크기의 버퍼를 할당합니다. 해당 버퍼는 첫 1,024개의 샘플을 포함합니다.
  - 버퍼에서 샘플이 무작위로 추출될 때마다, 데이터셋 내 다하음 샘플이 버퍼에 로드됩니다.

학습 루프를 정의합니다.
- 데이터셋에서 배치를 무작위 샘플링합니다.
- 각 학습 배치마다 최적화 단계를 실행합니다.
- 에폭의 각 배치마다 평균 학습 메트릭을 계산합니다.
- 업데이트된 매개변수를 사용하여 테스트셋 메트릭을 계산합니다.
- 시각화를 위해 학습 및 테스트 메트릭을 기록합니다.

10 에폭 뒤 학습 및 테스트가 완료되면, 대략 99%의 정확도가 달성된 것을 확인할 수 있습니다.


```python
# since train_ds is replicated num_epochs times in get_datasets(), we divide by num_epochs
num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs
```


```python
for step,batch in enumerate(train_ds.as_numpy_iterator()):

  # Run optimization steps over training batches and compute batch metrics
  state = train_step(state, batch) # get updated train state (which contains the updated parameters)
  state = compute_metrics(state=state, batch=batch) # aggregate batch metrics

  if (step+1) % num_steps_per_epoch == 0: # one training epoch has passed
    for metric,value in state.metrics.compute().items(): # compute metrics
      metrics_history[f'train_{metric}'].append(value) # record metrics
    state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

    # Compute metrics on the test set after each training epoch
    test_state = state
    for test_batch in test_ds.as_numpy_iterator():
      test_state = compute_metrics(state=test_state, batch=test_batch)

    for metric,value in test_state.metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)

    print(f"train epoch: {(step+1) // num_steps_per_epoch}, "
          f"loss: {metrics_history['train_loss'][-1]}, "
          f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
    print(f"test epoch: {(step+1) // num_steps_per_epoch}, "
          f"loss: {metrics_history['test_loss'][-1]}, "
          f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")
```

# 11. 메트릭 시각화



```python
import matplotlib.pyplot as plt  # Visualization

# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train','test'):
  ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
  ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()
plt.clf()
```

# 12. 테스트셋에서 추론 수행

jit 컴파일된 추론 함수 `pred_step`을 정의합니다. 학습된 매개변수를 사용하여 테스트셋에서 모델 추론을 수행하고, 입력 이미지와 예측된 레이블을 시각화합니다.


```python
@jax.jit
def pred_step(state, batch):
  logits = state.apply_fn({'params': state.params}, test_batch['image'])
  return logits.argmax(axis=1)

test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(state, test_batch)
```


```python
fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
    ax.set_title(f"label={pred[i]}")
    ax.axis('off')
```
