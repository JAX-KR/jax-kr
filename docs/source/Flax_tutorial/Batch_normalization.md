# 배치 정규화 (Batch normalization)
<a href="https://drive.google.com/file/d/1d7aM9W3nQjccglKXPVFcGNOFBt0h5YdX/view?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


*번역: 장혜선* <br/>

이 가이드에서는 `flax.linen.BatchNorm`을 사용하여 [배치 정규화](https://arxiv.org/abs/1502.03167)를 적용하는 방법을 알아볼 것입니다.

배치 정규화는 학습 속도를 높이고 수렴을 개선하기 위해 사용되는 정규화 기법입니다. 학습 중, 특징 차원을 기준으로 이동 평균을 계산합니다. 이로 인해 비미분 (non-differentiable) 상태의 새로운 형태가 추가되므로 적절히 처리해야 합니다.

가이드에서 Flax의 ``BatchNorm`` 을 사용한 코드 예제와 사용하지 않은 코드 예제를 비교할 수 있습니다.

## 배치 정규화를 사용하여 모델 정의하기

Flax에서 `BatchNorm`은 학습과 추론 사이에 서로 다른 런타임(runtime, 실행 시간?) 동작을 나타내는 `flax.linen.Module`입니다. 이는 아래 예시와 같이 `use_running_average` 인수를 사용하여 명시적으로 지정할 수 있습니다.

일반적인 패턴은 부모(parent) Flax 모듈(Module)에서 학습(학습 중인) 인수를 받아와 `BatchNorm`의 `use_running_average` 인수를 정의하는 것 입니다.

참고: Pytorch나 TensorFlow (Keras)와 같은 다른 머신 러닝 프레임워크에서는 이를 가변 상태 또는 호출 플래그(flag)를 통해 지정합니다. (예를 들어, [torch.nn.Module.eval](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)이나 `tf.keras.Model`에서 학습 플래그를 설정하는 것)

No BatchNorm


```python
class MLP(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=4)(x)

    x = nn.relu(x)
    x = nn.Dense(features=1)(x)
    return x
```

With BatchNorm


```python
class MLP(nn.Module):
  @nn.compact
  def __call__(self, x, train: bool):
    x = nn.Dense(features=4)(x)
    x = nn.BatchNorm(use_running_average=not train)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1)(x)
    return x
```

모델을 생성한 후에는 `flax.linen.init()`를 호출하여 `variable` 구조를 초기화합니다. 여기서 `BatchNorm`을 사용하지 않는 코드와 `BatchNorm`을 사용하는 코드의 주요 차이점은 `train`인수가 꼭 제공되어야 한다는 것입니다.

## `batch_stats` 컬렉션(collection)

`BatchNorm`은 `params` 컬렉션 외에도 배치 통계의 이동 평균을 포함하는 `batch_stats` 컬렉션을 추가합니다.

참고: `flax.linen` [변수](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/variable.html) API 문서에서 더 자세한 내용을 확인할 수 있습니다.

`batch_stats` 컬렉션은 나중에 사용할 수 있도록 `variables`에서 꼭 추출되어야 합니다.

No BatchNorm


```python
mlp = MLP()
x = jnp.ones((1, 3))
variables = mlp.init(jax.random.PRNGKey(0), x)
params = variables['params']


jax.tree_util.tree_map(jnp.shape, variables)
```

With BatchNorm


```python
mlp = MLP()
x = jnp.ones((1, 3))
variables = mlp.init(jax.random.PRNGKey(0), x, train=False)
params = variables['params']
batch_stats = variables['batch_stats']

jax.tree_util.tree_map(jnp.shape, variables)
```

Flax의 `BatchNorm`은 총 4개의 변수를 추가합니다: `batch_stats` 컬렉션에 있는 `mean`과 `var`, 그리고 `params` 컬렉션에 있는 `scale`과 `bias` 입니다.

No BatchNorm


```python
FrozenDict({
  'params': {
    'Dense_0': {
        'bias': (4,),
        'kernel': (3, 4),
    },
    'Dense_1': {
        'bias': (1,),
        'kernel': (4, 1),
    },
  },
})
```

With BatchNorm


```python
FrozenDict({
  'batch_stats': {
    'BatchNorm_0': {
        'mean': (4,),
        'var': (4,),
    },
  },
  'params': {
    'BatchNorm_0': {
        'bias': (4,),
        'scale': (4,),
    },
    'Dense_0': {
        'bias': (4,),
        'kernel': (3, 4),
    },
    'Dense_1': {
        'bias': (1,),
        'kernel': (4, 1),
    },
  },
})
```

## `flax.linen.apply` 수정하기

`flax.linen.apply`를 사용하여 `train==True` 인수와 모델을 실행할 때 (즉, `BatchNorm` 호출에서 `use_running_average==False`로 설정한 경우) 다음 사항을 고려해야 합니다.:

- `batch_stats`는 입력 변수로 전달되어야 합니다.
- `batch_stats` 컬렉션은 `mutable=['batch_stats]`로 설정하여 가변으로 표시되어야 합니다.
- 변경된 변수는 두 번째 출력으로 반환됩니다. 업데이트 된 `batch_stats`는 여기에서 추출되어야 합니다.

No BatchNorm


```python
y = mlp.apply(
  {'params': params},
  x,
)
...
```

With BatchNorm


```python
y, updates = mlp.apply(
  {'params': params, 'batch_stats': batch_stats},
  x,
  train=True, mutable=['batch_stats']
)
batch_stats = updates['batch_stats']
```

## 학습 및 평가

`BatchNorm`을 사용하는 모델을 학습 루프에 통합할 때 가장 큰 문제는 추가 `batch_stats` 상태를 처리하는 것 입니다. 이를 수행하기 위해서는 다음과 같이 해야합니다.:
- 사용자 정의 `flax.training.train_state.TrainState` 클래스에 `batch_stats` 필드를 추가합니다.
- `batch_stats` 값을 `train_state.TrainState.create` 메서드에 전달합니다.

No BatchNorm


```python
from flax.training import train_state


state = train_state.TrainState.create(
  apply_fn=mlp.apply,
  params=params,

  tx=optax.adam(1e-3),
)
```

With BatchNorm


```python
from flax.training import train_state

class TrainState(train_state.TrainState):
  batch_stats: Any

state = TrainState.create(
  apply_fn=mlp.apply,
  params=params,
  batch_stats=batch_stats,
  tx=optax.adam(1e-3),
)
```

또한 이러한 변경 사항을 반영하도록 `train_step` 함수를 업데이트 해야합니다.:
- 모든 새 매개변수를 `flax.linen.apply`에 전달합니다. (이전 설명대로)
- `batch_stats`에 대한 `updates`는 `loss_fn`에서 전파되어야 합니다.
- `TrainState`의 `batch_stats`를 업데이트 해야합니다.

No BatchNorm


```python
@jax.jit
def train_step(state: TrainState, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = state.apply_fn(
      {'params': params},
      x=batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)

  metrics = {
    'loss': loss,
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
  }
  return state, metrics
```

With BatchNorm


```python
@jax.jit
def train_step(state: TrainState, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits, updates = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats},
      x=batch['image'], train=True, mutable=['batch_stats'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label'])
    return loss, (logits, updates)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits, updates)), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  state = state.replace(batch_stats=updates['batch_stats'])
  metrics = {
    'loss': loss,
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
  }
  return state, metrics
```

`eval_step`은 훨씬 간단합니다. 왜냐하면 `batch_stats`가 가변적이지 않기 때문에 업데이트를 전파할 필요가 없습니다. 다만 `flax.linen.apply`에 `batch_stats`를 전달하고 `train` 인수가 `False`로 설정되어 있는지 확인해야 합니다.:

No BatchNorm


```python
@jax.jit
def eval_step(state: TrainState, batch):
  """Train for a single step."""
  logits = state.apply_fn(
    {'params': params},
    x=batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label'])
  metrics = {
    'loss': loss,
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
  }
  return state, metrics
```

With BatchNorm


```python
@jax.jit
def eval_step(state: TrainState, batch):
  """Train for a single step."""
  logits = state.apply_fn(
    {'params': params, 'batch_stats': state.batch_stats},
    x=batch['image'], train=False)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label'])
  metrics = {
    'loss': loss,
      'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
  }
  return state, metrics
```
