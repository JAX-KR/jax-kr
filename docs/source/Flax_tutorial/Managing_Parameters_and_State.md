
# 파라미터와 상태 관리하기


번역: 이영빈 \\



이번 세션에서는 다음과 같은 내용을 살펴볼 것이다.

- 초기화부터 업데이트까지 변수 관리하기
- 파라미터와 상태를 분리하고 재결합하기
- 배치 종속적인 상태와 함께 vmap 사용하기



```python

! pip install --upgrade pip jax jaxlib
! pip install --upgrade git+https://github.com/google/flax.git
! pip install optax
```


```python
from functools import partial

import jax
from jax import numpy as jnp
from jax import random
import optax

from flax import linen as nn



# 랜덤 변수 초기화하기
dummy_input = jnp.ones((32, 5))

X = random.uniform(random.PRNGKey(0), (128, 5),  minval=0.0, maxval=1.0)
noise = random.uniform(random.PRNGKey(0), (),  minval=0.0, maxval=0.1)
X += noise

W = random.uniform(random.PRNGKey(0), (5, 1),  minval=0.0, maxval=1.0)
b = random.uniform(random.PRNGKey(0), (),  minval=0.0, maxval=1.0)

Y = jnp.matmul(X, W) + b

num_epochs = 5
```


```python
class BiasAdderWithRunningMean(nn.Module):
  momentum: float = 0.9

  @nn.compact
  def __call__(self, x):
    is_initialized = self.has_variable('batch_stats', 'mean')
    mean = self.variable('batch_stats', 'mean', jnp.zeros, x.shape[1:])
    bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), x.shape[1:])
    if is_initialized:
      mean.value = (self.momentum * mean.value +
                    (1.0 - self.momentum) * jnp.mean(x, axis=0, keepdims=True))
    return mean.value + bias
```

이번 예시 모델은 파라미터(`self.param`)와 상태 변수(`self.variables`)를 모두 포함하는 간단한 모델이다.

여기서 초기화할 때 까다로운 부분은 상태 변수와 최적화할 매개변수를 분리해야 한다는 점입니다.

먼저 `update_step`을 다음과 같이 정의한다.(더미 손실 함수를 사용하여 사용자 함수로 대체해야 함):


```python
@partial(jax.jit, static_argnums=(0,))
def update_step(apply_fn, x, opt_state, params, state):
  def loss(params):
    y, updated_state = apply_fn({'params': params, **state},
                                x, mutable=list(state.keys()))
    l = ((x - y) ** 2).sum() # Replace with your loss here.
    return l, updated_state

  (l, updated_state), grads = jax.value_and_grad(
      loss, has_aux=True)(params)
  updates, opt_state = tx.update(grads, opt_state)  # Defined below.
  params = optax.apply_updates(params, updates)
  return opt_state, params, updated_state
```


실제 학습 코드를 작성할 수 있게 된다.



```python
model = BiasAdderWithRunningMean()
variables = model.init(random.PRNGKey(0), dummy_input)
# 옵티마이저에 의해 업데이트 된 상태변수와 파라미터들을 나눈다.

state, params = variables.pop('params')
del variables  # 리소스를 낭비하는 변수를 제거한다.
tx = optax.sgd(learning_rate=0.02)
opt_state = tx.init(params)

for epoch_num in range(num_epochs):
  opt_state, params, state = update_step(
      model.apply, dummy_input, opt_state, params, state)
```

##배치 차원에 걸친 vmap

`vmap`을 사용하면서 배치 차원에 따라 달라지는 상태를 관리하는 경우(예: `BatchNorm`을 사용하는 경우)에는 위의 설정을 약간 수정해야 한다. 배치 차원에 따라 상태가 달라지는 레이어는 엄밀히 말해 벡터화할 수 없기 때문이다. 배치 차원에 대한 통계의 평균을 내기 위해 `lax.pmean()`을 사용하여 배치의 각 항목에 대해 상태가 동기화되도록 해야 한다.

이를 위해서는 두 가지 작은 변경점이 있다. 먼저 모델 정의에서 배치 축의 이름을 지정해야 한다. 여기서는 `BatchNorm`의 `axis_name` 인수를 지정하여 이 작업을 수행한다. 사용자 코드에서는 `lax.pmean()`의 `axis_name` 인수를 직접 지정해야 할 수도 있다.


```python
class MLP(nn.Module):
  hidden_size: int
  out_size: int

  @nn.compact
  def __call__(self, x, train=False):
    norm = partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name="batch", # 배치 차원의 이름을 짓는다.
    )

    x = nn.Dense(self.hidden_size)(x)
    x = norm()(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = norm()(x)
    x = nn.relu(x)
    y = nn.Dense(self.out_size)(x)

    return y
```

둘째, 훈련 코드에서 `vmap`을 호출할 때 동일한 이름을 지정해야 한다.


```python
@partial(jax.jit, static_argnums=(0,))
def update_step(apply_fn, x_batch, y_batch, opt_state, params, state):

  def batch_loss(params):
    def loss_fn(x, y):
      pred, updated_state = apply_fn(
        {'params': params, **state},
        x, mutable=list(state.keys())
      )
      return (pred - y) ** 2, updated_state

    loss, updated_state = jax.vmap(
      loss_fn, out_axes=(0, None),  # `updated_state`을 vmap하면 안된다.
      axis_name='batch'  # 배치 차원의 이름을 짓는다.
    )(x_batch, y_batch)  # `state`는 빼고 `x`와 `y`에 vmap을 설정해라
    return jnp.mean(loss), updated_state

  (loss, updated_state), grads = jax.value_and_grad(
    batch_loss, has_aux=True
  )(params)

  updates, opt_state = tx.update(grads, opt_state)  # 아래와 같이 정의한다.
  params = optax.apply_updates(params, updates)
  return opt_state, params, updated_state, loss
```

Note that we also need to specify that the model state does not have a batch dimension. Now we are able to train the model:

모델 상태에 배치 차원이 없음을 명시해야 한다. 이제 모델을 훈련할 수 있다.


```python
model = MLP(hidden_size=10, out_size=1)
variables = model.init(random.PRNGKey(0), dummy_input)
# 옵티마이저에 의해 업데이트 된 상태변수와 파라미터들을 나눈다.
state, params = variables.pop('params')
del variables # 리소스를 낭비하는 변수를 제거한다.
tx = optax.sgd(learning_rate=0.02)
opt_state = tx.init(params)

for epoch_num in range(num_epochs):
  opt_state, params, state, loss = update_step(
      model.apply, X, Y, opt_state, params, state)
  print(f"Loss for epoch {epoch_num + 1}:", loss)
```
