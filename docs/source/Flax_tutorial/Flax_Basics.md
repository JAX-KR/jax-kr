# Flax 기초

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b35Yrx7E79qWuhovex-cL7g5TIUYVzju?usp=sharing)

[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/google/flax/blob/main/docs/notebooks/state_params.ipynb)


번역: 장진우 [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
)](www.linkedin.com/in/jinwoo1126) \\

해당 노트북은 아래의 흐름에 따라 여러분들께 Flax를 소개합니다.

- Flax 내장 레이어 또는 third-party 모델로부터 모델을 인스턴스화 하는 방법.
- 모델의 매개변수와 수동으로 작성된 훈련을 초기화 하는 방법.
- Flax에서 제공하는 optimizer를 사용하여 훈련을 용이하게 하는 방법.
- 파라미터들과 다른 객체들을 직렬화하는 방법.
- 자체 모델을 만들고 상태를 관리하는 방법.

## 환경 설정 방법

다음은 해당 노트북을 실행시키기 위해 필요한 환경 설정 코드입니다.


```python
# 최신 JAXlib version 설치.
!pip install --upgrade -q pip jax jaxlib
# Flax 설치:
!pip install --upgrade -q git+https://github.com/google/flax.git
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m49.1 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m71.6/71.6 MB[0m [31m13.5 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for jax (pyproject.toml) ... [?25l[?25hdone
      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Installing backend dependencies ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
      Building wheel for flax (pyproject.toml) ... [?25l[?25hdone



```python
import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
```

## Flax를 이용한 선형 회귀

이전의 JAX 노트북에서는 선형 회귀에 대한 실습을 진행했었습니다. 아시다시피, 선형 회귀는 하나의 dense neural network layer를 이용해서 만들 수 있습니다. 다음에는 이에 해당하는 예제를 보고 어떻게 동작이 수행되는지 확인합니다.

이 하나의 dense layer는 kernel parameter $W \in M_{m,n}(R)$을 가지고 있습니다. 해당 커널은 n차원의 input과 모델의 출력이 되는 m차원의 feature와 m차원의 bias parameter $b \in R^m$로 구성되어 있습니다. 해당 dense layer는 입력 값 $x \in R^n$으로부터 $Wx+b$를 반환합니다.

해당 dense layer는 이미 Flax의 `flax.linen` 모듈에서 제공되는 기능입니다.


```python
# 하나의 dense layer 생성 ('features'를 입력 파라미터로 가짐 )
model = nn.Dense(features=5)
```

일반적인 레이어들은 linen.Module 클래스의 서브 클래스에 있습니다.

## 모델 파라미터 & 초기화

파라미터 값들은 모델 자체에 저장되지 않습니다. 따라서, PRNG와 더미 입력데이터를 이용하여 `init`함수를 통해 초기화할 필요가 있습니다.


```python
key1, key2 = random.split(random.PRNGKey(0))
x = random.normal(key1, (10,)) # 더미 입력 데이터
params = model.init(key2, x) # 초기화 호출
jax.tree_util.tree_map(lambda x: x.shape, params) # 출력 형태 확인
```

    WARNING:jax._src.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)





    FrozenDict({
        params: {
            bias: (5,),
            kernel: (10, 5),
        },
    })



*주의 : JAX와 Flax는 NumPy와 같이 row-based 시스템을 따릅니다. 즉, 벡터들이 columne 벡터가 아닌 row 벡터로 표현된다는 뜻입니다. 이 내용은 커널의 shape에서도 확인할 수 있습니다. ( params → kernel : (10, 5) )*

결과는 예상한대로 kernel과 bias의 크기가 같은 사이즈로 생성이 되었습니다. 내부 동작은 다음과 같습니다.

- 더미 입력 데이터 `x`는 shape의 추론을 위해 사용됩니다. 우리는 모델을 선언할 때 모델의 출력에 대해 원하는 feature 갯수만 선언을 하였고, 입력의 크기는 선언하지 않았습니다. Flax는 이러한 선언에서 커널의 입력에 대한 크기를 자동으로 찾아줍니다.
- 랜덤 PRNG 키는 초기화 함수를 위해 사용됩니다. (해당 예제에서는 모듈에서 제공하는 기본값이 있는 초기화 함수를 사용하였습니다.)
- 초기화 함수는 모델이 사용할 초기의 파라미터 세트를 생성하기 위해서 호출됩니다. 이 함수는 `(PRNG key, shape, dtype)`을 인수로 받고, `shape`를 Array로 반홥니다.
- init 함수는 초기화된 매개변수 세트를 반환합니다. (`init` 대신 `init_with_output` 메서드를 사용하면 같은 문법으로 더미 입력에 대한 정방향 패스의 출력 또한 얻을 수 있습니다.)

출력 결과를 보면 parameter들은 `FrozenDict` 인스턴스에 저장이 됩니다. 이는 JAX의 함수적 특성을 다루기 위해 내부의 dict의 변경을 방지하고 사용자가 이러한 변경을 인지할 수 있도록 도와줍니다. 이에 대해서는 해당 내용을 참조하세요. **`[flax.core.frozen_dict.FrozenDict`** API docs](https://flax.readthedocs.io/en/latest/api_reference/flax.core.frozen_dict.html#flax.core.frozen_dict.FrozenDict).

결과적으로, 아래의 예제는 동작하지 않습니다.


```python
try:
    params['new_key'] = jnp.ones((2,2))
except ValueError as e:
    print("Error: ", e)
```

    Error:  FrozenDict is immutable.


주어진 매개변수 세트를 이용하여 모델의 정방향 패스를 수행하기 위해서는 apply 메서드를 사용하면 됩니다 (앞서 말했듯이 매개변수들은 모델과 함께 저장되지 않습니다.). apply 메서드를 사용하기 위해서는 사용할 매개변수와 입력 값을 제공하면 됩니다.


```python
model.apply(params, x)
```




    Array([-1.3721193 ,  0.61131495,  0.6442836 ,  2.2192965 , -1.1271116 ],      dtype=float32)



## 경사하강법

JAX Part를 거치지 않고 바로 해당 노트북으로 왔다면, 여기서 사용할 선형 회귀식은 다음과 같습니다.

특정 데이터 포인터 세트 $\{(x_i, y_i), i \in {1,…,k}, x_i \in R^n, y_i \in R^m\}$로부터, 함수 $f_{W,b}(x) = Wx +b$의 파라미터세트 $W \in M_{m,n}(R), b \in R^m$를 찾고자 하며 이는 해당 함수에 대한 mean squared error를 최소화 하는 방법을 사용할 수 있다.

$$
L(W,b) \rightarrow \frac{1}{k}\sum^k_{i=1}\frac{1}{2}||y_i -f_{W,b}(x_i)||^2_2
$$

여기에서 튜플 이 dense layer의 매개변수와 일치하는 것을 확인할 수 있다. 이를 이용하여 경사하강법을 수행할 수 있다. 사용할 더미 데이터를 생성하여 실습을 진행해봅시다. 해당 데이터는 JAX 파트의 linear regression pytree 예제와 같습니다.


```python
# 차원 설정
n_samples = 20
x_dim = 10
y_dim = 5

# 예측하고자 하는 W와 b를 생성
key = random.PRNGKey(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))
# FrozenDict pytree에 파라미터를 저장
true_params = freeze({'params': {'bias': b, 'kernel': W}})

# 노이즈를 추가하여 샘플 생성
key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(key_noise,(n_samples, y_dim))
print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)
```

    x shape: (20, 10) ; y shape: (20, 5)


jax.value_and_grad()를 이용하여 JAX pytree linear regression 예제에서 수행했던 것과 동일한 학습 루프를 사용하였지만, 차이점은 직접 정의한 feed-forwad 함수 대신에 model.apply()를 사용할 수 있습니다. (JAX example의 predict_pytree())


```python
# JAX 버전과 같지만 여기에서는 model.apply()를 사용
@jax.jit
def mse(params, x_batched, y_batched):
  # (x,y) 쌍에 대한 손실 함수 정의
  def squared_error(x, y):
    pred = model.apply(params, x)
    return jnp.inner(y-pred, y-pred) / 2.0
  # 모든 샘플에 대한 손실의 평균을 계산하기 위한 벡터화
  return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0)
```

마지막으로 gradient descent를 수행합니다.


```python
learning_rate = 0.3  # 학습률
print('Loss for "true" W,b: ', mse(true_params, x_samples, y_samples))
loss_grad_fn = jax.value_and_grad(mse)

@jax.jit
def update_params(params, learning_rate, grads):
  params = jax.tree_util.tree_map(
      lambda p, g: p - learning_rate * g, params, grads)
  return params

for i in range(101):
  # 업데이트 수행
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  params = update_params(params, learning_rate, grads)
  if i % 10 == 0:
    print(f'Loss step {i}: ', loss_val)
```

    Loss for "true" W,b:  0.02363979
    Loss step 0:  35.343876
    Loss step 10:  0.5143469
    Loss step 20:  0.11384159
    Loss step 30:  0.03932674
    Loss step 40:  0.01991621
    Loss step 50:  0.014209136
    Loss step 60:  0.012425653
    Loss step 70:  0.01185039
    Loss step 80:  0.011661786
    Loss step 90:  0.011599408
    Loss step 100:  0.011578696


## Optax를 이용한 최적화

Flax는 최적화를 위해 Flax의 `flax.optim` 패키지를 주로 사용합니다. 하지만 [FLIP #1009](https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md)로 인해 [Optax](https://github.com/deepmind/optax)가 대신 사용되기 때문에 이 패키지는 더 이상 사용되지 않습니다.

Optax의 기본적은 사용 방법은 직관적입니다:

1. 최적화 방법을 선택합니다. (e.g. `optax.adam`)
2. 파라미터를 이용해서 최적화 상태를 생성합니다. (Adam의 경우, 해당 상태는 [momentum values](https://optax.readthedocs.io/en/latest/api.html#optax.adam)를 포함하고 있습니다.)
3. 손실에 대해서 `jax.value_and_grad()`를 이용하여 gradient를 계산합니다.
4. 매 반복마다, Optax의 `update` 함수를 호출하여 내부의 최적화 상태를 업데이트하고 파라미터에 대한 업데이트를 생성합니다. 그 다음 Optax의 `apply_updates` 메서드를 통해 업데이트를 파라미터에 반영합니다.

Optax는 더 많은 일을 할 수 있습니다 :  간단한 gradient 변환을 더 복잡한 변환으로 구성하여 다양한 최적화를 구현할 수 있도록 설계 되어 있습니다. 또한, 시간에 따라 최적화에 사용되는 하이퍼 파라미터를 변경하는 (”스케쥴”) 기능을 지원하며, 매개변수 트리의 특정 부분에 대해 다르게 업데이트를 수행하는 기능 (”마스킹”) 등을 지원합니다. 자세한 내용은 공식 문서를 참조하십시오. [official documentation](https://optax.readthedocs.io/en/latest/).


```python
import optax
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)
```


```python
for i in range(101):
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)
```

    Loss step 0:  0.011577628
    Loss step 10:  0.26143155
    Loss step 20:  0.07675027
    Loss step 30:  0.03644055
    Loss step 40:  0.022012806
    Loss step 50:  0.016178599
    Loss step 60:  0.013002801
    Loss step 70:  0.012026143
    Loss step 80:  0.011764514
    Loss step 90:  0.011646044
    Loss step 100:  0.011585529


## 결과의 직렬화

학습의 결과가 만족스럽다면, 나중에 다시 사용할 수 있도록 매개변수를 저장하고자 할 수 있습니다. Flax에서는 이를 가능하게 해주는 직렬화 패키지를 제공합니다.


```python
from flax import serialization
bytes_output = serialization.to_bytes(params)
dict_output = serialization.to_state_dict(params)
print('Dict output')
print(dict_output)
print('Bytes output')
print(bytes_output)
```

    Dict output
    {'params': {'bias': Array([-1.4555768 , -2.0277991 ,  2.0790975 ,  1.2186145 , -0.99809754],      dtype=float32), 'kernel': Array([[ 1.0098814 ,  0.18934374,  0.04454996, -0.9280221 ,  0.3478402 ],
           [ 1.7298453 ,  0.9879368 ,  1.1640464 ,  1.1006076 , -0.10653935],
           [-1.2029463 ,  0.28635228,  1.4155979 ,  0.11870951, -1.3141483 ],
           [-1.1941489 , -0.18958491,  0.03413862,  1.3169426 ,  0.0806038 ],
           [ 0.1385241 ,  1.3713038 , -1.3187183 ,  0.53152674, -2.2404997 ],
           [ 0.56294024,  0.8122311 ,  0.3175201 ,  0.53455096,  0.9050039 ],
           [-0.37926027,  1.7410393 ,  1.0790287 , -0.5039833 ,  0.9283062 ],
           [ 0.9706492 , -1.3153403 ,  0.33681503,  0.8099344 , -1.2018458 ],
           [ 1.0194312 , -0.6202479 ,  1.0818833 , -1.838974  , -0.45805007],
           [-0.6436537 ,  0.45666698, -1.1329137 , -0.6853864 ,  0.16829035]],      dtype=float32)}}
    Bytes output
    b'\x81\xa6params\x82\xa4bias\xc7!\x01\x93\x91\x05\xa7float32\xc4\x14WP\xba\xbfv\xc7\x01\xc0\xef\x0f\x05@\x8f\xfb\x9b?R\x83\x7f\xbf\xa6kernel\xc7\xd6\x01\x93\x92\n\x05\xa7float32\xc4\xc8\xcbC\x81?S\xe3A>\x06z6=\xdb\x92m\xbf\x1c\x18\xb2>\x92k\xdd?m\xe9|?y\xff\x94?\xb6\xe0\x8c?M1\xda\xbd%\xfa\x99\xbf\xc4\x9c\x92>P2\xb5?\xf9\x1d\xf3=\x036\xa8\xbf\xdf\xd9\x98\xbf\x8c"B\xbe\xef\xd4\x0b=\x93\x91\xa8?\x9b\x13\xa5=C\xd9\r>\xe2\x86\xaf?\xc3\xcb\xa8\xbf#\x12\x08?Yd\x0f\xc0\xda\x1c\x10?a\xeeO?\xff\x91\xa2>U\xd8\x08?V\xaeg?g.\xc2\xbe`\xda\xde?\x9d\x1d\x8a?\r\x05\x01\xbfz\xa5m?w|x?\x12]\xa8\xbf\x05s\xac>\xdcWO?\x15\xd6\x99\xbf\xb9|\x82?\x91\xc8\x1e\xbf\'{\x8a?\x80c\xeb\xbf\x8a\x85\xea\xbe}\xc6$\xbfA\xd0\xe9>Q\x03\x91\xbf|u/\xbfNT,>'


모델을 다시 불러오려면, 모델 초기화 시 얻을 수 있는 것과 같은 모델 매개변수 구조 탬플릿을 사용해야 합니다. 여기서는 이전에 생성된 파라미터들을 탬플릿으로 사용합니다. 새로운 변수 구조를 생성하며, 기존 변수를 직접 변경하지는 않습니다.

탬플릿을 통해 구조를 강제하는 것의 목적은 사용자의 이슈를 방지하기 위함입니다. 따라서 먼저 매개변수의 구조를 생성하는 정확한 모델이 있어야 합니다.


```python
serialization.from_bytes(params, bytes_output)
```




    FrozenDict({
        params: {
            bias: array([-1.4555768 , -2.0277991 ,  2.0790975 ,  1.2186145 , -0.99809754],
                  dtype=float32),
            kernel: array([[ 1.0098814 ,  0.18934374,  0.04454996, -0.9280221 ,  0.3478402 ],
                   [ 1.7298453 ,  0.9879368 ,  1.1640464 ,  1.1006076 , -0.10653935],
                   [-1.2029463 ,  0.28635228,  1.4155979 ,  0.11870951, -1.3141483 ],
                   [-1.1941489 , -0.18958491,  0.03413862,  1.3169426 ,  0.0806038 ],
                   [ 0.1385241 ,  1.3713038 , -1.3187183 ,  0.53152674, -2.2404997 ],
                   [ 0.56294024,  0.8122311 ,  0.3175201 ,  0.53455096,  0.9050039 ],
                   [-0.37926027,  1.7410393 ,  1.0790287 , -0.5039833 ,  0.9283062 ],
                   [ 0.9706492 , -1.3153403 ,  0.33681503,  0.8099344 , -1.2018458 ],
                   [ 1.0194312 , -0.6202479 ,  1.0818833 , -1.838974  , -0.45805007],
                   [-0.6436537 ,  0.45666698, -1.1329137 , -0.6853864 ,  0.16829035]],
                  dtype=float32),
        },
    })



## 사용자 모델 정의하기

Flax는 단순 선형 회귀보다 복잡한 사용자 정의 모델을 정의할 수 있도록 도와줍니다. 해당 섹션에서는 간단한 모델을 구축하는 방법을 보여줍니다. 이를 위해 `nn.Module` 클래스의 서브 클래스를 생성해야 합니다.

기억해야 할 것은 위에서 `linen as nn`을 선언하였고 이는 새로운 linen API와 함께 작동한다는 점입니다.

### 기본 모듈

모델의 기본 추상화는 `nn.Module` 클래스이며, Flax의 미리 정의된 레이어(이전에 사용한 `Dense`와 같은)의 각 유형은 `nn.Module`의 하위 클래스입니다. 간단하지만 사용자 정의의 다중 퍼셉트론인 Dense 레이어와 비선형 함수가 번강하 나오는 시퀀스를 정의하고 살펴보겠습니다.


```python
class ExplicitMLP(nn.Module):
  features: Sequence[int]

  def setup(self):
    # 리스트를 이용하여 self.layers를 작성
    self.layers = [nn.Dense(feat) for feat in self.features]
    # 하나의 하위 모듈만 사용하는 경우, 아래와 같이 작성합니다.
    # self.layer1 = nn.Dense(feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = ExplicitMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
print('output:\n', y)
```

    initialized parameter shapes:
     {'params': {'layers_0': {'bias': (3,), 'kernel': (4, 3)}, 'layers_1': {'bias': (4,), 'kernel': (3, 4)}, 'layers_2': {'bias': (5,), 'kernel': (4, 5)}}}
    output:
     [[ 0.          0.          0.          0.          0.        ]
     [ 0.0072379  -0.00810348 -0.0255094   0.02151717 -0.01261241]
     [ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.        ]]


보시다시피, nn.Module의 하위 클래스는 다음과 같이 구성되어 있습니다. :

- 데이터 필드의 집합 (nn.Module은 Python 데이터 클래스입니다.) - 여기서는 Sequence[int] 유형의 feature 필드로만 구성되어 있습니다.
- `__postinit__`의 끝에서 호출되는 setup() 메서드가 있습니다. 여기에서 모델에 필요한 하위 모듈, 변수, 매개변수를 선언할 수 있습니다.
- `__call__` 함수는 주어진 입력으로부터 모델의 출력을 반환하는 역할을 합니다.
- 모델 구조는 모델과 동일한 트리 구조를 따라서 매개변수의 pytree를 정의합니다. params에는 레이어 당 하나의 `layers_n` 의 하위 dictionary가 있고, 각각에는 해당 Dense 레이어의 매개변수 값을 포함하고 있습니다. 이러한 레이아웃은 매우 명시적입니다.

참고 : 대부분은 예상한 대로 관리되지만, 여기[here](https://github.com/google/flax/issues/524)에서 언급된 대로 알고 있어야 할 코너 케이스들이 있습니다.

모듈 구조와 해당 매개변수는 서로 연결되어 있지 않기 때문에, 주어진 입력에 대해 직접 model(x)를 호출 할 수는 없고 호출하게 되면 오류가 발생합니다. `__call__`은 apply 함수로 쌓여있으며, 이 함수를 입력에 대해 호출 해야 합니다.


```python
try:
    y = model(x) # 에러를 반환
except AttributeError as e:
    print(e)
```

    "ExplicitMLP" object has no attribute "layers". If "layers" is defined in '.setup()', remember these fields are only accessible from inside 'init' or 'apply'.


이번 예제는 매주 간단한 모델이기 때문에, `@nn.compact` 어노테이션을 사용하여 `__call__` 내에서 하위 모듈을 인라인으로 선언하는 대안적인 방법을 사용할 수도 있습니다. 예시는 다음과 같습니다. :


```python
class SimpleMLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
      # layer에 이름을 지정할 수 있습니다!
      # 기본 이름은 "Dense_0", "Dense_1", ...와 같이 지정됩니다.
    return x

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
print('output:\n', y)
```

    initialized parameter shapes:
     {'params': {'layers_0': {'bias': (3,), 'kernel': (4, 3)}, 'layers_1': {'bias': (4,), 'kernel': (3, 4)}, 'layers_2': {'bias': (5,), 'kernel': (4, 5)}}}
    output:
     [[ 0.          0.          0.          0.          0.        ]
     [ 0.0072379  -0.00810348 -0.0255094   0.02151717 -0.01261241]
     [ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.        ]]


그러나 두 가지 모드 사이에 알아둬야 할 몇 가지 차이점이 있습니다:

- `setup`을 이용하면 일부 하위 레이어에 이름을 지정하고 나중에 사용할 수 있습니다 (예: 오토인코더의 인코더/디코더 메서드).
- 여러 메서드를 사용하려면 `@nn.compact` 어노테이션 대신 `setup`을 사용하여 모듈을 선언해야 합니다. `@nn.compact` 어노테이션은 하나의 메서드만 어노테이션으로 허용합니다.
- 마지막 초기화는 다르게 처리됩니다. 자세한 내용은 이 노트를 참조하세요 (TODO: 노트 링크 추가 예정).

## 모듈 파라미터

이전 MLP 예제에서는 미리 정의된 레이어와 연산자 (`Dense`, `relu`)만을 사용했습니다. Flax에서 제공하는 Dense 레이어를 사용할 수 없는 상황에서 직접 작성하고자 한다면, 다음과 같이 `@nn.compact` 방식을 사용하여 새로운 모듈을 선언할 수 있습니다:


```python
class SimpleDense(nn.Module):
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel',
                        self.kernel_init, # 초기화 함수
                        (inputs.shape[-1], self.features))  # 형태 정보
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),)
    bias = self.param('bias', self.bias_init, (self.features,))
    y = y + bias
    return y

key1, key2 = random.split(random.PRNGKey(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleDense(features=3)
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameters:\n', params)
print('output:\n', y)
```

    initialized parameters:
     FrozenDict({
        params: {
            kernel: Array([[ 0.61506   , -0.22728713,  0.6054702 ],
                   [-0.29617992,  1.1232013 , -0.879759  ],
                   [-0.35162622,  0.3806491 ,  0.6893246 ],
                   [-0.1151355 ,  0.04567898, -1.091212  ]], dtype=float32),
            bias: Array([0., 0., 0.], dtype=float32),
        },
    })
    output:
     [[-0.02996203  1.102088   -0.6660265 ]
     [-0.31092793  0.63239413 -0.53678817]
     [ 0.01424009  0.9424717  -0.63561463]
     [ 0.3681896   0.3586519  -0.00459218]]


여기서는 `self.param` 메서드를 사용하여 모델에 매개변수를 선언하고 할당하는 방법을 볼 수 있습니다. 이 메서드는 `(name, init_fn, *init_args)`를 인수로 사용합니다:

- `name`은 매개변수의 이름이며, 매개변수 구조에 저장됩니다.
- `init_fn`은 `(PRNGKey, *init_args)`를 입력으로 받아 Array를 반환하는 함수이며, `init_args`는 초기화 함수를 호출하는 데 필요한 인수입니다.
- `init_args`는 초기화 함수에 제공해야 하는 인수입니다.

이러한 매개변수는 `setup` 메서드에서도 선언할 수 있으며, Flax는 첫 번째 호출 지점에서 지연 초기화를 사용하기 때문에 형상 추론을 사용할 수 없습니다.

## 변수와 변수들의 집합

지금까지 본 바와 같이, 모델 작업은 다음과 같은 작업을 수행하는 것을 의미합니다:

- `nn.Module`의 하위 클래스
- 모델의 매개변수에 대한 pytree (일반적으로 `model.init()`을 수행하여 얻음)

그러나 이것으로는 머신러닝, 특히 신경망에 필요한 모든 것을 다루기에 충분하지 않습니다. 일부 경우에는 신경망이 실행되는 동안 일부 내부 상태를 추적하고자 할 수 있습니다 (예: 배치 정규화 레이어). `variable` 메서드를 사용하여 모델의 매개변수 이외의 변수를 선언하는 방법이 있습니다.

데모 목적으로 배치 정규화와 유사한 단순화된 메커니즘을 구현해 보겠습니다: 실행 평균을 저장하고 학습 시에 입력에서 이를 빼는 방식입니다. 실제 배치 정규화를 사용하려면 (구현 내용을 살펴보려면) 여기([here](https://github.com/google/flax/blob/main/flax/linen/normalization.py))에서 제공하는 구현을 사용해야 합니다.


```python
class BiasAdderWithRunningMean(nn.Module):
  decay: float = 0.99

  @nn.compact
  def __call__(self, x):
    is_initialized = self.has_variable('batch_stats', 'mean')
    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s),
                            x.shape[1:])
    mean = ra_mean.value
    bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), x.shape[1:])
    if is_initialized:
      ra_mean.value = self.decay * ra_mean.value + (1.0 - self.decay) * jnp.mean(x, axis=0, keepdims=True)

    return x - ra_mean.value + bias


key1, key2 = random.split(random.PRNGKey(0), 2)
x = jnp.ones((10,5))
model = BiasAdderWithRunningMean()
variables = model.init(key1, x)
print('initialized variables:\n', variables)
y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
print('updated state:\n', updated_state)
```

    initialized variables:
     FrozenDict({
        batch_stats: {
            mean: Array([0., 0., 0., 0., 0.], dtype=float32),
        },
        params: {
            bias: Array([0., 0., 0., 0., 0.], dtype=float32),
        },
    })
    updated state:
     FrozenDict({
        batch_stats: {
            mean: Array([[0.01, 0.01, 0.01, 0.01, 0.01]], dtype=float32),
        },
    })


여기서 updated_state는 모델이 데이터에 적용될 때 변경되는 상태 변수만을 반환합니다. 변수를 업데이트하고 모델의 새로운 매개변수를 얻으려면 다음 패턴을 사용할 수 있습니다:


```python
for val in [1.0, 2.0, 3.0]:
  x = val * jnp.ones((10,5))
  y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
  old_state, params = variables.pop('params')
  variables = freeze({'params': params, **updated_state})
  print('updated state:\n', updated_state) # mutable 부분만 보여줍니다.
```

    updated state:
     FrozenDict({
        batch_stats: {
            mean: Array([[0.01, 0.01, 0.01, 0.01, 0.01]], dtype=float32),
        },
    })
    updated state:
     FrozenDict({
        batch_stats: {
            mean: Array([[0.0299, 0.0299, 0.0299, 0.0299, 0.0299]], dtype=float32),
        },
    })
    updated state:
     FrozenDict({
        batch_stats: {
            mean: Array([[0.059601, 0.059601, 0.059601, 0.059601, 0.059601]], dtype=float32),
        },
    })


이 간단한 예제를 기반으로, 전체적인 BatchNorm 구현 또는 상태를 포함하는 모든 레이어를 유도할 수 있어야 합니다. 마지막으로, 매개변수를 업데이트하는 옵티마이저와 상태 변수를 함께 사용하는 방법을 보기 위해 옵티마이저를 추가해 보겠습니다.

이 예제는 아무 작업도 수행하지 않으며, 단지 데모 목적으로 사용됩니다.


```python
from functools import partial

@partial(jax.jit, static_argnums=(0, 1))
def update_step(tx, apply_fn, x, opt_state, params, state):

  def loss(params):
    y, updated_state = apply_fn({'params': params, **state},
                                x, mutable=list(state.keys()))
    l = ((x - y) ** 2).sum()
    return l, updated_state

  (l, state), grads = jax.value_and_grad(loss, has_aux=True)(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return opt_state, params, state

x = jnp.ones((10,5))
variables = model.init(random.PRNGKey(0), x)
state, params = variables.pop('params')
del variables
tx = optax.sgd(learning_rate=0.02)
opt_state = tx.init(params)

for _ in range(3):
  opt_state, params, state = update_step(tx, model.apply, x, opt_state, params, state)
  print('Updated state: ', state)
```

    Updated state:  FrozenDict({
        batch_stats: {
            mean: Array([[0.01, 0.01, 0.01, 0.01, 0.01]], dtype=float32),
        },
    })
    Updated state:  FrozenDict({
        batch_stats: {
            mean: Array([[0.0199, 0.0199, 0.0199, 0.0199, 0.0199]], dtype=float32),
        },
    })
    Updated state:  FrozenDict({
        batch_stats: {
            mean: Array([[0.029701, 0.029701, 0.029701, 0.029701, 0.029701]], dtype=float32),
        },
    })


위 함수는 꽤 상세한 시그니처를 가지고 있으며, 실제로는 jax.jit()와 함께 작동하지 않을 것입니다. 이는 함수 인수가 "유효한 JAX 타입"이 아니기 때문입니다.

Flax는 위의 코드를 단순화하는 유용한 래퍼인 `TrainState`를 제공합니다. 자세한 내용은 [flax.training.train_state.TrainState](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.train_state.TrainState)를 확인해 보세요.

## jax2tf를 이용하여 Tensorflow의 SavedModel로 포팅하기

[jax2tf](https://github.com/google/jax/tree/main/jax/experimental/jax2tf)를 사용하면 훈련된 Flax 모델을 Tensorflow의 SavedModel 형식으로 변환할 수 있습니다. 이렇게 변환된 모델은 [TF Hub](https://www.tensorflow.org/hub), [TF.lite](https://www.tensorflow.org/lite), [TF.js](https://www.tensorflow.org/js) 또는 다른 하향 응용 프로그램에서 사용할 수 있습니다. 해당 저장소에는 Flax에 대한 자세한 문서와 다양한 예제가 포함되어 있습니다.
