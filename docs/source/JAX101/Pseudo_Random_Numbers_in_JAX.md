# JAX의 유사난수생성(PRN)

<a href="https://colab.research.google.com/drive/1PkpzL8bkU4IlUbytFc-JX93QXtNX1tsD" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

*저자: Matteo Hessel & Rosalia Schneider* \
*번역: 조영빈*


```python
import warnings
warnings.filterwarnings(action='ignore') 
```

이 섹션에서는 유사난수 생성기(pseudo random number generation, PRNG)에 중점을 둘 것입니다. 이는, 적절한 분포에서 추출된 난수 시퀀스의 속성과 근사한 속성을 가지는 숫자 시퀀스를 알고리즘적으로 생성하는 프로세스입니다.


 PRNG 생성 시퀀스는 일반적으로 `seed`라고 하는 초기 값에 의해 결정되기 때문에 완전한 무작위(난수)는 아닙니다. 난수 추출의 각 단계는 샘플에서 다음으로 전달되는 `state`의 결정적 함수입니다.


 유사 난수 생성은 모든 머신러닝 또는 과학적인 컴퓨팅 프레임워크의 필수 구성 요소입니다. 일반적으로 JAX는 NumPy와 호환되기 위해 노력하지만 유사난수 생성은 눈에 띄는 예외입니다.

 JAX와 NumPy가 취하는 난수 생성 방식의 차이를 더 잘 이해하기 위해 이 섹션에서는 두 가지 접근 방식 모두에 대해 설명할 것 입니다.

## NumPy의 랜덤

NumPy에서는 유사 난수 생성이 `numpy.random` 모듈에 의해 기본적으로 지원됩니다.

 NumPy에서 유사 난수 생성은 전역 `state`를 기반으로 합니다.

 이는 `random.seed(SEED)`를 사용하여 결정적 초기 조건으로 설정할 수 있습니다.


```python
import numpy as np
np.random.seed(0)
```

다음 코드를 사용하여 상태(state) 내용을 검사할 수 있습니다.


```python
def print_truncated_random_state():
  """To avoid spamming the outputs, print only part of the state."""
  full_random_state = np.random.get_state()
  print(str(full_random_state)[:460], '...')

print_truncated_random_state()
```

    ('MT19937', array([         0,          1, 1812433255, 1900727105, 1208447044,
           2481403966, 4042607538,  337614300, 3232553940, 1018809052,
           3202401494, 1775180719, 3192392114,  594215549,  184016991,
            829906058,  610491522, 3879932251, 3139825610,  297902587,
           4075895579, 2943625357, 3530655617, 1423771745, 2135928312,
           2891506774, 1066338622,  135451537,  933040465, 2759011858,
           2273819758, 3545703099, 2516396728, 127 ...


`state`는 각 호출에 의해 랜덤 함수로 업데이트됩니다:


```python
np.random.seed(0)

print_truncated_random_state()

_ = np.random.uniform()

print_truncated_random_state()
```

    ('MT19937', array([         0,          1, 1812433255, 1900727105, 1208447044,
           2481403966, 4042607538,  337614300, 3232553940, 1018809052,
           3202401494, 1775180719, 3192392114,  594215549,  184016991,
            829906058,  610491522, 3879932251, 3139825610,  297902587,
           4075895579, 2943625357, 3530655617, 1423771745, 2135928312,
           2891506774, 1066338622,  135451537,  933040465, 2759011858,
           2273819758, 3545703099, 2516396728, 127 ...
    ('MT19937', array([2443250962, 1093594115, 1878467924, 2709361018, 1101979660,
           3904844661,  676747479, 2085143622, 1056793272, 3812477442,
           2168787041,  275552121, 2696932952, 3432054210, 1657102335,
           3518946594,  962584079, 1051271004, 3806145045, 1414436097,
           2032348584, 1661738718, 1116708477, 2562755208, 3176189976,
            696824676, 2399811678, 3992505346,  569184356, 2626558620,
            136797809, 4273176064,  296167901, 343 ...


NumPy를 사용하면 한번의 함수 호출로 개별 숫자 또는 숫자의 전체 벡터를 샘플링할 수 있습니다. 예를 들어, 다음과 같이 균일 분포(uniform distribution)에서 3개의 스칼라 벡터를 추출할 수 있습니다:


```python
np.random.seed(0)
print(np.random.uniform(size=3))
```

    [0.5488135  0.71518937 0.60276338]


NumPy는 순차적 등가 보증(sequential equivalent guarantee)을 제공합니다. 이는 N개의 숫자를 개별적으로 샘플링하거나 N개의 벡터를 샘플링하면 동일한 유사-난수(pseudo-random) 시퀀스가 생성된다는 것을 의미합니다:


```python
np.random.seed(0)
print("individually:", np.stack([np.random.uniform() for _ in range(3)]))

np.random.seed(0)
print("all at once: ", np.random.uniform(size=3))
```

    individually: [0.5488135  0.71518937 0.60276338]
    all at once:  [0.5488135  0.71518937 0.60276338]


## JAX의 난수

JAX의 난수 생성은 중요한 면에서 NumPy와 다릅니다. 이는 NumPy의 PRNG 디자인 때문인데, JAX에서 원하는 여러 속성을 동시에 보장하는 것이 어렵기 때문입니다. 구체적으로 코드는 다음과 같은 속성을 가지고 있어야 합니다:

1. 재현 가능(reproducible)
2. 병렬화 가능(parallelizable)
3. 벡터로 변환 가능(vectorisable)


 그 이유는 다음에서 논의할 것입니다. 첫째, 전역 상태(state)를 기반으로 한 PRNG 설계의 의미에 대해 초점을 맞추겠습니다. 이 코드를 깊게 생각해보세요:


```python
import numpy as np

np.random.seed(0)

def bar(): return np.random.uniform()
def baz(): return np.random.uniform()

def foo(): return bar() + 2 * baz()

print(foo())
```

    1.9791922366721637


함수 `foo`는 균일 분포에서 샘플링된 두 개의 스칼라를 합산합니다.

이 코드의 출력은 네이티브 파이썬처럼 `bar()` 및 `baz()`에 대한 특정 실행 순서를 가정하는 경우에만 요구 사항 #1을 충족할 수 있습니다.

이것은 이미 파이썬에 의해 시행되고 있기 때문에 NumPy에서는 큰 문제가 아니지만 JAX에서는 문제가 됩니다.

그러나 이렇게 하면 서로의 함수는 jit을 할 때, 서로 의존하지 않지만 순서가 지정되어 있어 `bar`와 `baz`가 병렬화가 가능해야 한다는 요구사항 #2가 위배됩니다.

이 문제를 방지하기 위해 JAX는 전역 상태를 사용하지 않습니다. 대신 랜덤 함수는 `key`라고 하는 상태를 명시적으로 사용합니다.


```python
from jax import random

key = random.PRNGKey(42)

print(key)
```

    [ 0 42]


key는 `(2,)`모양의 단순한 배열입니다.

'랜덤 key'는 본질적으로 '랜덤 시드'의 또 다른 단어입니다. 그러나 Numpy처럼 한 번만 설정하는 것이 아니라, JAX에서는 어떤 랜덤 함수를 호출하던 key가 지정되어야 합니다. 랜덤 함수는 key를 사용하지만 수정하지는 않습니다. 랜덤 함수에 동일한 key를 입력하면 항상 동일한 샘플이 생성됩니다:


```python
print(random.normal(key))
print(random.normal(key))
```

    -0.18471177
    -0.18471177


**참고:** 동일한 key를 다른 랜덤 함수에 제공하면 상관관계가 있는 출력이 생성될 수 있으며, 일반적으로 이는 바람직하지 않습니다.

*경험상, 동일한 출력을 원하는 것이 아니라면 key를 재사용하지 마세요*

서로 다른 독립적인 샘플을 생성하려면 랜덤 함수를 호출할 때마다 스스로 key를 `split()`해야 합니다.


```python
print("old key", key)
new_key, subkey = random.split(key)
del key  # The old key is discarded -- we must never use it again.
normal_sample = random.normal(subkey)
print(r"    \---SPLIT --> new key   ", new_key)
print(r"             \--> new subkey", subkey, "--> normal", normal_sample)
del subkey  # The subkey is also discarded after use.

# Note: you don't actually need to `del` keys -- that's just for emphasis.
# Not reusing the same values is enough.

key = new_key  # If we wanted to do this again, we would use new_key as the key.
```

    old key [ 0 42]
        \---SPLIT --> new key    [2465931498 3679230171]
                 \--> new subkey [255383827 267815257] --> normal 1.3694694


`split()`은 하나의 `key`를 여러 개의 독립적인(유사 난수성 의미에서) key로 변환하는 결정론적 함수입니다. 출력 중 하나를 new_key로 유지하고 고유한 추가 key(`subkey`라고 함)를 랜덤 함수의 입력으로 안전하게 사용한 다음 영원히 폐기할 수 있습니다.

정규 분포에서 다른 샘플을 얻으려면 `key`를 다시 분할해야 합니다. 중요한 점은 동일한 PRNGKey를 두 번 사용하지 않는다는 것입니다. `split()`은 key를 인수로 받기 때문에 분할할 때 이전 key를 버려야 합니다.

`split(key)` 출력의 어느 부분을 `key`라고 하고, 어느 부분을 `subkey`라고 하는지는 중요하지 않습니다. 그것들은 모두 동일한 상태를 가진 유사 난수입니다. key/subkey 규칙을 사용하는 이유는 이 규칙이 향후 어떻게 사용되는지를 추적하기 위해서입니다. subkey는 무작위 기능에 의해 즉시 사용되는 반면, key는 나중에 더 많은 무작위성을 생성하기 위해 유지됩니다.

일반적으로 위의 예는 다음과 같이 간결하게 작성됩니다.


```python
key, subkey = random.split(key)
```

그러면 이전 key가 자동으로 삭제됩니다.

`split()`는 2개가 아니라 필요한 만큼의 key를 생성할 수 있다는 점에 주목할 필요가 있습니다:


```python
key, *forty_two_subkeys = random.split(key, num=43)
```

NumPy와 JAX의 랜덤 모듈 간의 또 다른 차이점은 위에서 언급한 순차 동등성 보장과 관련이 있습니다.

NumPy와 마찬가지로 JAX의 랜덤 모듈은 숫자 벡터 샘플링도 가능하게 합니다. 그러나 JAX는 벡터화를 위한 SIMD 하드웨어에서의 병렬화와 충돌하기 때문에 순차적 동등성 보장을 제공하지 않습니다 (위의 요구사항 #3).

아래의 예시에서, 정규 분포에서 세 개의 subkey를 사용하여 개별적으로 3개의 값을 추출하면, 하나의 key를 제공하고 `shape=(3,)`을 지정하는 것과 다른 결과가 나옵니다.


```python
key = random.PRNGKey(42)
subkeys = random.split(key, 3)
sequence = np.stack([random.normal(subkey) for subkey in subkeys])
print("individually:", sequence)

key = random.PRNGKey(42)
print("all at once: ", random.normal(key, shape=(3,)))
```

    individually: [-0.04838832  0.10796154 -1.2226542 ]
    all at once:  [ 0.18693547 -1.2806505  -1.5593132 ]


위의 권장 사항과 달리 두 번째 예에서는 `random.normal()`에 대한 입력으로 `key`를 직접 사용합니다. 이것은 다른 곳에서 재사용하지 않을 것이기 때문에, 단일사용 원칙을 위반하지 않습니다.
