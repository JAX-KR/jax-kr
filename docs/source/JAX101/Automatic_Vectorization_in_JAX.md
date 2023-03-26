# JAX의 자동 벡터화
---
<a href="https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/03-vectorization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

*저자 : Matteo Hessel*

*역자 : 박정현*

*

이전 섹션에서 jax.jit 함수를 통한 JIT 컴파일에 대해 이야기해 보았습니다.
이 노트북은 JAX의 또 다른 변환인 jax.vmap을 통한 벡터화(vectorization)에 대해 이야기 합니다.

# 수동 벡터화

다음의 간단한 코드를 생각해 보세요. 이 코드는 두 개의 1차원 벡터의 합성곱(convolution)을 계산합니다:


```python
import jax
import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

convolve(x, w)
```

    WARNING:jax._src.lib.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)





    DeviceArray([11., 20., 29.], dtype=float32)



우리는 이 함수를 가중치(weight)의 배치 w와 벡터의 배치 x에 적용하려고 합니다.


```python
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])
```

가장 단순한 시도로는 파이썬으로 배치를 반복하는 방법입니다:


```python
def manually_batched_convolve(xs, ws):
  output = []
  for i in range(xs.shape[0]):
    output.append(convolve(xs[i], ws[i]))
  return jnp.stack(output)

manually_batched_convolve(xs, ws)
```




    DeviceArray([[11., 20., 29.],
                 [11., 20., 29.]], dtype=float32)



이 코드는 알맞는 결과를 잘 가져오지만 그다지 효율적이진 않습니다.

효율적인 배치 처리를 위해, 보통 함수를 수작업으로 재작성합니다. 이는 함수가 벡터화된 형태로 계산되도록 하기 위함입니다. 이런 구현이 특별히 어려운 것은 아니지만, 함수에서 인덱스, 축, 그리고 입력의 다른 부분을 처리하는 방법이 변하게 됩니다.

예를 들어, 다음과 같이 convolve() 함수를 일일이 재작성하여 배치 차원에 걸친 벡터화된 계산을 지원할 수 있습니다:


```python
def manually_vectorized_convolve(xs, ws):
  output = []
  for i in range(1, xs.shape[-1] -1):
    output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
  return jnp.stack(output, axis=1)

manually_vectorized_convolve(xs, ws)
```




    DeviceArray([[11., 20., 29.],
                 [11., 20., 29.]], dtype=float32)




이러한 재구현은 지저분하고 오류가 발생하기 쉽습니다. 하지만 JAX가 제공하는 더 좋은 방법이 있습니다.

# 자동 벡터화

JAX의 jax.vmap 변환은 함수의 벡터화 구현을 자동으로 생성하도록 설계되었습니다.


```python
auto_batch_convolve = jax.vmap(convolve)

auto_batch_convolve(xs, ws)
```




    DeviceArray([[11., 20., 29.],
                 [11., 20., 29.]], dtype=float32)



**jax.jit**처럼 함수를 트레이싱(tracing)하여 각 입력의 시작에 배치 축을 자동으로 추가하여 이를 수행합니다.

배치가 첫 번째 차원이 아닐 경우, **in_axes**나 **out_axes를** 사용해서 입력과 출력에서의 배치 차원의 위치를 지정할 수 있습니다. 배치 축이 모든 입력 및 출력에 대해 동일한 경우에는 정수, 그렇지 않은 경우에는 리스트(list)로 지정하면 됩니다.


```python
auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

auto_batch_convolve_v2(xst, wst)
```




    DeviceArray([[11., 11.],
                 [20., 20.],
                 [29., 29.]], dtype=float32)



**jax.vmap**은 인수가 하나인 경우도 지원합니다: 예를 들어, 벡터 **x**의 배치를 사용하여 가중치 w와 컨볼루션하려는 경우; 이 경우 **in_axes** 인수를 **None**으로 설정하면 됩니다.


```python
batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])

batch_convolve_v3(xs, w)
```




    DeviceArray([[11., 20., 29.],
                 [11., 20., 29.]], dtype=float32)



# 결합된 변환


모든 JAX 변환과 마찬가지로 **jax.jit** 및 **jax.vmap**은 조합하여 구성할 수 있도록 설계되었습니다. vmap 처리된 함수를 jit으로 감싸거나(wrap), JIT 처리된 함수를 vmap으로 감쌀 수 있으며 모두 올바르게 동작할 것 입니다:


```python
jitted_batch_convolve = jax.jit(auto_batch_convolve)

jitted_batch_convolve(xs, ws)
```




    DeviceArray([[11., 20., 29.],
                 [11., 20., 29.]], dtype=float32)


