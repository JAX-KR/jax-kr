# JAX의 고급 자동 미분

*원저자: Vlatimir Miklik & Matteo Hessel*

*역자 : 이영빈*

기울기(gradient)를 계산하는 것은 현대 머신러닝 기법에서 중요한 영역입니다. 이번 섹션은 현대 머신러닝에 관련된 자동 미분의 몇가지 고급 주제를 다룹니다.

JAX를 사용하는 대부분의 경우, 자동 미분 동작 방식을 이해할 필요는 없습니다. 하지만 더 깊은 이해를 위해 이 [영상](https://www.youtube.com/watch?v=wG_nF1awSSY)을 시청하는 것을 권장합니다.

자동 미분 쿡북 섹션은 JAX 백엔드에서 자동 미분 아이디어가 구현되는 방식에 대해 더 높은 수준의 자세한 설명을 제공합니다. 이는 JAX를 이해하기 위해 필수 조건은 아닙니다. 그러나 맞춤 미분 정의(defining custom derivatives)와 같은 몇가지 기능들은 이것을 이해해야 사용할 수 있습니다. 따라서 그러한 기능들을 사용해야할 때를 대비하여 알아둘 가치가 있습니다.



## 고계도함수(Higher-order derivatives)

JAX의 자동미분을 통해 고계도함수를 쉽게 계산할 수 있다. 도함수를 계산하는 함수 그 자체가 미분가능한 함수이기 때문이다. 그러므로 고계도함수는 변환을 쌓는것처럼 쉽게 구현할 수 있습니다.

이는 단일 변수 사례를 통해 확인할 수 있습니다.

$f(x) = x^3 + 2x^2 - 3x + 1$의 도함수는 다음과 같이 계산됩니다.


```python
import jax

f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)
```

$f$의 고차 미분은 다음과 같습니다.

$$
\begin{array}{l}
f'(x) = 3x^2 + 4x -3\\
f''(x) = 6x + 4\\
f'''(x) = 6\\
f^{iv}(x) = 0
\end{array}
$$

JAX에서 이 모든 계산은 grad 함수를 연쇄적으로 사용하는 것만으로 쉽게 해결됩니다.


```python
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)
```

위의 내용을 $x=1$이라 넣고 계산하면 다음과 같습니다.: 

$$
\begin{array}{l}
f'(1) = 4\\
f''(1) = 10\\
f'''(1) = 6\\
f^{iv}(1) = 0
\end{array}
$$

JAX를 사용하면:


```python
print(dfdx(1.))
print(d2fdx(1.))
print(d3fdx(1.))
print(d4fdx(1.))
```

    4.0
    10.0
    6.0
    0.0


다변수인 경우, 고계도함수는 더 복잡합니다. 어떤 함수의 2계도함수는 해당 함수의 [헤시안 행렬](https://en.wikipedia.org/wiki/Hessian_matrix)로 표현될 수 있습니다. 이는 다음과 같이 정의됩니다.

$$(\mathbf{H}f)_{i,j} = \frac{\partial^2 f}{\partial_i\partial_j}.$$

여러 변수의 실수 함수의 헤시안은 ($f: \mathbb R^n\to\mathbb R$)은 함수의 그레디언트의 자코비안 행렬과 동일하게 볼 수 있습니다. JAX는 함수의 자코비안을 계산하기 위해 `jax.jacfwd`와 `jax.jacrev`라는 2가지 변환을 제공합합니다. `jax.jacfwd`는 순뱡향 자동미분이며 `jax.jacrev`는 역방향향 자동미분이다. 이 변환들은 같은 답을 제공하지만 환경에 따른 효율성 차이가 있습니다. -자세한 내용은 [자동미분에 대한 비디오](https://www.youtube.com/watch?v=wG_nF1awSSY)를 참고하세요.


```python
def hessian(f):
  return jax.jacfwd(jax.grad(f))
```

접곱에서도 맞는지 다시 한번 확인해보자. $f: \mathbf{x} \mapsto \mathbf{x} ^\top \mathbf{x}$.

if $i=j$, $\frac{\partial^2 f}{\partial_i\partial_j}(\mathbf{x}) = 2$. Else, $\frac{\partial^2 f}{\partial_i\partial_j}(\mathbf{x}) = 0$.


```python
import jax.numpy as jnp

def f(x):
  return jnp.dot(x, x)

hessian(f)(jnp.array([1., 2., 3.]))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-5df217671693> in <module>
          4   return jnp.dot(x, x)
          5 
    ----> 6 hessian(f)(jnp.array([1., 2., 3.]))
    

    NameError: name 'hessian' is not defined


한편 헤시안 행렬 전체를 항상 계산할 필요는 없으며, 이는 매우 비효율적이기도 합니다.  [자동미분 쿡북](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)에서 헤시안-벡터곱(Hessian-vector product)과 같은 몇가지 트릭을 설명합니다. 헤시안-벡터곱은 헤시안 행렬 전체를 구현하지 않으면서 헤시안을 사용합니다.

만일 JAX에서 고계도함수를 사용하고자 한다면, 자동미분 쿡북을 일독하는 것을 강력히 권장합니다.

고차 최적화(Higher order optimization)

Model-Agnostic Meta-Learning([MAML](https://arxiv.org/abs/1703.03400)) 과 같은 메타러닝 기술들은 기울기 업데이트를 통한 미분이 필요합니다. 다른 프레임워크에서 이는 꽤 번거롭지만, JAX에서는 매우 간단합니다.

```python
def meta_loss_fn(params, data):
  """Computes the loss after one step of SGD."""
  grads = jax.grad(loss_fn)(params, data)
  return loss_fn(params - lr * grads, data)

meta_grads = jax.grad(meta_loss_fn)(params, data)
```

## 기울기 중지

자동미분은 함수의 입력에 대한 기울기를 자동으로 계산합니다. 그러나 때때로 추가적인 제어가 필요합니다. 예를 들어 계산 그래프 중 일부에서 기울기 역전파를 원하지 않을 수도 있습니다.


예를 들어 TD(0)([temporal difference](https://en.wikipedia.org/wiki/Temporal_difference_learning)) 강화학습 업데이트를 예로 들겠습니다. 이 업데이트는 어떤 환경과 상호작용하는 경험을 통해 그 환경에서 상태의 가치를 추정할 때 사용됩니다. 상태 $s_{t-1}$에 있는 값 추정치 $v_{\theta}(s_{t-1}$)가 선형 함수에 의해 파라미터화된다는걸 가정해봅시다.



```python
# 가치함수 와 초기 매개변수
value_fn = lambda theta, state: jnp.dot(theta, state)
theta = jnp.array([0.1, -0.1, 0.])
```

우리가 보상 $r_t$를 관찰하고 있을때 상태 $s_{t-1}$에서 상태 $s_t$로 이동한다고 가정해봅시다.


```python
# 전환예시
s_tm1 = jnp.array([1., 2., -1.])
r_t = jnp.array(1.)
s_t = jnp.array([2., 1., 0.])
```

네트워크 매개변수들을 업데이트한 TD(0)는 다음과 같습니다.

$$
\Delta \theta = (r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})) \nabla v_{\theta}(s_{t-1})
$$

이 업데이트는 어떠한 손실 함수의 그레디언트가 아닙니다.

그러나 만일 파라미터 $\theta$에서 타겟인 $r_t + v_{\theta}(s_t)$의 의존성을 무시한다면 해당 업데이트는 가짜 손실함수의 그레디언트로 **쓰일수도** 있습니다.

$$
L(\theta) = [r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})]^2
$$

어떻게 하면 JAX로 이를 구현할 수 있을까요? 우리가 유사 손실을 나이브하게 작성한다면 아래와 같습니다.


```python
def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return (target - v_tm1) ** 2

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

delta_theta
```




    DeviceArray([ 2.4, -2.4,  2.4], dtype=float32)



그러나 `td_update`는 TD(0) 업데이트를 계산하지 않습니다.. 왜냐하면 그레디언트 계산은 $\theta$에서 `target`의 의존성을 포함할 것이기 때문입니다.

우리는 `jax.lax.stop_gradient`를 이용해 JAX가 $\theta$에서 타겟의 의존성을 무시되도록 강제할 수 있습니다.


```python
def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return (jax.lax.stop_gradient(target) - v_tm1) ** 2

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

delta_theta
```




    DeviceArray([-2.4, -4.8,  2.4], dtype=float32)



이것은 마치 파라미터 $\theta$에 **의존하지 않고** 파라미터로 정확한 업데이트를 계산하는 것처럼 `target`을 계산합니다.

`jax.lax.stop_gradient`는 다른 상황에서도 매우 유용할 수도 있습니다. 예를 들어 만일 당신이 뉴럴 네트워크의 파라미터중 일부에게만 영향을 주기 위해 몇몇개의 손실로부터 그레디언트를 원한다면 유용하게 사용할 수 있다. 왜냐하면 다른 파라미터들은 다른 손실을을 사용해 훈련할 수 있기 떄문이다.



## `stop_gradient`를 이용한 Straight-through 측정기 (STE)

STE는 STE를 사용하지 않으면 미분불가능한 함수의 '그레디언트'를 정의할 때 쓰는 트릭입니다다. 미분불가능한 함수 $f : \mathbb{R}^n \to \mathbb{R}^n$ 가 우리가 더 큰 함수의 일부이며 우리가 그 함수의 그레디언트를 찾는것이라고 가정해봅시다. 우리는 단순히 역전파 하는 동안에 $f$가 항등함수로 간주합니다. 이는 `jax.lax.stop_gradient`를 사용해 깔끔하게 구현됩니다.


```python
def f(x):
  return jnp.round(x)  #미분불가능합니다.

def straight_through_f(x):
  #정확히 한 개의 기울기를 가진 Sterbenz 보조정리를 사용하여 정확히 0인 식을 만듭니다.
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(f(x))

print("f(x): ", f(3.2))
print("straight_through_f(x):", straight_through_f(3.2))

print("grad(f)(x):", jax.grad(f)(3.2))
print("grad(straight_through_f)(x):", jax.grad(straight_through_f)(3.2))
```

    f(x):  3.0
    straight_through_f(x): 3.0
    grad(f)(x): 0.0
    grad(straight_through_f)(x): 1.0


## 샘플별(Per-example) 기울기

대부분의 머신러닝 시스템은 계산 효율성 또는 분산 감소를 위해 데이터 배치(batches)로부터 기울기와 업데이트를 계산합니다. 하지만 때로는 배치(batch)의 특정 샘플과 연관된 기울기 및 업데이트에 접근해야 합니다.

예를 들어 기울기 크기에 따라 데이터 우선 순위를 정하거나 각 샘플 단위로 클리핑(clipping), 정규화를 적용하기 위해 필요합니다.

Pytorch, TF, Theano 등의 프레임워크에서 샘플별 기울기를 계산하는 것은 작은 일이 아닙니다. 해당 라이브러리들이 배치의 기울기를 직접 누적하기 때문입니다. 샘플별 손실을 각각 계산한 후 결과로 나온 기울기를 집계하는 것과 같은 나이브한 차선책은 매우 비효율적입니다.

JAX에서는 쉽고 효율적인 방법으로 샘플별 기울기 계산을 정의할 수 있습니다.

`jit`, `vmap` 그리고 `grad` 변환을 같이 조합하기만 하면 됩니다.


```python
perex_grads = jax.jit(jax.vmap(jax.grad(td_loss), in_axes=(None, 0, 0, 0)))

# 테스트해봅시다.
batched_s_tm1 = jnp.stack([s_tm1, s_tm1])
batched_r_t = jnp.stack([r_t, r_t])
batched_s_t = jnp.stack([s_t, s_t])

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```




    DeviceArray([[-2.4, -4.8,  2.4],
                 [-2.4, -4.8,  2.4]], dtype=float32)



이 변환을 한 번에 하나씩 살펴봅시다.

우선 `td_loss`에 `jax.grad`를 적용해 (배치 적용 되지 않은) 단일입력의 매개변수에 대한 손실의 기울기를 계산하는 함수를 얻습니다.


```python
dtdloss_dtheta = jax.grad(td_loss)

dtdloss_dtheta(theta, s_tm1, r_t, s_t)
```




    DeviceArray([-2.4, -4.8,  2.4], dtype=float32)



이 함수는 위 배열의 한 행을 계산합니다.

그리고나서 우리는 `jax.vmap`을 사용해 이 함수를 벡터화합니다. `jax.vmap`은 배치 차원에 모든 입력과 출력이 추가됩니다. 지금 입력의 배치가 있다고 하면 우리는 출력력의 배치를 얻는다. 배치의 각 출력은 입력 배치의 원소에 대한 기울기와 대응합니다.


```python
almost_perex_grads = jax.vmap(dtdloss_dtheta)

batched_theta = jnp.stack([theta, theta])
almost_perex_grads(batched_theta, batched_s_tm1, batched_r_t, batched_s_t)
```




    DeviceArray([[-2.4, -4.8,  2.4],
                 [-2.4, -4.8,  2.4]], dtype=float32)



이것은 우리가 원하는 것이 아닙니다. 왜냐하면 우리는 이 함수를 수동으로 `theta`의 배치들을 줘야 하는데 우리는 실질적으로 하나의 `theta`를 사용하고 싶기 때문입니다. 우리는 `in_axes`에 `jax.vmap`을 추가해서 해결할 수 있습니다. 이때 theta는 `None`으로 치고 다른 매개변수들은 `0`으로 지정합니다. 이 방식은 결과로 나온 함수를 만들고 다른 매개변수들만 추가 축을 더하고 
`theta`를 배치가 되지 않은채로 뺀다. 


```python
inefficient_perex_grads = jax.vmap(dtdloss_dtheta, in_axes=(None, 0, 0, 0))

inefficient_perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```




    DeviceArray([[-2.4, -4.8,  2.4],
                 [-2.4, -4.8,  2.4]], dtype=float32)



거의 다 왔습니다! 하지만 이대로는 목표로했던 것보다 느립니다. 컴파일된 효율적인 버전을 얻기 위해 전부 `jax.jit`으로 감쌉니다.


```python
perex_grads = jax.jit(inefficient_perex_grads)

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```




    DeviceArray([[-2.4, -4.8,  2.4],
                 [-2.4, -4.8,  2.4]], dtype=float32)




```python
%timeit inefficient_perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t).block_until_ready()
%timeit perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t).block_until_ready()
```

    10.6 ms ± 4.24 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
    53.9 µs ± 1.63 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

