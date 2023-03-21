# JAX에서 병렬 평가(Parallel Evaluation)

번역일 : 2023.02.06  
번역자 : [유현아](https://www.linkedin.com/in/hayoo2/)  
검수자 : 

<a href="https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/06-parallelism.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


*저자: Vladimir Mikulik & Roman Ring*

이번 세션에서는 SPMD(single-program, multiple-data) 코드를 위해 JAX에 내장된 기능을 설명합니다.

SPMD는 동일한 계산(예: 신경망의 순전파(forward pass))이 병렬로 다른 입력 데이터(예: 배치의 다른 입력)에서 다른 디바이스(예: 여러 개의 TPU)에 대해 실행되는 병렬 처리 기술을 의미합니다.


개념적으로 이것은 동일한 작업이 동일한 디바이스의 다른 메모리 부분에서 병렬로 발생하는 벡터화와 크게 다르지 않습니다. JAX에서 프로그램 변환인 `jax.vmap`으로 벡터화가 지원되는 것을 이미 살펴보았습니다. JAX는 `jax.pmap`을 사용하여 하나의 디바이스를 대상으로 작성된 함수를 여러 디바이스에서 병렬로 실행되는 함수로 변환하여 디바이스 병렬화를 유사하게 지원합니다. 이번 세션에서 모든 것을 알려드립니다.

## Colab TPU 설정(Setup)

Google Colab에서 이 코드를 실행하는 경우 *런타임*→*런타임 유형 변경*을 선택하고 하드웨어 가속기 메뉴에서 **TPU**를 선택해야 합니다.


![하드웨어가속기설정.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAxYAAAGuCAYAAAANuJCWAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAHYcAAB2HAY/l8WUAAD8JSURBVHhe7d0JvNTz/sfxT4tWbVKklBKhVIRE9rW/pZQsXdeWS4s1RLJdu4RwW1xE116E6l5kT0lFhJCIUtKmvVSn+vf+9vsd0zTnnJn5zpwzc87r+Xj8Huc3v5kzM2fOnDnf9+/7/Xy/pTZvYQAAAADgoXTwFQAAAACSRrAAAAAA4I1gAQAAAMAbwQIAAACAN4IFAAAAAG8ECwAAAADeCBYAAAAAvBEsAAAAAHgjWAAAAADwRrAAAAAA4I1gAQAAAMAbwQIAAACAN4IFAAAAAG8ECwAAAADeCBYAAAAAvBEsAAAAAHgjWAAAAADwRrAAAAAA4I1gAQAAAMAbwQIAAACAN4IFAAAAAG8ECwAAAADeCBYAAAAAvBEsAAAAAHgjWAAAAADwRrAAAAAA4I1gAQAAAMAbwQIAAACAN4IFAAAAAG8ECwAAAADeCBYAAAAAvBEsAAAAAHgjWAAAAADwRrAAAAAA4I1gAQAAAMAbwQIAAACAN4IFAAAAAG8ECwAAAADeCBYAAAAAvBEsAAAAAHgjWAAAAADwRrAAAAAA4I1gAQAAAMAbwQIAAACAN4IFAAAAAG8ECwAAAADeCBYAAAAAvBEsAAAAAHgjWAAAAADwRrAAAAAA4I1gAQAAAMAbwQIAAACAN4IFAAAAAG8EC2SEESNGWKNGjdw2adKk4GjmWLp0qXXs2NE9v6uvvtrWrl0bXAMAAAAhWMBLpgcCAAAAFA6CBbJav379coON76b7AgAAQHKKRbCIHKair7qcSTRsRsNnohuyqdoy8WcGAABAyUKPBQAAAABvBAt4WbVqVbBnNmfOnGCv8PTu3dtmzZqV53b//fcHtzR78cUXY94m3HRfAAAASA7BopC1bNnSPv/885gN22S3kSNHWo0aNYJHKDwbNmywmTNnBpfMfvnlF9u8eXNwKfOsWbMm2AMAAECqESyQtPnz528zE9T48eNt4cKFwaW/6DaxakMitxtuuCG4dWotWrQo2DP77rvvgj0AAACkGsECSVHPxP/+9z/7+eefgyNmX3/9tY0bNy64VPRWr15t06ZNCy6ZffLJJ7Zs2bLg0l/iKa5v1aqVffnll8F3AAAAIBrBAknREKjnn3/e7VeuXNlt8tRTTxVJrUUsM2bMcGEiNHnyZPviiy+CSwAAAEglggUSpuFOt912m82bN89d7tWrl/Xt29ft//DDD/boo49uU8/QunXrmLUhkVtkkXUq6PGHDRvmei1COTk59tBDD7khXAAAAEgtgkUh03AaDauJNdwmma2w17BYsmSJ3XLLLbm1FUcccYSdccYZdvrpp7vnIiomv+OOO4qsWDocpvXmm2+6ywo2p512mtufPn263Xfffdu8ZhUrVrQBAwbEDDzhpoJ7Fd4DAAAgNoIF4vbrr7/aVVddZe+884673LBhQ7v55putevXqVqlSJbviiiusRYsW7rrhw4dbnz59imThvokTJ9rdd9/teiiqVavmnteNN95oTZs2ddePHj3aXVZIAgAAQGoQLFCgTZs22dixY+3MM8/MrVmoW7euO/O/1157ucvSoEED69evnwscoga8ejFU21AY09DqMVQ83rNnT1u+fLk7dsEFF7geizp16rigU6tWLXdc4eiSSy7ZZrpcAAAAJI9gUchSvY5FOtewUENda1P06NHDunXrljt1q4LDY489ZgcffLC7HElBY+DAgbb33nu7y7Nnz7bzzjvPNfZVf5GugLFx40a3AN5ll12WGyoUai699FIrU6aMu3zYYYfZww8/nBsuNGOUhnANGjSINS4AAAA8ESwQ07p161ydxIknnuh6K0LHHHOM/ec//8m33mCfffaxJ554wjXkRUOS3nrrLTv55JNt6NChrgcklTQLlYKLhmXpeYtqKlRQriFakfScHn/88dxeFd2+f//+dsIJJ9gLL7xgK1eudMcBAACQmGIXLFJdHB1uWudA6x2UFOXLl7cuXbrY7rvvnnv5zjvvtMGDB7thUAXR9/373/92PR1ly5Z1x1To3alTJytdOjVvuwULFtgDDzxgJ5100jbhR6Hi9ttvz7MnR6HomWeeyQ0+opmi9D1h/QgAAAASQ48F8qRhTQoTChhvv/22/e1vf7Ny5coF1xZMvQXXX3+9vfLKK66xHxZ6p4KGVL322msu6IS9FHL++ee7wu2ChoeFweeaa67JDT4aFqVeFQAAACSOYIF86az+XXfdZfXr1w+OJKZUqVLWvHlze+SRR7Yp9Pal+1XQ0VAtUY+Knqemwt1xxx3dsYIo+Fx++eWuTkWF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77nnHtewePDBB1M61EONNTXS1KBQ8Js8ebJrXIUNiK+//to1JtXY0rCg+++/P8+ZdeIJMsgeem8/+eST7v13zTXX2FlnneVqB/73v/8VSk+AehxOOukka9KkibVt29b1DEycONEOOOAAe+KJJ+yNN95w4UFDUUP169e3QYMG2euvv+6G7ehM/1133eXuZ//997c2bdq4Y9lCr7PC0vvvv29du3Z1P4t6jfQ5wAyEAJJFsMgAv/76q/uqs2QF0T830Zmm4jh8JNkai8hC3ESoMaOCyMhGa7xbOqaYVMNbZ/U11jh8X4SiCznDLZwSM15qNKjxoJ6a888/350x1plZPd6tt94a1+QA0bNIRffe6HX57LPPXG+IHktTJu+8887bNNR03aGHHmqPPPKIa5RpiANjrIs/fW7prL96yTQLnsLFZZdd5opyb7rpJteLle4i4t133931QuirgoF61hQsNOzpuOOOcydxYtH7t3nz5u5MvYKygrPqQfbcc0/XM7fXXnsFt8xbPL2k+jyLVZuQyuJt/X3qb08ntM444wwXpHr06OF6DPr06ZNnUXTk3366epIAZC+CRQYIh4eojiJe+tDXmhcoPtSQ0PCkkIYmpTo86v40xEONh9NOO80VvKuxpIaFGhXqCbjllltcr0kqVKhQwcqWLRtcSl5BQSabqEEXWWuSzFbYxcypovffwIED3Zn/Dh06uFBbqlQpV1em2gU19G+++Wb75z//6YqK00WPecEFF9hHH32UGw40FEjH41W9evXcUPLOO++4Gfs04UamU0+F/s579eplq1atcj1G+tn13C+99FLXe6S1lPSafPDBBxl1AotZoYDMR7BAieQ7K4+2VE8xOXv2bPcP/5BDDnFnJnVGNLJxFV3IGW4aQhQP1Tw89NBDrlGns8SXX355bphVo0LBQsfUSDrnnHPsww8/TPrMsV4XDdvTkBety6JhUfpZIu9Pw/8UpHS9zharwaBhU5kocoE030a9GnHx9E4WN2oAqghafzf6XV933XXb/P2ovuzhhx927031aFx//fUlutGo1yZ6Olzf4m31VipIaSIQnTxQGFJvYUifB+q11PX6PFL4UBAGgHgRLDJAeEY3kXGtKm7Lq7s+ldSIChtUiW6cTYqfziK+99577p+9ikhPPvlkV4OgRn4qaEx5z549XaNCZ4X1e40etqH3kxr5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)

이 작업이 완료되면 다음을 실행하여 JAX와 함께 사용할 Colab TPU를 설정할 수 있습니다.


```python
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
```

다음을 실행하여 사용 가능한 TPU 기기를 확인합니다.


```python
import jax
jax.devices()
```




    [TpuDevice(id=0, host_id=0, coords=(0,0,0), core_on_chip=0),
     TpuDevice(id=1, host_id=0, coords=(0,0,0), core_on_chip=1),
     TpuDevice(id=2, host_id=0, coords=(1,0,0), core_on_chip=0),
     TpuDevice(id=3, host_id=0, coords=(1,0,0), core_on_chip=1),
     TpuDevice(id=4, host_id=0, coords=(0,1,0), core_on_chip=0),
     TpuDevice(id=5, host_id=0, coords=(0,1,0), core_on_chip=1),
     TpuDevice(id=6, host_id=0, coords=(1,1,0), core_on_chip=0),
     TpuDevice(id=7, host_id=0, coords=(1,1,0), core_on_chip=1)]



## 기본(The basics)

`jax.pmap`의 가장 기본적인 사용법은 `jax.vmap`과 완전히 유사하므로 [Vectorisation notebook](https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/03-vectorization.ipynb) 에서 다룬 컨볼루션(convolution) 예제로 돌아가보겠습니다.


```python
import numpy as np
import jax.numpy as jnp

x = np.arange(5)
w = np.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

convolve(x, w)
```




    DeviceArray([11., 20., 29.], dtype=float32)



이제 `convolve` 함수를 데이터의 전체 배치에서 실행되도록 변환해 보겠습니다. 배치를 여러 디바이스에 분산시킬 것을 대비하여 배치 크기를 디바이스의 수와 동일하게 만들겠습니다.


```python
n_devices = jax.local_device_count() 
xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)

xs
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24],
           [25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34],
           [35, 36, 37, 38, 39]])




```python
ws
```




    array([[2., 3., 4.],
           [2., 3., 4.],
           [2., 3., 4.],
           [2., 3., 4.],
           [2., 3., 4.],
           [2., 3., 4.],
           [2., 3., 4.],
           [2., 3., 4.]])



이전과 마찬가지로 `jax.vmap`을 사용하여 벡터화할 수 있습니다.


```python
jax.vmap(convolve)(xs, ws)
```




    DeviceArray([[ 11.,  20.,  29.],
                 [ 56.,  65.,  74.],
                 [101., 110., 119.],
                 [146., 155., 164.],
                 [191., 200., 209.],
                 [236., 245., 254.],
                 [281., 290., 299.],
                 [326., 335., 344.]], dtype=float32)



여러 디바이스에 계산을 분산하려면 `jax.vmap`을 `jax.pmap`으로 바꾸면 됩니다.


```python
jax.pmap(convolve)(xs, ws)
```




    ShardedDeviceArray([[ 11.,  20.,  29.],
                        [ 56.,  65.,  74.],
                        [101., 110., 119.],
                        [146., 155., 164.],
                        [191., 200., 209.],
                        [236., 245., 254.],
                        [281., 290., 299.],
                        [326., 335., 344.]], dtype=float32)



병렬화된 `convolve` 함수는 `ShardedDeviceArray`를 반환한다는 것에 유의해주세요. 이는 이 배열의 요소가 병렬 처리에 사용되는 모든 디바이스에 분산되기 때문입니다. 만약, 다른 병렬 계산을 실행하는 경우, 요소는 디바이스 간 통신 비용을 발생시키지 않고 각의 디바이스에 유지됩니다.


```python
jax.pmap(convolve)(xs, jax.pmap(convolve)(xs, ws))
```




    ShardedDeviceArray([[   78.,   138.,   198.],
                        [ 1188.,  1383.,  1578.],
                        [ 3648.,  3978.,  4308.],
                        [ 7458.,  7923.,  8388.],
                        [12618., 13218., 13818.],
                        [19128., 19863., 20598.],
                        [26988., 27858., 28728.],
                        [36198., 37203., 38208.]], dtype=float32)



내부 `jax.pmap(convolve)`의 출력은 외부 `jax.pmap(convolve)`에 입력될 때 디바이스를 떠나지 않았습니다.

## 'in_axes' 지정

`vmap`과 마찬가지로 `in_axes`를 사용하여 병렬화된 함수에 대한 인수를 브로드캐스트(`None`)할지 또는 주어진 축을 따라 분할할지 여부를 지정할 수 있습니다. 단, `vmap`과 달리 `pmap`은 이 가이드 작성 시점에서 선행 축(`0`)만 지원한다는 점에 유의하십시오.


```python
jax.pmap(convolve, in_axes=(0, None))(xs, w)
```




    ShardedDeviceArray([[ 11.,  20.,  29.],
                        [ 56.,  65.,  74.],
                        [101., 110., 119.],
                        [146., 155., 164.],
                        [191., 200., 209.],
                        [236., 245., 254.],
                        [281., 290., 299.],
                        [326., 335., 344.]], dtype=float32)



`ws`를 만들 때 `w`를 수동으로 복제한 `jax.pmap(convolve)(xs, ws)`를 사용하여 위에서 관찰한 것과 동등한 출력을 얻는 방법에 주목하십시오. 여기서는 `in_axes`에서 `None`으로 지정하여 브로드캐스팅을 통해 복제합니다.

변환된 함수를 호출할 때 인수에 지정된 축의 크기가 호스트에서 사용할 수 있는 디바이스 수를 초과하면 안 됩니다.

## `pmap`과 `jit`

`jax.pmap`은 작업의 일부로 주어진 함수를 JIT 컴파일하므로 `jax.jit`를 추가로 할 필요가 없습니다.

## 디바이스 간 통신

위의 내용으로는 단순한 병렬 연산, 예를 들어 다수의 디바이스에서 간단한 MLP 순전파를 배치하는 것을 수행하기에 충분합니다. 그러나 때로는 디바이스 간에 정보를 전달해야 합니다. 예를 들어 각 디바이스의 출력을 정규화하여 합계가 1이 되도록 하는 데 관심이 있을 수 있습니다.

이를 위해서는는 특별한 [collective ops](https://jax.readthedocs.io/en/latest/jax.lax.html#parallel-operators)(예: `jax.lax.p*` ops ` psum`, `pmean`, `pmax`, ...)를 사용할 수 있습니다. Collective ops를 사용하기 위해서는 `axis_name` 인수를 통해 `pmap`이 적용된 축의 이름을 지정한 다음 op를 호출할 때 참조해야 합니다. 방법은 다음과 같습니다.


```python
def normalized_convolution(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  output = jnp.array(output)
  return output / jax.lax.psum(output, axis_name='p')

jax.pmap(normalized_convolution, axis_name='p')(xs, ws)
```




    ShardedDeviceArray([[0.00816024, 0.01408451, 0.019437  ],
                        [0.04154303, 0.04577465, 0.04959785],
                        [0.07492582, 0.07746479, 0.07975871],
                        [0.10830861, 0.10915492, 0.10991956],
                        [0.14169139, 0.14084506, 0.14008042],
                        [0.17507419, 0.17253521, 0.17024128],
                        [0.20845698, 0.20422535, 0.20040214],
                        [0.24183977, 0.23591548, 0.23056298]], dtype=float32)



>`번역하기 어렵네요`  The `axis_name` is just a string label that allows collective operations like `jax.lax.psum` to refer to the axis bound by `jax.pmap`. It can be named anything you want -- in this case, `p`. This name is essentially invisible to anything but those functions, and those functions use it to know which axis to communicate across.

`axis_name`은 `jax.pmap`에 의해 결정된 축을 참조하기 위해 `jax.lax.psum`과 같은 집단 연산(collective operations)을 허용하는 문자열 레이블입니다. 원하는 이름으로 지정할 수 있습니다. 이 경우에는 `p`입니다. 이 이름은 본질적으로 해당 기능 외에는 보이지 않으며 해당 기능은 통신할 축을 알기 위해 이름을 사용합니다.

`jax.vmap`도 `axis_name`을 지원합니다. 이는 `jax.lax.p*`연산이 `jax.pmap`과 동일한 방식으로 `jax.lax.p*`의 벡터화 컨텍스트에서 사용될 수 있도록 합니다.


```python
jax.vmap(normalized_convolution, axis_name='p')(xs, ws)
```




    DeviceArray([[0.00816024, 0.01408451, 0.019437  ],
                 [0.04154303, 0.04577465, 0.04959785],
                 [0.07492582, 0.07746479, 0.07975871],
                 [0.10830861, 0.10915492, 0.10991956],
                 [0.14169139, 0.14084506, 0.14008042],
                 [0.17507419, 0.17253521, 0.17024128],
                 [0.20845698, 0.20422535, 0.20040214],
                 [0.24183977, 0.23591548, 0.23056298]], dtype=float32)



`jax.lax.psum`은 명명된 축(이 경우 `'p'`)이 있을 것으로 예상하기 때문에 `normalized_convolution`은 더 이상 `jax.pmap` 또는 `jax.vmap`에 의해 변환되지 않고는 더 이상 작동하지 않습니다. 이 두 변환이 하나를 결합하하는 유일한 방법입니다.


## `jax.pmap`과 `jax.vmap`의 중첩

The reason we specify `axis_name` as a string is so we can use collective operations when nesting `jax.pmap` and `jax.vmap`. For example:

`axis_name`을 문자열로 지정하는 이유는 `jax.pmap`과 `jax.vmap`을 중첩할 때 집합 연산(collective operations)을 사용할 수 있기 때문입니다. 예:  

```python
jax.vmap(jax.pmap(f, axis_name='i'), axis_name='j')
```
`f`의 `jax.lax.psum(..., axis_name='i')`은 `axis_name`을 공유하므로 pmapped 축만 참조합니다.

일반적으로 `jax.pmap` 과 `jax.vmap`은 임의의 순서로 중첩될 수 있습니다. 예를 들어 다른 `pmap` 내에 `pmap`이 있을 수 있습니다.

## 예제(Example)

다음은 각 배치가 별도의 디바이스에서 평가되는 서브 배치로 분할되는 데이터 병렬 처리가 있는 회귀 훈련 루프(regression training loop)의 예입니다.  

다음 두 가지 사항에 주의해야 합니다:
* `update()` 함수
* 매개변수(parameters) 복제 및 디바이스 간 데이터 분할.

이 예제가 너무 복잡하다면, 다음 노트북 [State in JAX](https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/07-state.ipynb)에서 병렬 처리가 없는 같은은 예제를 찾을 수 있습니다. 이 예제가 이해되면 병렬화가 어떻게 다른지 비교하여 그림이 어떻게 변경되는지 이해할 수 있습니다.


```python
from typing import NamedTuple, Tuple
import functools

class Params(NamedTuple):
  weight: jnp.ndarray
  bias: jnp.ndarray


def init(rng) -> Params:
  """Returns the initial model params."""
  weights_key, bias_key = jax.random.split(rng)
  weight = jax.random.normal(weights_key, ())
  bias = jax.random.normal(bias_key, ())
  return Params(weight, bias)


def loss_fn(params: Params, xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
  """Computes the least squares error of the model's predictions on x against y."""
  pred = params.weight * xs + params.bias
  return jnp.mean((pred - ys) ** 2)

LEARNING_RATE = 0.005

# So far, the code is identical to the single-device case. Here's what's new:


# Remember that the `axis_name` is just an arbitrary string label used
# to later tell `jax.lax.pmean` which axis to reduce over. Here, we call it
# 'num_devices', but could have used anything, so long as `pmean` used the same.
@functools.partial(jax.pmap, axis_name='num_devices')
def update(params: Params, xs: jnp.ndarray, ys: jnp.ndarray) -> Tuple[Params, jnp.ndarray]:
  """Performs one SGD update step on params using the given data."""

  # Compute the gradients on the given minibatch (individually on each device).
  loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

  # Combine the gradient across all devices (by taking their mean).
  grads = jax.lax.pmean(grads, axis_name='num_devices')

  # Also combine the loss. Unnecessary for the update, but useful for logging.
  loss = jax.lax.pmean(loss, axis_name='num_devices')

  # Each device performs its own update, but since we start with the same params
  # and synchronise gradients, the params stay in sync.
  new_params = jax.tree_map(
      lambda param, g: param - g * LEARNING_RATE, params, grads)

  return new_params, loss
```

다음은 `update()` 작동 방식입니다.

`update()`는 데코레이션이 되지않고 `pmean`이 없는 `[batch, ...]` 형태의 데이터 텐서를 가져와 해당 배치에 대한 손실 함수를 계산하고 기울기를 평가합니다.

We want to spread the `batch` dimension across all available devices. To do that, we add a new axis using `pmap`. The arguments to the decorated `update()` thus need to have shape `[num_devices, batch_per_device, ...]`. So, to call the new `update()`, we'll need to reshape data batches so that what used to be `batch` is reshaped to `[num_devices, batch_per_device]`. That's what `split()` does below. Additionally, we'll need to replicate our model parameters, adding the `num_devices` axis. This reshaping is how a pmapped function knows which devices to send which data.

사용 가능한 모든 디바이스에 'batch' 차원을 확산하려고 합니다. 이를 위해 `pmap`을 사용하여 새 축을 추가합니다. 따라서 데코레이션된 `update()`에 대한 인수는 `[num_devices, batch_per_device, ...]` 형태를 가져야 합니다. 따라서 새로운 `update()`를 호출하려면 `batch`였던 것이 `[num_devices, batch_per_device]`로 재구성되도록 데이터 배치를 재구성해야 합니다. 이것이 `split()`이 아래에서 하는 일입니다. 또한 모델 매개변수를 복제하여 `num_devices` 축을 추가해야 합니다. 이 재구성은 매핑된 함수가 어떤 디바이스에 어떤 데이터를 보낼지 아는 방법입니다.

업데이트 단계 중 어느 시점에서 각 디바이스에서 계산된 그래디언트를 결합해야 합니다. 그렇지 않으면 각 디바이스에서 수행하는 업데이트가 달라집니다. 그래서 `jax.lax.pmean`을 사용하여 `num_devices` 축 전체의 평균을 계산하여 배치의 평균 그래디언트를 제공합니다. 그 평균 그래디언트는 우리가 업데이트를 계산하는 데 사용하는 것입니다.

이름을 짓는 것 외에도, 여기서는 `jax.pmap`을 도입하는 동안 교훈적인 명확성을 위해 `axis_name`에 `num_devices`를 사용합니다. 그러나 어떤 점에서는 자기 반증적(tautologous)입니다. pmap에 의해 도입된 모든 축은 여러 디바이스를 나타냅니다. 따라서 'batch', 'data'(데이터 병렬화를 나타냄) 또는 'model'(모델 병렬화를 나타냄)과 같이 의미상 의미 있는 것으로 축 이름을 지정하는 것이 일반적입니다.


```python
# Generate true data from y = w*x + b + noise
true_w, true_b = 2, -1
xs = np.random.normal(size=(128, 1))
noise = 0.5 * np.random.normal(size=(128, 1))
ys = xs * true_w + true_b + noise

# Initialise parameters and replicate across devices.
params = init(jax.random.PRNGKey(123))
n_devices = jax.local_device_count()
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
```

지금까지는 선행 차원이 추가된 배열을 구성했습니다. 매개변수는 여전히 모두 호스트(CPU)에 있습니다. `pmap`은 `update()`가 처음 호출될 때 디바이스에 이를 전달하고 각 복사본은 이후에 자체 디바이스에 유지됩니다. ShardedDeviceArray가 아니라 DeviceArray이기 때문에 알 수 있습니다.


```python
type(replicated_params.weight)
```




    jax.interpreters.xla._DeviceArray



매개변수(params)는 맵핑된 `update()`에서 반환되면 ShardedDeviceArray로 변환될 것입니다. (자세한 내용은 뒷부분을 참고하세요).

데이터에 대해서도 동일한 작업을 수행합니다.


```python
def split(arr):
  """Splits the first axis of `arr` evenly across the number of devices."""
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

# Reshape xs and ys for the pmapped `update()`.
x_split = split(xs)
y_split = split(ys)

type(x_split)
```




    numpy.ndarray



데이터는 재구성된 바닐라 NumPy 배열일 뿐입니다. 따라서 NumPy는 CPU에서만 실행되므로 호스트 이외의 위치에 있을 수 없습니다. 우리는 그것을 수정하지 않기 때문에 데이터가 일반적으로 각 단계에서 CPU에서 디바이스로 스트리밍되는 실제 파이프라인에서와 같이 각 `update` 호출마다 디바이스로 전송됩니다.  


```python
def type_after_update(name, obj):
  print(f"after first `update()`, `{name}` is a", type(obj))

# Actual training loop.
for i in range(1000):

  # This is where the params and data gets communicated to devices:
  replicated_params, loss = update(replicated_params, x_split, y_split)

  # The returned `replicated_params` and `loss` are now both ShardedDeviceArrays,
  # indicating that they're on the devices.
  # `x_split`, of course, remains a NumPy array on the host.
  if i == 0:
    type_after_update('replicated_params.weight', replicated_params.weight)
    type_after_update('loss', loss)
    type_after_update('x_split', x_split)

  if i % 100 == 0:
    # Note that loss is actually an array of shape [num_devices], with identical
    # entries, because each device returns its copy of the loss.
    # So, we take the first element to print it.
    print(f"Step {i:3d}, loss: {loss[0]:.3f}")


# Plot results.

# Like the loss, the leaves of params have an extra leading dimension,
# so we take the params from the first device.
params = jax.device_get(jax.tree_map(lambda x: x[0], replicated_params))
```

    after first `update()`, `replicated_params.weight` is a <class 'jax.interpreters.pxla.ShardedDeviceArray'>
    after first `update()`, `loss` is a <class 'jax.interpreters.pxla.ShardedDeviceArray'>
    after first `update()`, `x_split` is a <class 'numpy.ndarray'>
    Step   0, loss: 0.228
    Step 100, loss: 0.228
    Step 200, loss: 0.228
    Step 300, loss: 0.228
    Step 400, loss: 0.228
    Step 500, loss: 0.228
    Step 600, loss: 0.228
    Step 700, loss: 0.228
    Step 800, loss: 0.228
    Step 900, loss: 0.228



```python
import matplotlib.pyplot as plt
plt.scatter(xs, ys)
plt.plot(xs, params.weight * xs + params.bias, c='red', label='Model Prediction')
plt.legend()
plt.show()
```


    
![png](JAX_Parallelism_%ED%98%84%EC%95%84_files/JAX_Parallelism_%ED%98%84%EC%95%84_41_0.png)
    


## Aside: JAX에서의 호스트와 디바이스

TPU에서 실행할 때 '호스트'라는 개념이 중요해집니다. 호스트는 여러 디바이스를 관리하는 CPU입니다. 단일 호스트가 관리할 수 있는 디바이스 수는 8개(보통 8개)만 관리할 수 있으므로 대규모 병렬 프로그램을 실행할 때 여러 호스트가 필요하며 이를 관리하려면 약간의 기술이 필요합니다. 


```python
jax.devices()
```




    [TpuDevice(id=0, host_id=0, coords=(0,0,0), core_on_chip=0),
     TpuDevice(id=1, host_id=0, coords=(0,0,0), core_on_chip=1),
     TpuDevice(id=2, host_id=0, coords=(1,0,0), core_on_chip=0),
     TpuDevice(id=3, host_id=0, coords=(1,0,0), core_on_chip=1),
     TpuDevice(id=4, host_id=0, coords=(0,1,0), core_on_chip=0),
     TpuDevice(id=5, host_id=0, coords=(0,1,0), core_on_chip=1),
     TpuDevice(id=6, host_id=0, coords=(1,1,0), core_on_chip=0),
     TpuDevice(id=7, host_id=0, coords=(1,1,0), core_on_chip=1)]



CPU에서 실행할 때는 `--xla_force_host_platform_device_count` XLA 플래그를 사용하여 임의의 수의 디바이스를 에뮬레이션할 수 있습니다. 예를 들어 JAX를 가져오기 전에 다음을 실행하면 됩니다.  
```python
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
jax.devices()
```
```
[CpuDevice(id=0),
 CpuDevice(id=1),
 CpuDevice(id=2),
 CpuDevice(id=3),
 CpuDevice(id=4),
 CpuDevice(id=5),
 CpuDevice(id=6),
 CpuDevice(id=7)]
```
이것은 CPU 런타임이 (재)시작하는 것이 더 빠르기 때문에 로컬에서 디버깅 및 테스트하거나 Colab에서 프로토타이핑하는 데 특히 유용합니다.  


```python

```
