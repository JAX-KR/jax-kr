# JAX로 다중-GPU 분산학습 진행하기 (Multi-GPU distributed training with JAX)

<a href="https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/06-parallelism.ipynb" target="_parent"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle Notebook"/></a>

**저자:** [fchollet](https://twitter.com/fchollet)<br>
**역자:** [이영빈](https://github.com/mezcalagave)\
**생성일자:** 2023/07/11<br>
**최근수정일자:** 2023/07/11<br>
**번역일자:** 2023/10/15\
**개요:** JAX를 사용한 케라스 모델을 멀티 GPU/TPU 학습하기

## 역자설명
본 노트북의 경우 Kaggle Notebook으로 작성했습니다 Kaggle Notebook의 경우 다중 GPU(T4 * 2개)를 지원하기 때문에 병렬학습이 가능합니다. 다만 데이터를 올려놓아야 하기 때문에 케라스에서 mnist.npz를 다운받고 Add Data해주세요!

## 시작하며


일반적으로 여러 기기를 사용해서 분산학습하는 방법에는 크게 2가지가 있습니다:

**데이터 병렬처리** : 단일 모델이 여러개의 디바이스나 머신에 복제되는 방식입니다. 각 모델은 서로 다른 데이터 배치를 처리하고 결과를 병합합니다. 데이터 병렬처리는 여러 모델 복제본이 결과를 병합하는 방식도 있으며 모든 배치 상태에서 동기화 상태를 유지하는 방식이 있을수도 있고 동기화를 유지하지 않는 방식도 있습니다.

**모델 병렬처리** : 단일 모델의 여러 부분이 서로 다른 장치에서 실행되며 단일 데이터 배치를 함께 처리하는 방식입니다. 이 방식의 경우 병렬처리가 용이한 모델에서 잘 작동합니다.

이번 예제에서는 데이터 병렬처리에 중점을 두며 특히 모델 여러개의 복제본이 처리할 때마다 동기화 상태를 유지시키는 예제입니다. 동기화는 모델을 병합할 때 하나의 GPU에서 볼 수 있는 것과 같이 동일하게 유지합니다.

특히 이번 예제에서는 jax.sharding API를 활용해 코드 변경을 최소화시키고 단일 머신에 설치된 여러개의 GPU 혹은 TPU에서 케라스 모델을 훈련하는 방법을 설명합니다. 이 방식은 연구자들이나 소규모 워크플로우에서 가장 많이 사용하는 방식입니다.

## 설치

우선 JAX와 케라스 코어를 먼저 설치합니다.


```python
!pip install jax jaxlib
!pip install keras-core
```

    Requirement already satisfied: jax in /opt/conda/lib/python3.10/site-packages (0.4.13)
    Requirement already satisfied: jaxlib in /opt/conda/lib/python3.10/site-packages (0.4.13+cuda11.cudnn86)
    Requirement already satisfied: ml-dtypes>=0.1.0 in /opt/conda/lib/python3.10/site-packages (from jax) (0.2.0)
    Requirement already satisfied: numpy>=1.21 in /opt/conda/lib/python3.10/site-packages (from jax) (1.23.5)
    Requirement already satisfied: opt-einsum in /opt/conda/lib/python3.10/site-packages (from jax) (3.3.0)
    Requirement already satisfied: scipy>=1.7 in /opt/conda/lib/python3.10/site-packages (from jax) (1.11.2)
    Requirement already satisfied: keras-core in /opt/conda/lib/python3.10/site-packages (0.1.5)
    Requirement already satisfied: absl-py in /opt/conda/lib/python3.10/site-packages (from keras-core) (1.4.0)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (from keras-core) (1.23.5)
    Requirement already satisfied: rich in /opt/conda/lib/python3.10/site-packages (from keras-core) (13.4.2)
    Requirement already satisfied: namex in /opt/conda/lib/python3.10/site-packages (from keras-core) (0.0.7)
    Requirement already satisfied: h5py in /opt/conda/lib/python3.10/site-packages (from keras-core) (3.9.0)
    Requirement already satisfied: dm-tree in /opt/conda/lib/python3.10/site-packages (from keras-core) (0.1.8)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/conda/lib/python3.10/site-packages (from rich->keras-core) (2.2.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/conda/lib/python3.10/site-packages (from rich->keras-core) (2.15.1)
    Requirement already satisfied: mdurl~=0.1 in /opt/conda/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich->keras-core) (0.1.0)


Kaggle에 mnist.npz를 압축해제합니다.


```python
!unzip /kaggle/input/mnist-npz/mnist.npz
```

    Archive:  /kaggle/input/mnist-npz/mnist.npz
      inflating: x_test.npy              
      inflating: x_train.npy             
      inflating: y_train.npy             
      inflating: y_test.npy              




## 환경세팅

먼저 학습할 모델을 생성하는 함수와 학습할 데이터 세트를 생성하는 함수를 정의해 보겠습니다.


```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import numpy as np
import tensorflow as tf
import keras_core as keras

from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def get_model():
    # 배치 정규화와 드랍아웃이 들어간 간단한 컨볼루션 네트워크를 만듭니다.
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(filters=12, kernel_size=3, padding="same", use_bias=False)(
        x
    )
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=24,
        kernel_size=6,
        use_bias=False,
        strides=2,
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=6,
        padding="same",
        strides=2,
        name="large_k",
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model


def get_datasets():
    # 학습 데이터와 테스트 데이터를 로드해주세요.
    x_train = np.load('/kaggle/working/x_train.npy')
    y_train = np.load('/kaggle/working/y_train.npy')
    x_test = np.load('/kaggle/working/x_test.npy')
    y_test = np.load('/kaggle/working/y_test.npy')

    # 이미지들을 [0,1] 범위로 만들어줍니다.
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # 이미지의 크기가 반드시 (28,28,1)인지 잊지 말아주세요!
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # TF 데이터셋으로 만들어주세요
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    eval_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, eval_data

```

## 단일 호스트, 다양한 디바이스를 사용한 동기화된 학습방법

이번에 사용하는 환경은 하나의 머신에 2개의 T4가 되어 있습니다. 각 GPU에는 모델의 복사본을 실행하게 됩니다.

**동작하는 원리**

학습하는 단계는 다음과 같습니다:

- 현재 데이터 배치(**글로벌 배치**라고 함)는 2개의 서로 다른 하위 배치(**로컬 배치**라고 함)로 분할됩니다. 예를 들어, 글로벌 배치에 512개의 샘플이 있는 경우, 2개의 로컬 배치 각각에는 256개의 샘플이 있습니다.
- 2개의 복제본은 각각 로컬 배치를 독립적으로 처리하며, 순방향 전달(forward pass)를 실행한 다음 역방향전달(backward pass)를 실행하여 로컬 배치에서 모델의 손실에 대한 가중치의 기울기를 출력합니다.
- 로컬 그레디언트에서 시작된 가중치 업데이트는 2개의 복제본에서 효율적으로 병합됩니다. 이 작업은 모든 단계의 마지막에 수행되므로 복제본은 항상 동기화 상태를 유지합니다.

이번 예제에서 모델 복제본의 가중치를 동기화해서 업데이트하는 방식은 각 개별 가중치 변수 수준에서 처리되빈다. 이번 예제에서는 `jax.sharding.NamedSharding`을 사용해서 변수를 복제하도록 구성합니다.


**사용방법**

케라스 모델로 단일 호스트, 다중 디바이스 동기화된 학습방법을 사용하기 위해서는 `jax.sharding` 기능을 이용해야 합니다.\
사용방법은 다음과 같습니다: 

- 먼저 `mesh_utils.create_device_mesh`를 사용하여 디바이스 메시를 생성합니다.
- `jax.sharding.Mesh`, `jax.sharding.NamedSharding`, `jax.sharding.PartitionSpec`을 사용하여 JAX Array를 분할하는 방법을 정의합니다.
  - 축이 아닌 다른 사양으로 파티션을 사용하여 모든 장치에서 모델 및 최적화 프로그램 변수를 복제하도록 지정합니다.
  - 배치 차원을 따라 분할하는 사양을 사용하여 여러 기기에서 데이터를 분할하도록 지정합니다.
- `jax.device_put`을 사용하여 모델 및 옵티마이저 변수를 기기 간에 복제합니다. 이 작업은 처음에 한 번 수행됩니다.
- 훈련 루프에서 처리하는 각 배치에 대해 `jax.device_put`을 사용하여 훈련 단계를 호출하기 전에 배치를 여러 디바이스에 분할합니다.

다음은 각 단계가 자체 유틸리티 함수로 분할되는 방식입니다.


```python
# 하이퍼 파라미터 세팅
num_epochs = 2
batch_size = 64

train_data, eval_data = get_datasets()
train_data = train_data.batch(batch_size, drop_remainder=True)

model = get_model()
optimizer = keras.optimizers.Adam(1e-3)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# build()를 활용해 모든 상태를 초기화한다.
(one_batch, one_batch_labels) = next(iter(train_data))
model.build(one_batch)
optimizer.build(model.trainable_variables)


# 해당 함수는 미분이 되는 손실함수입니다.
# 케라스는 순수 함수형으로 이루어진 순방향 전달을 지원합니다 : model.stateless_call
def compute_loss(trainable_variables, non_trainable_variables, x, y):
    y_pred, updated_non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss_value = loss(y, y_pred)
    return loss_value, updated_non_trainable_variables


# 그레디언트를 계산하는 함수입니다.
compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)


# train_step 함수 만들기. 케라스는 순수 함수형인 optimizer.stateless_apply를 제공하고 있습니다.
@jax.jit
def train_step(train_state, x, y):
    trainable_variables, non_trainable_variables, optimizer_variables = train_state
    (loss_value, non_trainable_variables), grads = compute_gradients(
        trainable_variables, non_trainable_variables, x, y
    )

    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )

    return loss_value, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )

# 모든 디바이스에서 모델과 최적화 함수 변수를 복제합니다.
def get_replicated_train_state(devices):
    # 모든 변수는 모든 디바이스에서 복제됩니다.
    var_mesh = Mesh(devices, axis_names=("_"))
    # NamedShading에서 언급되지 않는 축이 복제됩니다 (여기에서는 모든 축입니다.)
    var_replication = NamedSharding(var_mesh, P())

    # 모델 변수들을 분산환경에 적용합니다.
    trainable_variables = jax.device_put(model.trainable_variables, var_replication)
    non_trainable_variables = jax.device_put(
        model.non_trainable_variables, var_replication
    )
    optimizer_variables = jax.device_put(optimizer.variables, var_replication)

    # 튜플 1개에 모든 상태를 합칩니다.
    return (trainable_variables, non_trainable_variables, optimizer_variables)


num_devices = len(jax.local_devices())
print(f"Running on {num_devices} devices: {jax.local_devices()}")
devices = mesh_utils.create_device_mesh((num_devices,))

# 데이터는 배치 축으로 분할됩니다.
data_mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
data_sharding = NamedSharding(
    data_mesh,
    P(
        "batch",
    ),
)  # 샤딩된 파티션의 툭의 이름을 지정합니다.

# 데이터 샤딩을 시각화합니다.
x, y = next(iter(train_data))
sharded_x = jax.device_put(x.numpy(), data_sharding)
print("Data sharding")
jax.debug.visualize_array_sharding(jax.numpy.reshape(sharded_x, [-1, 28 * 28]))

train_state = get_replicated_train_state(devices)

# 커스터마이징된 훈련 루프를 만듭니다.
for epoch in range(num_epochs):
    data_iter = iter(train_data)
    for data in data_iter:
        x, y = data
        sharded_x = jax.device_put(x.numpy(), data_sharding)
        loss_value, train_state = train_step(train_state, sharded_x, y.numpy())
    print("Epoch", epoch, "loss:", loss_value)

# 모델에 적용하기 위해 모델 상태 업데이트를 후처리합니다.
trainable_variables, non_trainable_variables, optimizer_variables = train_state
for variable, value in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for variable, value in zip(model.non_trainable_variables, non_trainable_variables):
    variable.assign(value)
```

    x_train shape: (60000, 28, 28, 1)
    60000 train samples
    10000 test samples
    Running on 2 devices: [gpu(id=0), gpu(id=1)]
    Data sharding



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                     GPU 0                                      </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                     GPU 1                                      </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">                                                                                </span>
</pre>



    Epoch 0 loss: 0.49051714
    Epoch 1 loss: 0.5971312



```python

```


```python

```
