# JAX로 `fit()`에서 일어나는 일 사용자 정의하기(Customizing what happens in fit() with JAX)

**저자:** [fchollet](https://twitter.com/fchollet)<br>
**역자:** [조현석](mailto:hoyajigi@gmail.com)<br>
**검수:** 이영빈, 박정현<br>
**생성 날짜:** 2023/06/27<br>
**마지막 수정:** 2023/06/27<br>
**설명:** 모델 클래스의 훈련 단계를 JAX로 재정의합니다.

<a href="https://colab.research.google.com/drive/1sSz6_fi8S0OHn3T_73046sI2P1rB4-xy" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


## 소개

지도 학습을 할 때 `fit()`을 사용하면 모든 것이 원활하게 작동합니다.

모든 세부 사항을 제어해야 하는 경우, 자신만의 훈련
루프를 완전히 처음부터 작성할 수 있습니다.

하지만 사용자 지정 학습 알고리즘이 필요하지만 여전히
콜백, 기본 제공 배포 지원과 같은 `fit()`의 편리한 기능을 활용하고 싶다면 어떻게 해야 할까요?
또는 단계 융합과 같은 편리한 기능을 활용하고 싶으신가요?

Keras의 핵심 원칙은 **복잡성의 점진적 공개**입니다. 사용자는
항상 점진적인 방식으로 로우 레벨의 워크플로우에 진입할 수 있어야 합니다.
하이 레벨의 기능이 사용 사례와 정확히 일치하지 않는다고 해서 갑자기 로우 레벨로 바뀌면 안 됩니다. 높은 수준의 편의성을 유지하면서 작은 세부 사항을 더 잘 제어할 수 있어야 합니다.

`fit()`의 기능을 사용자 정의해야 하는 경우, `Model` 클래스의 훈련 단계 함수를 **재정의해야 합니다**. 이 함수는 모든 데이터 배치에 대해 `fit()`에 의해 호출되는 함수입니다. 그러면 평소처럼 `fit()`을 호출할 수 있으며, 자체 학습 알고리즘이 실행됩니다.

이 패턴은 함수형 API로 모델을 빌드하는 것을 방해하지 않습니다. '시퀀셜' 모델, 함수형 API 모델 또는 하위 클래스 모델을 빌드하든 상관없이 이 작업을 수행할 수 있습니다.

어떻게 작동하는지 살펴보겠습니다.

## 설정


```python
import os

# 이 가이드는 JAX 백엔드에서만 실행할 수 있습니다.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras_core as keras
import numpy as np
```

    Using JAX backend.


## 첫 번째 간단한 예시

간단한 예제부터 시작하겠습니다:

- 우리는 `keras.Model`을 상속하는 새로운 클래스를 생성합니다.
- 모델의 비학습 가능 변수에 대한 업데이트된 값과 손실을 계산하기 위해 완전 스테이트리스 `compute_loss_and_updates()` 메서드를 구현합니다. 내부적으로는 `stateless_call()`과 내장된 `compute_loss()`를 호출합니다.
- 완전 스테이트리스 `train_step()` 메서드를 구현하여 현재 메트릭 값(손실 포함)과 학습 가능한 변수, 옵티마이저 변수, 메트릭 변수에 대한 업데이트된 값을 계산합니다.

참고로 `sample_weight` 인수를 다음과 같이 고려할 수도 있습니다:

- 데이터를 `x, y, sample_weight = data`로 언패킹합니다.
- sample_weight`를 `compute_loss()`에 전달합니다.
- sample_weight`를 `y` 및 `y_pred`와 함께 전달합니다.
와 함께 `stateless_update_state()`의 메트릭에 전달하기


```python
class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        x,
        y,
        training=False,
    ):
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            training=training,
        )
        loss = self.compute_loss(x, y, y_pred)
        return loss, (y_pred, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y = data

        # 그라데이션 함수를 가져옵니다.
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        # 그라디언트를 계산합니다.
        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            x,
            y,
            training=True,
        )

        # 학습 가능한 변수 및 최적화 변수 업데이트합니다.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # 메트릭을 업데이트합니다.
        new_metrics_vars = []
        for metric in self.metrics:
            this_metric_vars = metrics_variables[
                len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)
            ]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars, y, y_pred
                )
            logs = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        # 메트릭 로그와 업데이트된 상태 변수를 반환합니다.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state

```

한번 사용해 보겠습니다:


```python
# CustomModel의 인스턴스를 생성하고 컴파일합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 평소처럼 'fit'을 사용하세요.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)

```

    Epoch 1/3
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - mae: 0.4484 - loss: 0.2870
    Epoch 2/3
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - mae: 0.4020 - loss: 0.2704
    Epoch 3/3
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - mae: 0.4182 - loss: 0.2542





    <keras_core.src.callbacks.history.History at 0x7be1c8068400>



## 로우 레벨로 해보기

당연히 `compile()`에서 손실 함수를 전달하는 것을 건너뛰고 대신 `train_step`에서
모든 것을 *수동으로* 할 수 있습니다. 메트릭도 마찬가지입니다.

다음은 `compile()`만 사용하여 옵티마이저를 구성하는 로우 레벨의 예제입니다:


```python
class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_fn = keras.losses.MeanSquaredError()

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        x,
        y,
        training=False,
    ):
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            training=training,
        )
        loss = self.loss_fn(y, y_pred)
        return loss, (y_pred, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y = data

        # 그라데이션 함수를 가져옵니다.
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        # 그라디언트를 계산합니다.
        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            x,
            y,
            training=True,
        )

        # 학습 가능한 변수 및 최적화 변수 업데이트합니다.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # 메트릭을 업데이트합니다.
        loss_tracker_vars = metrics_variables[: len(self.loss_tracker.variables)]
        mae_metric_vars = metrics_variables[len(self.loss_tracker.variables) :]

        loss_tracker_vars = self.loss_tracker.stateless_update_state(
            loss_tracker_vars, loss
        )
        mae_metric_vars = self.mae_metric.stateless_update_state(
            mae_metric_vars, y, y_pred
        )

        logs = {}
        logs[self.loss_tracker.name] = self.loss_tracker.stateless_result(
            loss_tracker_vars
        )
        logs[self.mae_metric.name] = self.mae_metric.stateless_result(mae_metric_vars)

        new_metrics_vars = loss_tracker_vars + mae_metric_vars

        # 메트릭 로그와 업데이트된 상태 변수를 반환합니다.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state

    @property
    def metrics(self):
        # 여기에 `Metric` 객체를 나열하여 `reset_states()`가
        # 각 에포크가 시작될 때 자동으로 호출되거나
        # 또는 `evaluate()`가 시작될 때 자동으로 호출될 수 있도록 합니다.
        return [self.loss_tracker, self.mae_metric]


# CustomModel의 인스턴스를 생성하고 컴파일합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)

# 여기서는 손실이나 지표를 전달하지 않습니다.
model.compile(optimizer="adam")

# 평소처럼 `fit`을 사용하면 됩니다. 콜백 등을 사용할 수 있습니다.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)

```

    Epoch 1/5
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 17ms/step - loss: 0.3712 - mae: 0.4860
    Epoch 2/5
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2637 - mae: 0.4173
    Epoch 3/5
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2385 - mae: 0.4012
    Epoch 4/5
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - loss: 0.2407 - mae: 0.3952
    Epoch 5/5
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 0.2145 - mae: 0.3782





    <keras_core.src.callbacks.history.History at 0x7be1b02f3a90>



## 자체 평가 단계 제공

`model.evaluate()` 호출에 대해 동일한 작업을 수행하려면 어떻게 해야 할까요? 그렇다면
test_step`을 정확히 같은 방식으로 재정의하면 됩니다. 이렇게 하면 됩니다:


```python
class CustomModel(keras.Model):
    def test_step(self, state, data):
        # 데이터 언팩킹합니다.
        x, y = data
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state

        # 예측과 손실을 계산합니다.
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            training=False,
        )
        loss = self.compute_loss(x, y, y_pred)

        # 메트릭을 업데이트합니다.
        new_metrics_vars = []
        for metric in self.metrics:
            this_metric_vars = metrics_variables[
                len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)
            ]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars, y, y_pred
                )
            logs = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        # 메트릭 로그와 업데이트된 상태 변수를 반환합니다.
        state = (
            trainable_variables,
            non_trainable_variables,
            new_metrics_vars,
        )
        return logs, state


# CustomModel의 인스턴스를 생성합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(loss="mse", metrics=["mae"])

# 사용자 정의 test_step으로 평가하기
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.evaluate(x, y)

```

    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - mae: 0.6693 - loss: 0.6276





    [0.6276098489761353, 0.6762693524360657]



이게 전부입니다!
