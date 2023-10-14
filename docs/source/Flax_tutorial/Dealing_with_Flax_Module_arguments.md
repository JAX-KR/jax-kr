# Flax 모듈 인자 다루기 (Dealing with Flax Module arguments)
<a href="https://drive.google.com/file/d/1aAD1fxTQiETSXEO9dp9UGBQNpKWxVbgy/view?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


번역: 조영빈     


Flax Linen에서는 `Module` 인자를 데이터 클래스 속성으로 정의하거나 (일반적으로 `__call__` 메서드의 인자로) 메서드에 전달할 수 있습니다. 일반적으로 구분은 명확합니다:

* 커널 이니셜라이저의 선택 또는 출력 피쳐 수와 같은 완전히 고정된 속성은 하이퍼파라미터이며 데이터 클래스 속성으로 정의되어야 합니다. 일반적으로 서로 다른 하이퍼파라미터를 가진 두 개의 모듈 인스턴스는 의미 있는 방식으로 공유할 수 없습니다.

* 입력 데이터 및 `train=True/False`와 같은 상위 수준의 "모드 전환"과 같은 동적 속성은 `__call__` 또는 다른 메서드에 인자로 전달해야 합니다.

그러나 일부 경우에는 구분이 명확하지 않을 수 있습니다. 예를 들어 Dropout 모듈을 살펴보겠습니다. 몇 가지 명확한 하이퍼파라미터가 있습니다:

1. 드랍아웃 비율
2. 드랍아웃 마스크가 생성되는 축

그리고 명확한 호출 시간 인자도 있습니다:

1. 드랍아웃을 사용하여 마스킹 할 입력 데이터
2. (선택적인) 무작위 마스크를 샘플링하는 데 사용되는 rng

그러나 Dropout 모듈에서 모호한 하나의 속성이 있습니다 - Dropout 모듈에서의 `deterministic` 속성입니다.

`deterministic`이 `True`이면 드롭아웃 마스크가 샘플되지 않습니다. 이는 일반적으로 모델 평가 중에 사용됩니다. 그러나 `eval=True` 또는 `train=False`를 최상위 모듈에 전달하면 `deterministic` 인자는 모든 곳에 적용되어 부울 인자가 `Dropout`을 사용할 수 있는 모든 레이어로 전달되어야 합니다. `deterministic`이 데이터 클래스 속성인 경우 다음과 같이 수행할 수 있습니다:


```python
from functools import partial
from flax import linen as nn

class ResidualModel(nn.Module):
  drop_rate: float

  @nn.compact
  def __call__(self, x, *, train):
    dropout = partial(nn.Dropout, rate=self.drop_rate, deterministic=not train)
    for i in range(10):
    #   x += ResidualBlock(dropout=dropout, ...)(x)
      x += ResidualBlock(dropout=dropout)(x)
```

여기서 `determinstic`을 생성자에 전달하는 것은 의미가 있습니다. 이렇게 하면 드롭아웃 템플릿을 하위 모듈에 전달할 수 있습니다. 이제 하위 모듈은 train 대 eval 모드를 처리할 필요가 없으며 간단히 `dropout` 인자를 사용할 수 있습니다. 드롭아웃 레이어는 하위 모듈에서만 구성될 수 있으므로 생성자에는 `determinstic`을 부분적으로 적용할 수 있지만 `__call__`에는 적용할 수 없습니다.

그러나 `determinstic`이 데이터 클래스 속성인 경우, setup 패턴을 사용할 때 문제가 발생합니다. 모듈 코드를 다음과 같이 작성하려고 할 것입니다:


```python
class SomeModule(nn.Module):
  drop_rate: float

  def setup(self):
    self.dropout = nn.Dropout(rate=self.drop_rate)

  @nn.compact
  def __call__(self, x, *, train):
    # ...
    x = self.dropout(x, deterministic=not train)
    # ...
```

하지만 위에서 정의한대로 `deterministic`은 속성이므로 이는 작동하지 않습니다. 여기서는 `train` 인자에 따라 결정되므로 `__call__` 중에 `deterministic`을 전달하는 것이 합리적입니다.

# Solution

이전에 설명한 두 가지 사용 사례를 모두 지원하기 위해 특정 속성을 데이터 클래스 속성이나 메서드 인자로 전달할 수 있도록 하여 (하지만 둘 다는 불가능합니다!) 구현할 수 있습니다. 다음과 같이 구현할 수 있습니다:


```python
from typing import Optional

class MyDropout(nn.Module):
  drop_rate: float
  deterministic: Optional[bool] = None

  @nn.compact
  def __call__(self, x, deterministic=None):
    deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
    # ...
```

이 예제에서 `nn.merge_param`은 `self.deterministic` 또는 `deterministic` 중 하나가 설정되도록 하지만 둘 다 설정되지 않도록 보장합니다. 두 값이 모두 `None`이거나 모두 `None`이 아닌 경우 오류가 발생합니다. 이를 통해 코드의 두 부분이 동일한 매개변수를 설정하고 하나가 다른 것에 의해 무시되는 혼란스러운 동작을 피할 수 있습니다. 또한, 기본값을 피하면 기본적으로 훈련 절차의 훈련 단계 또는 평가 단계가 손상될 수 있습니다.

# Functional Core


함수형 코어(Functional core)는 클래스 대신 함수를 정의합니다. 따라서 하이퍼파라미터와 호출 시점 인자 간에 명확한 구분이 없습니다. 하이퍼파라미터를 사전에 결정하는 유일한 방법은 `partial`을 사용하는 것입니다. 이러한 접근 방식의 장점은 메서드 인자가 속성일 수 있는 모호한 경우가 없다는 것입니다.
