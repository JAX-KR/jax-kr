---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>

<a href="https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/full_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>



번역: 유현아     



원본 : https://flax.readthedocs.io/en/latest/guides/full_eval.html


# 전체 데이터셋 처리



효율성을 위해, 여러 예제가 포함된 배치들(batches)을 구성하여 병렬로 처리합니다. 특히 모델을 평가할 때,모든 예제를 처리하고 **나머지 예제들이 손실되지 않도록** 마지막에 완전한 배치를 형성하지 않도록 하는 것이 중요합니다.


## 문제(The problem)



단일 디바이스에서 평가할 때, 마지막 불완전한 배치를 삭제하거나 이전 배치와 다른 형태로 마지막 배치를 형성할 수 있습니다. 후자를 사용하면 XLA가 모양 다형성(shape polymorphic)이 아니기 때문에 `eval_step()`을 **다시 컴파일**해야 한다는 단점이 있습니다.



> [다형성(shape polymorphic)](https://ko.wikipedia.org/wiki/%EB%8B%A4%ED%98%95%EC%84%B1_(%EC%BB%B4%ED%93%A8%ED%84%B0_%EA%B3%BC%ED%95%99))  
> 하나의 객체가 여러가지 타입을 갖는 것



```python
collections.Counter(
    tuple(batch['image'].shape)
    for batch in tfds.load('mnist', split='test').batch(per_device_batch_size)
)
# output:
# Counter({(272, 28, 28, 1): 1, (512, 28, 28, 1): 19})
```

이 문제는 데이터 병렬 처리를 위해 여러 장치를 사용할 때 더욱 두드러집니다. 배치 크기를 **디바이스 수로 나눌 수 없는 경우**, 마지막 단계는 단일 디바이스(또는 디바이스의 하위 집합)에서 실행해야 합니다. 일반적으로 마지막 배치를 삭제하지만 잘못된 결과를 초래할 수 있습니다.



```python
sum(
    np.prod(batch['label'].shape)
    for batch in tfds.load('mnist', split='test')
        .batch(per_device_batch_size, drop_remainder=True)
        .batch(jax.local_device_count())
)
# output:
# 9728
```

여러 호스트를 사용하면 JAX가 SPMD 패러다임을 사용하고 모든 호스트가 동일한 프로그램을 실행해야 하기 때문에 상황이 더욱 복잡해집니다. 일반적으로 tfds.split_for_jax_process()를 사용하여 서로 다른 호스트에 대해 겹치지 않는 분할을 형성하지만, 이렇게 하면 호스트마다 다른 숫자가 발생하여 모든 예제를 처리해야 할 때 서로 다른 JAX 프로그램이 생성될 수 있습니다.



```python
process_count = 6
[
    len(tfds.load(dataset_name, split=tfds.split_for_jax_process(
        'test', process_index=process_index, process_count=process_count)))
    for process_index in range(process_count)
]
# output:
# [1667, 1667, 1667, 1667, 1666, 1666]
```

## 해결책 : 패딩(The solution: padding)  



서로 다른 호스트마다 다른 디바이스에서 실행되는 배치의 수를 영리하게 조정하여 이 문제를 해결할 수는 있지만, 이러한 솔루션은 빠르게 복잡해지고 번거로운 로직이 많아 메인 평가 루프를 읽기 어렵게 만듭니다.



이 문제에 대한 보다 간단한 해결책은 데이터셋의 끝에 패딩을 사용하여 마지막 배치가 이전 배치와 동일한 크기를 갖도록 하는 것입니다.


## 메뉴 구현(Manual implementation)

마지막 배치에는 이전 배치와 동일한 수의 예제가 포함되도록 수동으로 패딩됩니다. 패딩된 예제에 대한 예측은 계산에서 삭제됩니다.



```python
shard = lambda x: einops.rearrange(
    x, '(d b) ... -> d b ...', d=jax.local_device_count())
unshard = lambda x: einops.rearrange(x, 'd b ... -> (d b) ...')

correct = total = 0
for batch in ds.as_numpy_iterator():
  images = batch['image']
  n = len(images)
  padding = np.zeros([per_host_batch_size - n, *images.shape[1:]], images.dtype)
  padded_images = np.concatenate([images, padding])
  preds = unshard(get_preds(variables, shard(padded_images)))[:n]
  total += n
  correct += (batch['label'] == preds.argmax(axis=-1)).sum()
```

## `pad_shard_unpad()` 사용하기

위 패턴, 즉 pad→shard→predict→unshard→unpad 시퀀스를 유틸리티 래퍼 `pad_shard_unpad()`로 추출하여 위 평가 루프를 크게 간소화할 수 있습니다.



```python
correct = total = 0
for batch in ds.as_numpy_iterator():
  preds = flax.jax_utils.pad_shard_unpad(get_preds)(
      vs, batch['image'], min_device_batch=per_device_batch_size)
  total += len(batch['image'])
  correct += (batch['label'] == preds.argmax(axis=-1)).sum()
```

## `eval_step()`에서 메트릭 계산하기

메인 평가 루프에서 예측을 반환하고 메트릭을 계산하는 대신, 특히 [`clu.metrics`](https://github.com/cgarciae/jax_metrics) 또는 [`clu.metrics`](https://github.com/google/CommonLoopUtils/blob/main/clu/metrics.py)와 같은 라이브러리를 사용할 때 메트릭 계산을 평가 단계의 일부로 만들고자 하는 경우가 많습니다.

  

이 경우 메트릭을 `static_argnums`로 전달하고(즉, 샤딩/패딩하지 않음) 반환 값도 `static_return`으로 처리합니다.(즉, 언샤딩 또는 언패 없음) :






```python
def eval_step(metrics, variables, batch):
  print('retrigger compilation', {k: v.shape for k, v in batch.items()})
  preds = model.apply(variables, batch['image'])
  correct = (batch['mask'] & (batch['label'] == preds.argmax(axis=-1))).sum()
  total = batch['mask'].sum()
  return dict(
      correct=metrics['correct'] + jax.lax.psum(correct, axis_name='batch'),
      total=metrics['total'] + jax.lax.psum(total, axis_name='batch'),
  )

eval_step = jax.pmap(eval_step, axis_name='batch')
eval_step = flax.jax_utils.pad_shard_unpad(
    eval_step, static_argnums=(0, 1), static_return=True)
```

## "무한 패딩" 추가하기(Adding “infinite padding)

위의 솔루션은 대부분의 경우 작동하지만 몇 가지 제한 사항이 있습니다 :
1. 드물지만 여러 호스트에서 데이터셋을 분할해도 배치 수가 달라지는 경우가 있습니다. `n=4097`개의 예제의 데이터셋이 있고,이를 `h=8`에서 평가하고, 각각 `d=8`개의 로컬 장치를 가지고 있으며, 온디바이스 배치 크기가 `b=128`이라고 가정해 보겠습니다. 데이터셋을 균등하게 분할하면 첫 번째 호스트는 `4096/8+1==513`개의 예제를 얻게 되고 다른 모든 호스트는 `4096/8==512`개의 예제를 얻게 됩니다. 호스트별로 `d*b==512`의 배치를 형성하면 첫 번째 호스트에는 두 개의 배치가 생성되고, 다른 모든 호스트에는 하나의 배치가 생성되어 SPMD 원칙을 위반하고 마지막 `psum()` 지시문에서 다중 호스트 설정이 중단됩니다(첫 번째 호스트에서만 실행되고 다른 호스트에서는 실행되지 않음).



2. `ds.filter()`를 사용하여 예제를 동적으로 드롭할 때.



이렇게 더 복잡한 경우에는 데이터셋에 "무한 패딩"을 각 호스트에 독립적으로 추가하고 모든 호스트에 패딩되지 않은 예제가소진될 때까지 예제를 계속 처리할 수 있습니다.



```python
correct = total = 0
for batch in ds.as_numpy_iterator():
  n = count_p(batch['mask'])[0].item()  # adds sync barrier
  if not n: break

  preds = get_preds(vs, batch['image']).argmax(axis=-1)
  total += n
  correct += count_correct_p(batch['label'], preds, batch['mask'])[0]
```

이 방법이 포함된 다른 예제의 경우, 전체 실행 코드는 Colab에서 확인할 수 있습니다:


<a href="https://colab.research.google.com/github/google/flax/blob/main/docs/notebooks/full_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>
