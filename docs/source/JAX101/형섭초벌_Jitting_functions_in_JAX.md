# JAX의 Just-In-Time 컴파일


<a href="https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/02-jitting.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

*저자: Rosalia Schneider & Vladimir Mikulik*

*역자 : 김형섭*

*검수 : 이태호, 

이 섹션에서는 JAX의 동작 방식과 JAX의 성능 향상 방법을 살펴볼 것입니다. `jax.jit()` 변환을 통해 JIT(Just-In-Time) 컴파일을 수행하여 JAX 파이썬 함수를 XLA에서 효율적으로 실행할 수 있는 방법을 설명할 것입니다.

## JAX 변환의 동작 방식

이전 섹션에서 JAX가 파이썬 함수를 변환할 수 있다는 것을 언급했습니다. 이는 파이썬 함수를 jaxpr이라는 간단한 중간 언어로 먼저 변환한 다음, 그 변환이 jaxpr 표현에서 작동하는 것입니다.

`jax.make_jaxpr`을 사용하여 jaxpr의 표현을 확인할 수 있습니다.


```python
import jax
import jax.numpy as jnp

global_list = []

def log2(x):
  global_list.append(x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0))
```

    { lambda ; a:f32[]. let
        b:f32[] = log a
        c:f32[] = log 2.0
        d:f32[] = div b c
      in (d,) }


[Jaxprs Understanding](https://jax.readthedocs.io/en/latest/jaxpr.html) 섹션은 출력에 대한 자세한 정보를 제공합니다.

하지만 함수의 측면 효과가 jaxpr에서 나타나지 않는 것이 중요합니다. 예를 들어, `global_list.append(x)`와 같은 것은 나타나지 않습니다. 이는 JAX가 측면 효과가 없는 (기능적으로 순수한) 코드를 처리하도록 설계된 것이기 때문입니다. "순수 함수"와 "측면 효과"라는 개념이 새로운 경우에는 [🔪 JAX - The Sharp Bits 🔪: Pure Functions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)에서 더 자세히 설명되어 있습니다.

JAX는 부정적인 함수도 작성하고 실행할 수 있지만, jaxpr으로 변환된 후의 행동에 대한 보장은 하지 않습니다. 그러나 JAX 변형 함수의 측면 효과는 한 번만 실행될 것으로 기대할 수 있습니다. 이는 JAX가 jaxpr을 생성하는 과정인 'tracing'을 사용하기 때문입니다.

JAX는 각 인수를 `tracer` 객체로 감싸며, 이 트레이서는 함수 호출 동안 JAX에서 수행된 모든 작업을 기록합니다. JAX는 이 트레이스 기록을 사용하여 전체 함수를 재구성하고, 이 결과를 jaxpr으로 만듭니다. 트레이스는 Python의 부작용을 기록하지 않기 때문에, jaxpr에서는 부작용이 나타나지 않습니다. 그러나 부작용은 트레이스 자체에서 일어납니다.

노트: `print()` 함수는 순수하지 않으며, 텍스트 출력이 함수의 부작용입니다. 따라서 `print()` 호출은 추적할 때만 일어나고 jaxpr에는 나타나지 않습니다.


```python
def log2_with_print(x):
  print("printed x:", x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2_with_print)(3.))
```

    printed x: Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>
    { lambda ; a:f32[]. let
        b:f32[] = log a
        c:f32[] = log 2.0
        d:f32[] = div b c
      in (d,) }


`x`가 `Traced` 객체로 출력되는 것은 JAX의 내부 동작입니다. 이는 JAX가 함수의 수행 과정을 기록하기 위해 각 인수를 tracer 객체로 감싸는 과정입니다. 그러나, Python 코드의 실행은 구현 세부 사항이며 이에 의존하면 안 됩니다. 디버깅에서 값을 확인할 때 유용하지만, 중요하지 않은 부분입니다.





jaxpr의 핵심은 함수가 주어진 매개 변수에서 실행되는 것을 캡처하는 것입니다. 예를 들어, 조건이 있는 경우 jaxpr은 우리가 가진 분기에 대해서만 알게 될 것입니다.


```python
def log2_if_rank_2(x):
  if x.ndim == 2:
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2
  else:
    return x

print(jax.make_jaxpr(log2_if_rank_2)(jax.numpy.array([1, 2, 3])))
```

    { lambda ; a:i32[3]. let  in (a,) }


## 함수 계산을 위한 JIT 컴파일링

JAX는 CPU, GPU, TPU에서 동일한 코드로 작업을 실행할 수 있도록 합니다. 예를 들어, 딥러닝에서 자주 사용되는 *Scaled Exponential Linear Unit* ([SELU](https://proceedings.neurips.cc/paper/6698-self-normalizing-neural-networks.pdf))함수를 계산하는 것을 살펴보겠습니다.



```python
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)
%timeit selu(x).block_until_ready()
```

    100 loops, best of 5: 2.05 ms per loop


위 코드에서는 가속기에 한 번에 하나의 작업만을 보냅니다. 이로 인해 XLA 컴파일러의 함수 최적화 능력이 제한됩니다.

XLA 컴파일러에게 가능한 한 많은 코드를 제공하는 것이 목표입니다. JAX에서는 이를 위해 `jax.jit` 변환을 제공합니다. 이 변환을 사용하면 JAX 호환 함수를 JIT 컴파일 할 수 있습니다. 아래의 예제는 JIT을 사용해 이전 함수를 가속화하는 방법을 보여줍니다.


```python
selu_jit = jax.jit(selu)

# Warm up
selu_jit(x).block_until_ready()

%timeit selu_jit(x).block_until_ready()
```

    10000 loops, best of 5: 150 µs per loop


1) JAX에서 JIT을 사용하면 함수를 JIT 컴파일 할 수 있습니다. 예제에서 `selu` 함수를 JIT 컴파일하여 `selu_jit`라는 새로운 함수가 정의되었습니다.

2) `selu_jit` 함수는 `x` 값을 입력으로 받아 JAX의 트레이싱을 시작합니다. 트레이싱 과정에서 JAX는 XLA 최적화된 코드로 컴파일할 수 있는 jaxpr을 생성합니다. 이후 호출은 이전의 Python 구현보다 더 효율적인 코드를 사용할 수 있게 됩니다.

(만약 따로 데이터가 준비되지 않았다면, 그래도 `selu_jit`은 정상적으로 동작하지만 벤치마크에서 컴파일 시간이 포함되기 때문에 공정한 비교가 어려울 수 있습니다. 많은 반복이 있어도 여전히 빠를 것입니다.)

3) 컴파일된 버전의 성능이 측정되었습니다. (JAX의 [비동기 실행](https://jax.readthedocs.io/en/latest/async_dispatch.html) 모델 때문에 `block_until_ready()`를 사용하여 동기화가 필요함에 주의하세요.).


## 왜 모든 것을 JIT 컴파일할 수 없을까?

위의 예제를 통해 `jax.jit`을 모든 함수에 적용해야 하는지 궁금할 수 있습니다. JIT이 작동하지 않는 경우와 `jit`을 적용해야 하거나 하지 않아야 하는 경우를 알아보기 위해 JIT이 작동하지 않는 어떤 경우를 검토하세요.


```python
# Condition on value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

f_jit = jax.jit(f)
f_jit(10)  # Should raise an error. 
```


    ---------------------------------------------------------------------------

    UnfilteredStackTrace                      Traceback (most recent call last)

    <ipython-input-12-2c1a07641e48> in <module>()
          9 f_jit = jax.jit(f)
    ---> 10 f_jit(10)  # Should raise an error.
    

    /usr/local/lib/python3.7/dist-packages/jax/_src/traceback_util.py in reraise_with_filtered_traceback(*args, **kwargs)
        161     try:
    --> 162       return fun(*args, **kwargs)
        163     except Exception as e:


    /usr/local/lib/python3.7/dist-packages/jax/_src/api.py in cache_miss(*args, **kwargs)
        418         device=device, backend=backend, name=flat_fun.__name__,
    --> 419         donated_invars=donated_invars, inline=inline)
        420     out_pytree_def = out_tree()


    /usr/local/lib/python3.7/dist-packages/jax/core.py in bind(self, fun, *args, **params)
       1631   def bind(self, fun, *args, **params):
    -> 1632     return call_bind(self, fun, *args, **params)
       1633 


    /usr/local/lib/python3.7/dist-packages/jax/core.py in call_bind(primitive, fun, *args, **params)
       1622   tracers = map(top_trace.full_raise, args)
    -> 1623   outs = primitive.process(top_trace, fun, tracers, params)
       1624   return map(full_lower, apply_todos(env_trace_todo(), outs))


    /usr/local/lib/python3.7/dist-packages/jax/core.py in process(self, trace, fun, tracers, params)
       1634   def process(self, trace, fun, tracers, params):
    -> 1635     return trace.process_call(self, fun, tracers, params)
       1636 


    /usr/local/lib/python3.7/dist-packages/jax/core.py in process_call(self, primitive, f, tracers, params)
        626   def process_call(self, primitive, f, tracers, params):
    --> 627     return primitive.impl(f, *tracers, **params)
        628   process_map = process_call


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/xla.py in _xla_call_impl(***failed resolving arguments***)
        687   compiled_fun = _xla_callable(fun, device, backend, name, donated_invars,
    --> 688                                *unsafe_map(arg_spec, args))
        689   try:


    /usr/local/lib/python3.7/dist-packages/jax/linear_util.py in memoized_fun(fun, *args)
        262     else:
    --> 263       ans = call(fun, *args)
        264       cache[key] = (ans, fun.stores)


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/xla.py in _xla_callable_uncached(fun, device, backend, name, donated_invars, *arg_specs)
        759   return lower_xla_callable(fun, device, backend, name, donated_invars,
    --> 760                             *arg_specs).compile().unsafe_call
        761 


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/xla.py in lower_xla_callable(fun, device, backend, name, donated_invars, *arg_specs)
        771   jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(
    --> 772       fun, abstract_args, pe.debug_info_final(fun, "jit"))
        773   if any(isinstance(c, core.Tracer) for c in consts):


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/partial_eval.py in trace_to_jaxpr_final(fun, in_avals, debug_info)
       1541     with core.new_sublevel():
    -> 1542       jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, main, in_avals)
       1543     del fun, main


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/partial_eval.py in trace_to_subjaxpr_dynamic(fun, main, in_avals)
       1519     in_tracers = map(trace.new_arg, in_avals)
    -> 1520     ans = fun.call_wrapped(*in_tracers)
       1521     out_tracers = map(trace.full_raise, ans)


    /usr/local/lib/python3.7/dist-packages/jax/linear_util.py in call_wrapped(self, *args, **kwargs)
        165     try:
    --> 166       ans = self.f(*args, **dict(self.params, **kwargs))
        167     except:


    <ipython-input-12-2c1a07641e48> in f(x)
          3 def f(x):
    ----> 4   if x > 0:
          5     return x


    /usr/local/lib/python3.7/dist-packages/jax/core.py in __bool__(self)
        548   def __nonzero__(self): return self.aval._nonzero(self)
    --> 549   def __bool__(self): return self.aval._bool(self)
        550   def __int__(self): return self.aval._int(self)


    /usr/local/lib/python3.7/dist-packages/jax/core.py in error(self, arg)
        999   def error(self, arg):
    -> 1000     raise ConcretizationTypeError(arg, fname_context)
       1001   return error


    UnfilteredStackTrace: jax._src.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: Traced<ShapedArray(bool[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
    The problem arose with the bool function. 
    While tracing the function f at <ipython-input-12-2c1a07641e48>:3 for jit, this concrete value was not available in Python because it depends on the value of the argument 'x'.
    
    See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
    
    The stack trace below excludes JAX-internal frames.
    The preceding is the original exception that occurred, unmodified.
    
    --------------------

    
    The above exception was the direct cause of the following exception:


    ConcretizationTypeError                   Traceback (most recent call last)

    <ipython-input-12-2c1a07641e48> in <module>()
          8 
          9 f_jit = jax.jit(f)
    ---> 10 f_jit(10)  # Should raise an error.
    

    <ipython-input-12-2c1a07641e48> in f(x)
          2 
          3 def f(x):
    ----> 4   if x > 0:
          5     return x
          6   else:


    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: Traced<ShapedArray(bool[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
    The problem arose with the bool function. 
    While tracing the function f at <ipython-input-12-2c1a07641e48>:3 for jit, this concrete value was not available in Python because it depends on the value of the argument 'x'.
    
    See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError



```python
# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

g_jit = jax.jit(g)
g_jit(10, 20)  # Should raise an error. 
```


    ---------------------------------------------------------------------------

    UnfilteredStackTrace                      Traceback (most recent call last)

    <ipython-input-13-2aa78f448d5d> in <module>()
          9 g_jit = jax.jit(g)
    ---> 10 g_jit(10, 20)  # Should raise an error.
    

    /usr/local/lib/python3.7/dist-packages/jax/_src/traceback_util.py in reraise_with_filtered_traceback(*args, **kwargs)
        161     try:
    --> 162       return fun(*args, **kwargs)
        163     except Exception as e:


    /usr/local/lib/python3.7/dist-packages/jax/_src/api.py in cache_miss(*args, **kwargs)
        418         device=device, backend=backend, name=flat_fun.__name__,
    --> 419         donated_invars=donated_invars, inline=inline)
        420     out_pytree_def = out_tree()


    /usr/local/lib/python3.7/dist-packages/jax/core.py in bind(self, fun, *args, **params)
       1631   def bind(self, fun, *args, **params):
    -> 1632     return call_bind(self, fun, *args, **params)
       1633 


    /usr/local/lib/python3.7/dist-packages/jax/core.py in call_bind(primitive, fun, *args, **params)
       1622   tracers = map(top_trace.full_raise, args)
    -> 1623   outs = primitive.process(top_trace, fun, tracers, params)
       1624   return map(full_lower, apply_todos(env_trace_todo(), outs))


    /usr/local/lib/python3.7/dist-packages/jax/core.py in process(self, trace, fun, tracers, params)
       1634   def process(self, trace, fun, tracers, params):
    -> 1635     return trace.process_call(self, fun, tracers, params)
       1636 


    /usr/local/lib/python3.7/dist-packages/jax/core.py in process_call(self, primitive, f, tracers, params)
        626   def process_call(self, primitive, f, tracers, params):
    --> 627     return primitive.impl(f, *tracers, **params)
        628   process_map = process_call


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/xla.py in _xla_call_impl(***failed resolving arguments***)
        687   compiled_fun = _xla_callable(fun, device, backend, name, donated_invars,
    --> 688                                *unsafe_map(arg_spec, args))
        689   try:


    /usr/local/lib/python3.7/dist-packages/jax/linear_util.py in memoized_fun(fun, *args)
        262     else:
    --> 263       ans = call(fun, *args)
        264       cache[key] = (ans, fun.stores)


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/xla.py in _xla_callable_uncached(fun, device, backend, name, donated_invars, *arg_specs)
        759   return lower_xla_callable(fun, device, backend, name, donated_invars,
    --> 760                             *arg_specs).compile().unsafe_call
        761 


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/xla.py in lower_xla_callable(fun, device, backend, name, donated_invars, *arg_specs)
        771   jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(
    --> 772       fun, abstract_args, pe.debug_info_final(fun, "jit"))
        773   if any(isinstance(c, core.Tracer) for c in consts):


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/partial_eval.py in trace_to_jaxpr_final(fun, in_avals, debug_info)
       1541     with core.new_sublevel():
    -> 1542       jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, main, in_avals)
       1543     del fun, main


    /usr/local/lib/python3.7/dist-packages/jax/interpreters/partial_eval.py in trace_to_subjaxpr_dynamic(fun, main, in_avals)
       1519     in_tracers = map(trace.new_arg, in_avals)
    -> 1520     ans = fun.call_wrapped(*in_tracers)
       1521     out_tracers = map(trace.full_raise, ans)


    /usr/local/lib/python3.7/dist-packages/jax/linear_util.py in call_wrapped(self, *args, **kwargs)
        165     try:
    --> 166       ans = self.f(*args, **dict(self.params, **kwargs))
        167     except:


    <ipython-input-13-2aa78f448d5d> in g(x, n)
          4   i = 0
    ----> 5   while i < n:
          6     i += 1


    /usr/local/lib/python3.7/dist-packages/jax/core.py in __bool__(self)
        548   def __nonzero__(self): return self.aval._nonzero(self)
    --> 549   def __bool__(self): return self.aval._bool(self)
        550   def __int__(self): return self.aval._int(self)


    /usr/local/lib/python3.7/dist-packages/jax/core.py in error(self, arg)
        999   def error(self, arg):
    -> 1000     raise ConcretizationTypeError(arg, fname_context)
       1001   return error


    UnfilteredStackTrace: jax._src.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: Traced<ShapedArray(bool[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
    The problem arose with the bool function. 
    While tracing the function g at <ipython-input-13-2aa78f448d5d>:3 for jit, this concrete value was not available in Python because it depends on the value of the argument 'n'.
    
    See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
    
    The stack trace below excludes JAX-internal frames.
    The preceding is the original exception that occurred, unmodified.
    
    --------------------

    
    The above exception was the direct cause of the following exception:


    ConcretizationTypeError                   Traceback (most recent call last)

    <ipython-input-13-2aa78f448d5d> in <module>()
          8 
          9 g_jit = jax.jit(g)
    ---> 10 g_jit(10, 20)  # Should raise an error.
    

    <ipython-input-13-2aa78f448d5d> in g(x, n)
          3 def g(x, n):
          4   i = 0
    ----> 5   while i < n:
          6     i += 1
          7   return x + i


    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: Traced<ShapedArray(bool[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
    The problem arose with the bool function. 
    While tracing the function g at <ipython-input-13-2aa78f448d5d>:3 for jit, this concrete value was not available in Python because it depends on the value of the argument 'n'.
    
    See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError


우리가 *특정 값*의 입력에 따라 JIT 컴파일된 함수를 결정하려고 시도했기 때문에 문제가 발생했습니다. 위에서 언급한 것처럼 jaxpr은 실제 추적에 사용된 값에 의존하기 때문에 이러한 문제가 발생한 것입니다.

JAX는 다른 목적을 위해 다른 추상화 수준에서 트레이싱함으로써 이 문제를 해결합니다. 트레이싱하는 값에 대한 정보가 더 구체적이면 표준 파이썬 제어 흐름을 사용하여 자신을 표현할 수 있습니다. 하지만 너무 구체적인 것은 같은 트레이싱 된 함수를 다른 값에 재사용할 수 없다는 것을 의미합니다.

`jax.jit`의 기본 레벨은 `ShapedArray`입니다. 즉, 각 트레이서에는 구체적인 모양이 있지만, 구체적인 값은 없습니다. 이는 정해진 모양의 모든 가능한 입력에서 함수를 작동시키는 것을 허용합니다. 즉, 기계 학습의 표준 사용 사례입니다. 그러나 트레이서에 구체적인 값이 없기 때문에, 그 값에 조건을 둔다면 위에서 언급한 오류가 발생합니다.

`jax.grad`에서는 제약이 더 완화되어 더 많은 것을 할 수 있습니다. 하지만 여러 변환을 결합하면 가장 엄격한 것의 제약을 만족해야합니다. 따라서 `jit(grad(f))`를 하면 `f`는 값에 대한 조건을 만족하지 않아야합니다. Python 제어 흐름과 JAX 간의 상호 작용에 대한 자세한 내용은 [🔪 JAX - The Sharp Bits 🔪: Control Flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow)을 참조하십시오.


이 문제를 해결하는 한 가지 방법은 코드를 수정하여 값에 대한 조건을 피하는 것입니다. 또 다른 것은 [제어 흐름 연산자](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators)인 `jax.lax.cond`와 같은 특별한 제어 흐름 연산자를 사용하는 것입니다. 그러나 때로는 그것이 불가능할 수도 있습니다. 그러한 경우에는 함수의 일부분만 JIT 컴파일할 수 있습니다. 예를 들어, 함수의 계산적으로 가장 비용이 많이 드는 부분이 루프 내에 있다면, 안쪽 부분만 JIT 컴파일 할 수 있습니다(캐싱에 대한 다음 섹션을 검토하여 발자취를 피하는 것을 잊지 마십시오).


```python
# While loop conditioned on x and n with a jitted body.

@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)
```




    DeviceArray(30, dtype=int32, weak_type=True)



입력값의 값에 기반한 조건이 있는 함수를 JIT 해야 한다면, JAX에게 `static_argnums` 또는 `static_argnames`를 지정하여 특정 입력에 대해 적은 추상적인 트레이서를 사용하도록 허용할 수 있습니다. 이는 얻는 결과의 jaxpr이 덜 유연해지기 때문에, 지정된 정적 입력의 값이 변경될 때마다 함수를 다시 컴파일해야 한다는 비용이 있습니다. 이는 함수가 제한적인 다른 값만 얻는다는 것이 보장될 때만 좋은 전략입니다.


```python
f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))
```

    10



```python
g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))
```

    30


함수에 입력 값의 값에 대한 조건이 있을 때 `jit`을 적용하려면 `static_argnums` 또는 `static_argnames`을 지정하여 JAX에게 특정 입력에 대해 적은 추상적인 트레이서를 사용하도록 알려줄 수 있습니다. 


```python
from functools import partial

@partial(jax.jit, static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))
```

    30


## JIT을 언제 사용해야 하는가?

위의 많은 예시에서, JIT 적용은 유용하지 않을 수 있습니다.


```python
print("g jitted:")
%timeit g_jit_correct(10, 20).block_until_ready()

print("g:")
%timeit g(10, 20)
```

    g jitted:
    The slowest run took 13.54 times longer than the fastest. This could mean that an intermediate result is being cached.
    1000 loops, best of 5: 229 µs per loop
    g:
    The slowest run took 11.72 times longer than the fastest. This could mean that an intermediate result is being cached.
    1000000 loops, best of 5: 1.2 µs per loop


JIT이 유용하지 않은 경우가 있습니다. 이는 `jax.jit` 함수 자체가 오버헤드를 가지고 있기 때문입니다. 따라서, 컴파일된 함수가 복잡하고 많은 횟수 실행되어야 JIT의 이점을 얻게 됩니다. 기계 학습에서는 특히 이러한 경우가 많습니다. 예를 들어, 복잡한 모델을 컴파일하고 수백만 번의 반복 연산을 수행하는 경우입니다.

일반적으로, JIT을 적용할 때 계산의 가능한 가장 큰 부분(가능하면 전체 업데이트 단계)을 JIT하는 것이 좋습니다. 이는 컴파일러에게 최적화를 위한 최대한의 자유를 제공하기 때문입니다.

## 캐싱

`jax.jit`의 캐싱 기능에 대해 이해하는 것이 중요합니다.

예를 들어, `f = jax.jit(g)`가 정의되었다면, 첫 번째 `f` 호출 시 컴파일되고 결과적으로 생성된 XLA 코드가 캐시됩니다. 이후의 `f` 호출은 캐시된 코드를 재사용할 것입니다. 이렇게 `jax.jit`이 컴파일 작업의 앞단 비용을 보상할 수 있습니다.

만약 `static_argnums`가 지정되어 있다면, 캐시된 코드는 정적으로 지정된 인수 값이 동일한 경우에만 사용될 것입니다. 만약 인수 값이 변경된다면, 재컴파일이 일어날 것입니다. 많은 인수 값이 있을 경우, 프로그램은 연산을 하나씩 수행하는 것보다 더 많은 시간을 컴파일하는 것에 쓸 수 있습니다.

루프 안에서 `jax.jit` 호출은 피하는 것이 좋습니다. JAX는 대개, 다음 호출에서 `jax.jit`으로 컴파일된 캐시된 함수를 재사용할 수 있습니다. 하지만, 캐시가 함수의 해시에 의존하기 때문에, 동일한 함수가 다시 정의될 경우 문제가 될 수 있습니다. 결과적으로, 각 루프마다 불필요한 컴파일이 발생할 수 있습니다. 


```python
from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called in a loop with partials:")
%timeit g_inner_jitted_partial(10, 20).block_until_ready()

print("jit called in a loop with lambdas:")
%timeit g_inner_jitted_lambda(10, 20).block_until_ready()

print("jit called in a loop with caching:")
%timeit g_inner_jitted_normal(10, 20).block_until_ready()
```

    jit called in a loop with partials:
    1 loop, best of 5: 192 ms per loop
    jit called in a loop with lambdas:
    10 loops, best of 5: 199 ms per loop
    jit called in a loop with caching:
    10 loops, best of 5: 21.6 ms per loop

