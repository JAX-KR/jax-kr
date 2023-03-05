JAX - 세부적인 특징들
++++++++++++++++++++

.. container:: cell markdown

   .. rubric:: # 🔪JAX - 세부적인 특징들🔪
      :name: -jax---세부적인-특징들

.. container:: cell markdown

   이탈리아 변두리에서 걷다보면, 사람들이 ``JAX``\ 에 대해서 `“una anima
   di pura programmazione funzionale(순수한 함수형 프로그래밍의
   영혼)” <https://www.sscardapane.it/iaml-backup/jax-intro/>`__\ 라고
   말하는 것에 거리낌이 없다는 것을 알 수 있을 것입니다.

   ``JAX``\ 는 수치형 프로그램의 변환을 표현 및 구성하는 언어입니다.
   ``JAX``\ 는 수치형 프로그램을 CPU 혹은 가속기(GPU/TPU)에서 동작하도록
   컴파일할 수 있습니다. JAX는 아래에서 설명하는 특정한 제약조건으로
   작성하는 것을 만족한다면, 많은 수치적 그리고 과학적 프로그램에서 잘
   동작합니다.

   (2차 검수) ``JAX``\ 는 수치해석 프로그램의 변환을 표현 및 구성하는
   언어입니다. ``JAX``\ 는 수치형 프로그램을 CPU 혹은
   가속기(GPU/TPU)에서 동작하도록 컴파일할 수 있습니다. JAX는 아래에서
   설명하는 특정한 제약조건을 만족한다면, 많은 수치적 그리고 과학적
   프로그램에서 잘 동작합니다.

.. container:: cell code

   .. code:: python

      import numpy as np
      from jax import grad, jit
      from jax import lax
      from jax import random
      import jax
      import jax.numpy as jnp
      import matplotlib as mpl
      from matplotlib import pyplot as plt
      from matplotlib import rcParams
      rcParams['image.interpolation'] = 'nearest'
      rcParams['image.cmap'] = 'viridis'
      rcParams['axes.grid'] = False

.. container:: cell markdown

   .. rubric:: **🔪순수 함수**
      :name: 순수-함수

   --------------

   ``JAX``\ 의 변환 및 컴파일은 함수적으로 순수한 경우, Python 함수에서
   잘 동작하도록 설계되어 있습니다.

   (함수적으로 순수한 경우란, 모든 입력 데이터가 함수의 매개변수를 통해
   전달되고, 모든 출력 결과가 함수의 결과를 통해 나오는 경우를
   의미합니다.)

   따라서, 순수 함수는 같은 입력이 주어진다면 항상 같은 결과를 반환하는
   함수입니다.

   다음은 ``JAX``\ 가 Python 인터프리터와 다르게 동작하는 함수적으로
   순수하지 않은 함수의 몇 가지 예입니다. 이와 같은 예는 JAX
   system에서의 동작이 보장되지 않는다는 점에 주목합시다. JAX를 사용하는
   적절한 방법은 함수적으로 순수한 Python 함수에 대해서만 사용하는
   것입니다..

.. container:: cell code

   .. code:: python

      def impure_print_side_effect(x):
        print("Executing function")  # This is a side-effect 
        return x

      # The side-effects appear during the first run  
      print ("First call: ", jit(impure_print_side_effect)(4.))

      # Subsequent runs with parameters of same type and shape may not show the side-effect
      # This is because JAX now invokes a cached compilation of the function
      print ("Second call: ", jit(impure_print_side_effect)(5.))

      # JAX re-runs the Python function when the type or shape of the argument changes
      print ("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))

   .. container:: output stream stderr

      ::

         WARNING:jax._src.lib.xla_bridge:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

   .. container:: output stream stdout

      ::

         Executing function
         First call:  4.0
         Second call:  5.0
         Executing function
         Third call, different type:  [5.]

.. container:: cell code

   .. code:: python

      g = 0.
      def impure_uses_globals(x):
        return x + g

      # JAX captures the value of the global during the first run
      print ("First call: ", jit(impure_uses_globals)(4.))
      g = 10.  # Update the global

      # Subsequent runs may silently use the cached value of the globals
      print ("Second call: ", jit(impure_uses_globals)(5.))

      # JAX re-runs the Python function when the type or shape of the argument changes
      # This will end up reading the latest value of the global
      print ("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))

   .. container:: output stream stdout

      ::

         First call:  4.0
         Second call:  5.0
         Third call, different type:  [14.]

.. container:: cell code

   .. code:: python

      g = 0.
      def impure_saves_global(x):
        global g
        g = x
        return x

      # JAX runs once the transformed function with special Traced values for arguments
      print ("First call: ", jit(impure_saves_global)(4.))
      print ("Saved global: ", g)  # Saved global has an internal JAX value

   .. container:: output stream stdout

      ::

         First call:  4.0
         Saved global:  Traced<ShapedArray(float32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>

.. container:: cell markdown

   Python 함수가 만약 실제로 상태 저장 객체를 내부적으로 사용하더라도
   이를 외부에서 읽거나 쓰지만 않는다면 함수적으로 순수하다고 할 수
   있습니다.

   --> **(2차 검수)** Python 함수가 만약 실제로 스테이트풀 객체를
   내부적으로 사용하더라도 이를 외부에서 읽거나 쓰지만 않는다면
   함수적으로 순수하다고 할 수 있습니다.

.. container:: cell code

   .. code:: python

      def pure_uses_internal_state(x):
        state = dict(even=0, odd=0)
        for i in range(10):
          state['even' if i % 2 == 0 else 'odd'] += x
        return state['even'] + state['odd']

      print(jit(pure_uses_internal_state)(5.))

   .. container:: output stream stdout

      ::

         50.0

.. container:: cell markdown

   ``jit``\ 을 사용하려는 JAX 함수나 어떤 제어 흐름 프리미티브에서
   iterators를 사용하는 것은 추천하지 않습니다. 그 이유는 iterator가
   다음 요소를 가져오기 위한 상태를 찾으려 도입된 파이썬 객체이기
   때문입니다. 그러므로, iterator는 JAX의 함수적 프로그래밍 모델과
   호환되지 않습니다. 아래 코드에서, JAX에서 iterators를 사용하려는
   부적절한 시도들에 대한 예제가 있습니다. 대부분의 경우 오류를
   반환하지만, 어떤 경우는 예상치 못한 결과를 보여줍니다.

   -> (2차 검수) ``jit``\ 을 사용하려는 JAX 함수나 어떤 제어 흐름
   구성요소에서 iterators를 사용하는 것은 추천하지 않습니다. 그 이유는
   iterator(반복자)가 다음 요소를 가져오기 위한 상태(state)를 찾으려
   도입된 파이썬 객체이기 때문입니다. 그러므로, iterator는 JAX의 함수적
   프로그래밍 모델과 호환되지 않습니다. 아래 코드에서, JAX에서
   iterators를 사용하려는 부적절한 시도들에 대한 예제가 있습니다.
   대부분의 경우 오류를 반환하지만, 어떤 경우는 예상치 못한 결과를
   보여줍니다.

.. container:: cell code

   .. code:: python

      import jax.numpy as jnp
      import jax.lax as lax
      from jax import make_jaxpr

      # lax.fori_loop
      array = jnp.arange(10)
      print(lax.fori_loop(0, 10, lambda i,x: x+array[i], 0)) # expected result 45
      iterator = iter(range(10))
      print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0)) # unexpected result 0

      # lax.scan
      def func11(arr, extra):
          ones = jnp.ones(arr.shape)  
          def body(carry, aelems):
              ae1, ae2 = aelems
              return (carry + ae1 * ae2 + extra, carry)
          return lax.scan(body, 0., (arr, ones))    
      make_jaxpr(func11)(jnp.arange(16), 5.)
      # make_jaxpr(func11)(iter(range(16)), 5.) # throws error

      # lax.cond
      array_operand = jnp.array([0.])
      lax.cond(True, lambda x: x+1, lambda x: x-1, array_operand)
      iter_operand = iter(range(10))
      # lax.cond(True, lambda x: next(x)+1, lambda x: next(x)-1, iter_operand) # throws error

   .. container:: output stream stdout

      ::

         45
         0

.. container:: cell markdown

   .. rubric:: ## 🔪In-Place 업데이트
      :name: -in-place-업데이트

   Numpy를 사용할 때, 여러분들은 종종 이렇게 사용할 것입니다.

.. container:: cell code

   .. code:: python

      numpy_array = np.zeros((3,3), dtype=np.float32)
      print("original array:")
      print(numpy_array)

      # In place, mutating update
      numpy_array[1, :] = 1.0
      print("updated array:")
      print(numpy_array)

   .. container:: output stream stdout

      ::

         original array:
         [[0. 0. 0.]
          [0. 0. 0.]
          [0. 0. 0.]]
         updated array:
         [[0. 0. 0.]
          [1. 1. 1.]
          [0. 0. 0.]]

.. container:: cell markdown

   하지만, 만약 JAX device array에 in-place로 업데이트를 시도하려 하면,
   오류가 발생하는 것을 볼 수 있을 겁니다. (☉_☉)

   -> (2차 검수) 하지만, 만약 JAX device array에 in-place로 업데이트를
   시도하면, 오류가 발생하는 것을 볼 수 있습니다. (☉_☉)

.. container:: cell code

   .. code:: python

      jax_array = jnp.zeros((3,3), dtype=jnp.float32)

      # In place update of JAX's array will yield an error!
      try:
        jax_array[1, :] = 1.0
      except Exception as e:
        print("Exception {}".format(e))

   .. container:: output stream stdout

      ::

         Exception '<class 'jaxlib.xla_extension.DeviceArray'>' object does not support item assignment. JAX arrays are immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` or another .at[] method: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html

.. container:: cell markdown

   변수의 in-place 변형을 허용하는 것은 프로그램의 분석과 변환이
   어려워집니다. JAX에서는 프로그램이 순수 함수여야 한다는 것을
   기억합시다.

   -> (2차 검수..?) 변수의 제자리 알고리즘(in-place)을 허용하는 것은
   프로그램의 분석과 변환이 어려워집니다. JAX에서는 프로그램이 순수
   함수여야 한다는 것을 기억합시다.

   in-place 기법 대신에, JAX에서는 JAX array에 ``.at`` 속성을 이용하여
   함수적 배열 업데이트를 수행합니다.

   ⚠️ ``jit``\ 된 코드 와 lax.while_loop 또는 lax.fori_loop 내부에서
   슬라이스의 크기는 인수 *값*\ 의 함수가 아니라 인수 *형태*\ 의
   함수여야 가능합니다. - 슬라이스 시작 인덱스에는 그러한 제한이
   없습니다. 아래의 **제어 흐름** 부분에서 이러한 제약에 대한 정보를
   확인할 수 있습니다.

   배열 업데이트\ **:** ``x.at[idx].set(y)``

   예를 들어, 업데이트는 아래와 같이 작성할 수 있습니다.

.. container:: cell code

   .. code:: python

      updated_array = jax_array.at[1, :].set(1.0)
      print("updated array:\n", updated_array)

   .. container:: output stream stdout

      ::

         updated array:
          [[0. 0. 0.]
          [1. 1. 1.]
          [0. 0. 0.]]

.. container:: cell markdown

   JAX의 업데이트 함수는 NumPy와는 다르게 out-of-place로 동작합니다. 즉,
   업데이트된 배열은 새 배열로 반환되며 원래 배열은 업데이트로 수정되지
   않습니다.

.. container:: cell code

   .. code:: python

      print("original array unchanged:\n", jax_array)

   .. container:: output stream stdout

      ::

         original array unchanged:
          [[0. 0. 0.]
          [0. 0. 0.]
          [0. 0. 0.]]

.. container:: cell markdown

   하지만, ``jit``\ 으로 컴파일 된 코드 내에서 ``x.at[idx].set(y)``\ 의
   입력 값 x가 재사용되지 않으면 컴파일러는 in-place로 배열이 업데이트
   되도록 최적화할 것입니다.

.. container:: cell markdown

   .. rubric:: 다른 연산과 함께 배열 업데이트
      :name: 다른-연산과-함께-배열-업데이트

   인덱스가 지정된 배열의 업데이트는 단순히 값을 덮어쓰는 것에만
   제한되지는 않습니다. 예를 들어, 아래의 예시와 같이 인덱스에 덧셈을
   하는 연산을 수행할 수 있습니다.

.. container:: cell code

   .. code:: python

      print("original array:")
      jax_array = jnp.ones((5, 6))
      print(jax_array)

      new_jax_array = jax_array.at[::2, 3:].add(7.)
      print("new array post-addition:")
      print(new_jax_array)

   .. container:: output stream stdout

      ::

         original array:
         [[1. 1. 1. 1. 1. 1.]
          [1. 1. 1. 1. 1. 1.]
          [1. 1. 1. 1. 1. 1.]
          [1. 1. 1. 1. 1. 1.]
          [1. 1. 1. 1. 1. 1.]]
         new array post-addition:
         [[1. 1. 1. 8. 8. 8.]
          [1. 1. 1. 1. 1. 1.]
          [1. 1. 1. 8. 8. 8.]
          [1. 1. 1. 1. 1. 1.]
          [1. 1. 1. 8. 8. 8.]]

.. container:: cell markdown

   보다 더 자세한 인덱스된 배열의 업데이트에 관련하여서는, 해당 문서를
   참고해주세요. documentation for the .at property

.. container:: cell markdown

   .. rubric:: ## 🔪 범위를 벗어난 인덱싱
      :name: --범위를-벗어난-인덱싱

   NumPy에서는 여러분이 인덱스 배열의 인덱스 범위를 벗어나는 동작을
   수행하면 아래와 같은 에러를 볼 수 있습니다.

.. container:: cell code

   .. code:: python

      np.arange(10)[11]

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         IndexError                                Traceback (most recent call last)
         <ipython-input-13-1dcbbeb664fb> in <module>
         ----> 1 np.arange(10)[11]

         IndexError: index 11 is out of bounds for axis 0 with size 10

.. container:: cell markdown

   하지만, 가속기에서 동작하는 코드로부터 에러를 발생시키는 것은
   어렵거나 심지어는 불가능할 수 있습니다. 그러므로, JAX는 배열의 범위를
   벗어나는 인덱싱에 대해서 오류가 아닌 동작을 선택해야 합니다.
   (유효하지 않은 부동 소수점의 산술적 결과가 NaN이 되는 것과
   유사합니다.). 인덱싱 작업이 배열 인덱스 업데이트(예: ``index_add``
   또는 ``scatter``-유사 프리미티브)인 경우, 범위를 벗어난 인덱스의
   업데이트는 건너뜁니다. 작업이 배열 인덱스 검색(예: NumPy 인덱싱 또는
   ``gather``-유사 프리미티브)인 경우, 무언가를 반환해야 하므로 인덱스가
   배열의 범위에 고정됩니다. 예를 들어, 아래의 인덱싱 동작에서는 배열의
   마지막 값이 반환될 것입니다.

   (2차 검수) 하지만, 가속기에서 동작하는 코드로부터 에러를 발생시키는
   것은 어렵거나 심지어는 불가능할 수 있습니다. 그러므로, JAX는 배열의
   범위를 벗어나는 인덱싱에 대해서 오류가 아닌 동작을 선택해야 합니다.
   (유효하지 않은 부동 소수점의 산술적 결과가 NaN이 되는 것과
   유사합니다.). 인덱싱 작업이 배열 인덱스 업데이트(예: ``index_add``
   또는 ``scatter``-유사한 기본 요소)인 경우, 범위를 벗어난 인덱스의
   업데이트는 건너뜁니다. 작업이 배열 인덱스 검색(예: NumPy 인덱싱 또는
   ``gather``-유사 프리미티브)인 경우, 무언가를 반환해야 하므로 인덱스가
   배열의 범위에 고정됩니다. 예를 들어, 아래의 인덱싱 동작에서는 배열의
   마지막 값이 반환될 것입니다.

.. container:: cell code

   .. code:: python

      jnp.arange(10)[11]

   .. container:: output execute_result

      ::

         DeviceArray(9, dtype=int32)

.. container:: cell markdown

   인덱스 검색에 대한 이러한 동작으로 인해 ``jnp.nanargmin`` 및
   ``jnp.nanargmax``\ 와 같은 함수는 NaN으로 구성된 슬라이스에 대해 -1을
   반환하지만 Numpy는 오류를 발생시킵니다.

   위에서 설명한 두 가지 동작이 서로 상쇄되지 않기 때문에, 역방향 자동
   미분(인덱스 업데이트를 인덱스 검색으로변환하고 그 반대로 전환)은
   범위를 벗어난 인덱싱의 의미를 보존하지 않습니다. 따라서 JAX의 범위를
   벗어난 인덱싱을 정의되지 않은 동작의 경우로 생각하는 것이 좋습니다.

   -> (2차 검수) 위에서 설명한 두 가지 동작이 서로 역의 관계가 아니기
   때문에, 역방향 자동 미분(인덱스 업데이트를 인덱스 검색으로 변환하고
   그 반대로 전환)은 범위를 벗어난 인덱싱의 의미를 보존하지 않습니다.
   따라서 JAX의 범위를 벗어난 인덱싱을 정의되지 않은 동작으로 생각하는
   것이 좋습니다.

.. container:: cell markdown

   .. rubric:: ## 🔪 비배열 입력: NumPy vs. Jax
      :name: --비배열-입력-numpy-vs-jax

   NumPy는 일반적으로 Python의 리스트 또는 튜플을 API 함수에 대한
   입력으로 사용합니다.

.. container:: cell code

   .. code:: python

      np.sum([1, 2, 3])

   .. container:: output execute_result

      ::

         6

.. container:: cell markdown

   JAX는 이와 다르게 일반적으로 유용한 오류를 반환합니다.

.. container:: cell code

   .. code:: python

      jnp.sum([1, 2, 3])

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         UnfilteredStackTrace                      Traceback (most recent call last)
         <ipython-input-16-730cb94339bb> in <module>
         ----> 1 jnp.sum([1, 2, 3])

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/reductions.py in sum(a, axis, dtype, out, keepdims, initial, where, promote_integers)
             215         where: Optional[ArrayLike] = None, promote_integers: bool = True) -> Array:
         --> 216   return _reduce_sum(a, axis=_ensure_optional_axes(axis), dtype=dtype, out=out,
             217                      keepdims=keepdims, initial=initial, where=where,

         /usr/local/lib/python3.8/dist-packages/jax/_src/traceback_util.py in reraise_with_filtered_traceback(*args, **kwargs)
             161     try:
         --> 162       return fun(*args, **kwargs)
             163     except Exception as e:

         /usr/local/lib/python3.8/dist-packages/jax/_src/api.py in cache_miss(*args, **kwargs)
             621         jax.config.jax_debug_nans or jax.config.jax_debug_infs):
         --> 622       execute = dispatch._xla_call_impl_lazy(fun_, *tracers, **params)
             623       out_flat = call_bind_continuation(execute(*args_flat))

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in _xla_call_impl_lazy(***failed resolving arguments***)
             235     arg_specs = [(None, getattr(x, '_device', None)) for x in args]
         --> 236   return xla_callable(fun, device, backend, name, donated_invars, keep_unused,
             237                       *arg_specs)

         /usr/local/lib/python3.8/dist-packages/jax/linear_util.py in memoized_fun(fun, *args)
             302     else:
         --> 303       ans = call(fun, *args)
             304       cache[key] = (ans, fun.stores)

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in _xla_callable_uncached(fun, device, backend, name, donated_invars, keep_unused, *arg_specs)
             358   else:
         --> 359     return lower_xla_callable(fun, device, backend, name, donated_invars, False,
             360                               keep_unused, *arg_specs).compile().unsafe_call

         /usr/local/lib/python3.8/dist-packages/jax/_src/profiler.py in wrapper(*args, **kwargs)
             313     with TraceAnnotation(name, **decorator_kwargs):
         --> 314       return func(*args, **kwargs)
             315     return wrapper

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in lower_xla_callable(fun, device, backend, name, donated_invars, always_lower, keep_unused, *arg_specs)
             444                         "for jit in {elapsed_time} sec"):
         --> 445     jaxpr, out_type, consts = pe.trace_to_jaxpr_final2(
             446         fun, pe.debug_info_final(fun, "jit"))

         /usr/local/lib/python3.8/dist-packages/jax/_src/profiler.py in wrapper(*args, **kwargs)
             313     with TraceAnnotation(name, **decorator_kwargs):
         --> 314       return func(*args, **kwargs)
             315     return wrapper

         /usr/local/lib/python3.8/dist-packages/jax/interpreters/partial_eval.py in trace_to_jaxpr_final2(fun, debug_info)
            2076     with core.new_sublevel():
         -> 2077       jaxpr, out_type, consts = trace_to_subjaxpr_dynamic2(fun, main, debug_info)
            2078     del fun, main

         /usr/local/lib/python3.8/dist-packages/jax/interpreters/partial_eval.py in trace_to_subjaxpr_dynamic2(fun, main, debug_info)
            2026     in_tracers_ = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
         -> 2027     ans = fun.call_wrapped(*in_tracers_)
            2028     out_tracers = map(trace.full_raise, ans)

         /usr/local/lib/python3.8/dist-packages/jax/linear_util.py in call_wrapped(self, *args, **kwargs)
             166     try:
         --> 167       ans = self.f(*args, **dict(self.params, **kwargs))
             168     except:

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/reductions.py in _reduce_sum(a, axis, dtype, out, keepdims, initial, where, promote_integers)
             205                 promote_integers: bool = True) -> Array:
         --> 206   return _reduction(a, "sum", np.sum, lax.add, 0, preproc=_cast_to_numeric,
             207                     bool_op=lax.bitwise_or, upcast_f16_for_computation=True,

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/reductions.py in _reduction(a, name, np_fun, op, init_val, has_identity, preproc, bool_op, upcast_f16_for_computation, axis, dtype, out, keepdims, initial, where_, parallel_reduce, promote_integers)
              83     raise NotImplementedError(f"The 'out' argument to jnp.{name} is not supported.")
         ---> 84   _check_arraylike(name, a)
              85   lax_internal._check_user_dtype_supported(dtype, name)

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/util.py in _check_arraylike(fun_name, *args)
             344     msg = "{} requires ndarray or scalar arguments, got {} at position {}."
         --> 345     raise TypeError(msg.format(fun_name, type(arg), pos))
             346 

         UnfilteredStackTrace: TypeError: sum requires ndarray or scalar arguments, got <class 'list'> at position 0.

         The stack trace below excludes JAX-internal frames.
         The preceding is the original exception that occurred, unmodified.

         --------------------

         The above exception was the direct cause of the following exception:

         TypeError                                 Traceback (most recent call last)
         <ipython-input-16-730cb94339bb> in <module>
         ----> 1 jnp.sum([1, 2, 3])

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/reductions.py in sum(a, axis, dtype, out, keepdims, initial, where, promote_integers)
             214         out: None = None, keepdims: bool = False, initial: Optional[ArrayLike] = None,
             215         where: Optional[ArrayLike] = None, promote_integers: bool = True) -> Array:
         --> 216   return _reduce_sum(a, axis=_ensure_optional_axes(axis), dtype=dtype, out=out,
             217                      keepdims=keepdims, initial=initial, where=where,
             218                      promote_integers=promote_integers)

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/reductions.py in _reduce_sum(a, axis, dtype, out, keepdims, initial, where, promote_integers)
             204                 initial: Optional[ArrayLike] = None, where: Optional[ArrayLike] = None,
             205                 promote_integers: bool = True) -> Array:
         --> 206   return _reduction(a, "sum", np.sum, lax.add, 0, preproc=_cast_to_numeric,
             207                     bool_op=lax.bitwise_or, upcast_f16_for_computation=True,
             208                     axis=axis, dtype=dtype, out=out, keepdims=keepdims,

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/reductions.py in _reduction(a, name, np_fun, op, init_val, has_identity, preproc, bool_op, upcast_f16_for_computation, axis, dtype, out, keepdims, initial, where_, parallel_reduce, promote_integers)
              82   if out is not None:
              83     raise NotImplementedError(f"The 'out' argument to jnp.{name} is not supported.")
         ---> 84   _check_arraylike(name, a)
              85   lax_internal._check_user_dtype_supported(dtype, name)
              86   axis = core.concrete_or_error(None, axis, f"axis argument to jnp.{name}().")

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/util.py in _check_arraylike(fun_name, *args)
             343                     if not _arraylike(arg))
             344     msg = "{} requires ndarray or scalar arguments, got {} at position {}."
         --> 345     raise TypeError(msg.format(fun_name, type(arg), pos))
             346 
             347 

         TypeError: sum requires ndarray or scalar arguments, got <class 'list'> at position 0.

.. container:: cell markdown

   이는 의도적인 설계의 선택으로, 그 이유는 추적된 함수에 리스트나
   튜플을 전달하게 되면 감지하기 어려운 조용한 성능의 저하가 유도될 수
   있기 때문입니다.

   예를 들어, 리스트의 입력을 허용하는 다음 버전의 ``jnp.sum``\ 을
   고려해봅시다.

   -> (2차 검수) 이는 의도적으로 설계된 결과입니다. 왜냐하면 추적된
   함수에 리스트나 튜플을 전달하게 되면 감지하기 어려운 조용한 성능의
   저하가 유도될 수 있기 때문입니다.

.. container:: cell code

   .. code:: python

      def permissive_sum(x):
        return jnp.sum(jnp.array(x))

      x = list(range(10))
      permissive_sum(x)

   .. container:: output execute_result

      ::

         DeviceArray(45, dtype=int32)

.. container:: cell markdown

   결과는 우리가 예상한 대로이지만 여기에는 잠재적인 성능 문제가 숨겨져
   있습니다. JAX의 추적 및 JIT 컴파일 모델에서 Python의 리스트 혹은
   튜플의 각 요소는 별도의 JAX 변수로 취급되며 이는 개별적으로 처리되어
   디바이스로 전송됩니다. 이는 위의 ``permissive_sum`` 함수에 대한
   jaxpr에서 볼 수 있습니다.

.. container:: cell code

   .. code:: python

      make_jaxpr(permissive_sum)(x)

   .. container:: output execute_result

      ::

         { lambda ; a:i32[] b:i32[] c:i32[] d:i32[] e:i32[] f:i32[] g:i32[] h:i32[] i:i32[]
             j:i32[]. let
             k:i32[] = convert_element_type[new_dtype=int32 weak_type=False] a
             l:i32[] = convert_element_type[new_dtype=int32 weak_type=False] b
             m:i32[] = convert_element_type[new_dtype=int32 weak_type=False] c
             n:i32[] = convert_element_type[new_dtype=int32 weak_type=False] d
             o:i32[] = convert_element_type[new_dtype=int32 weak_type=False] e
             p:i32[] = convert_element_type[new_dtype=int32 weak_type=False] f
             q:i32[] = convert_element_type[new_dtype=int32 weak_type=False] g
             r:i32[] = convert_element_type[new_dtype=int32 weak_type=False] h
             s:i32[] = convert_element_type[new_dtype=int32 weak_type=False] i
             t:i32[] = convert_element_type[new_dtype=int32 weak_type=False] j
             u:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] k
             v:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] l
             w:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] m
             x:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] n
             y:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] o
             z:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] p
             ba:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] q
             bb:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] r
             bc:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] s
             bd:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] t
             be:i32[10] = concatenate[dimension=0] u v w x y z ba bb bc bd
             bf:i32[] = reduce_sum[axes=(0,)] be
           in (bf,) }

.. container:: cell markdown

   리스트의 각 항목은 별도의 입력으로 처리되므로 리스트 크기에 따라
   선형적으로 증가하는 추적 및 컴파일의 오버헤드가 발생합니다. 이와 같은
   일을 방지하기 위해서 JAX는 임시적으로 리스트와 튜플을 배열로 임시적인
   변환을 피합니다.

   튜플 또는 리스트를 JAX 함수에 전달하려면 먼저 명시적으로 배열로
   변환한 후 전달하면 됩니다.

   -> (2차 검수) 이와 같은 일을 방지하기 위해서 JAX는 리스트 및 튜플을
   배열로 임시적으로 변환하는 것을 피합니다.

.. container:: cell code

   .. code:: python

      jnp.sum(jnp.array(x))

   .. container:: output execute_result

      ::

         DeviceArray(45, dtype=int32)

.. container:: cell markdown

   .. rubric:: ## 🔪 난수
      :name: --난수

      If all scientific papers whose results are in doubt because of
      bad rand()s were to disappear from library shelves, there would be
      a gap on each shelf about as big as your fist. - Numerical Recipes

   ..

      나쁜 rand()로 인해 결과가 의심스러운 모든 과학 논문들이 도서관
      책장에서 사라진다면 각 책장에는 주먹만한 간격이 생길 겁니다. -
      Numerical Recipes

.. container:: cell markdown

   **RNGs와 State**

   여러분들은 NumPy 및 기타 라이브러리의 상태 유지적인 의사 난수
   생성기(PRNG)에 익숙할 것입니다. 이 라이브러리들은 의사 난수의 소스를
   제공하기 위해 많은 세부 정보들을 백그라운드에서 유용하게 숨깁니다.

   --> **(1차 검수)** 여러분들은 NumPy 및 기타 라이브러리의 스테이트풀
   의사 난수 생성기(PRNG)에 익숙할 것입니다. 이 라이브러리들은 의사
   난수의 소스를 제공하기 위해 많은 세부 정보들을 백그라운드에서
   유용하게 숨깁니다.

.. container:: cell code

   .. code:: python

      print(np.random.random())
      print(np.random.random())
      print(np.random.random())

   .. container:: output stream stdout

      ::

         0.09578029358870122
         0.11225196191621833
         0.7374410423665646

.. container:: cell markdown

   백그라운드에서 numpy는 Mersenne Twister PRNG를 사용하여 의사 난수
   기능을 강화합니다. PRNG의 주기는 :math:`2^{19937} - 1`\ 이고 어느
   시점에서든 624개의 32비트 부호 없는 정수와 이 “엔트로피”가 얼마나
   많이 사용되었는지에 대한 위치로 설명할 수 있습니다.

.. container:: cell code

   .. code:: python

      np.random.seed(0)
      rng_state = np.random.get_state()
      # print(rng_state)
      # --> ('MT19937', array([0, 1, 1812433255, 1900727105, 1208447044,
      #       2481403966, 4042607538,  337614300, ... 614 more numbers...,
      #       3048484911, 1796872496], dtype=uint32), 624, 0, 0.0)

.. container:: cell markdown

   이 의사 난수 상태 벡터는 난수가 필요할 때마다 백그라운드에서
   자동적으로 업데이트되어 Mersenne twister 상태 벡터의 uint32 중 2개를
   “소비”합니다.

.. container:: cell code

   .. code:: python

      _ = np.random.uniform()
      rng_state = np.random.get_state()
      #print(rng_state)
      # --> ('MT19937', array([2443250962, 1093594115, 1878467924,
      #       ..., 2648828502, 1678096082], dtype=uint32), 2, 0, 0.0)

      # Let's exhaust the entropy in this PRNG statevector
      for i in range(311):
        _ = np.random.uniform()
      rng_state = np.random.get_state()
      #print(rng_state)
      # --> ('MT19937', array([2443250962, 1093594115, 1878467924,
      #       ..., 2648828502, 1678096082], dtype=uint32), 624, 0, 0.0)

      # Next call iterates the RNG state for a new batch of fake "entropy".
      _ = np.random.uniform()
      rng_state = np.random.get_state()
      # print(rng_state)
      # --> ('MT19937', array([1499117434, 2949980591, 2242547484,
      #      4162027047, 3277342478], dtype=uint32), 2, 0, 0.0)

.. container:: cell markdown

   Magic PRNG 상태의 문제는 서로 다른 스레드, 프로세스 및 장치에서 사용
   및 업데이트 되는 방식에 대해 추론하기 어렵고 엔트로피 생성 및 소비에
   대한 세부 정보가 최종 사용자에게 숨겨져 있을 때 문제를 일으키기 매우
   쉽다는 것입니다.

   Mersenne Twister PRNG는 또한 많은 문제가 있는 것으로 알려져 있으며,
   2.5Kb의 큰 상태 크기를 가지고 있어 초기화 문제를 야기할 수 있습니다.
   또한, 최신 BigCrush 테스트를 만족하지 못하고 일반적으로 느리다는
   단점이 있습니다.

.. container:: cell markdown

   **JAX PRNG**

   JAX는 대신 PRNG 상태를 명시적으로 전달하고 반복하여 엔트로피 생성 및
   소비를 처리하는 명시적 PRNG를 구현했습니다. JAX는 분할 가능한 최신
   Threefry counter 기반 PRNG를 사용합니다(`Threefry counter-based
   PRNG <https://github.com/google/jax/blob/main/docs/jep/263-prng.md>`__).
   즉, 이러한 설계를 통해 PRNG 상태를 병렬 확률적 생성을 위해 사용하기
   위해 새로운 PRNG로 분기할 수 있습니다.

   무작위 상태는 키라고 부르는 두 개의 unsigned-int32로 설명됩니다.

.. container:: cell code

   .. code:: python

      from jax import random
      key = random.PRNGKey(0)
      key

   .. container:: output execute_result

      ::

         DeviceArray([0, 0], dtype=uint32)

.. container:: cell markdown

   JAX의 임의 함수는 PRNG 상태에서 의사 난수를 생성하지만 상태를
   변경하지는 않습니다!

   동일한 상태를 재사용하는 것은 **sadness**\ 와 **monotony**\ 가
   발생하여 최종 사용자에게 **lifegiving chaos**\ 이 발생하지 않습니다.
   ..? → 제안 부탁드려요.

   (Reusing the same state will cause sadness and monotony, depriving
   the end user of lifegiving chaos:)

   -> (2차 검수) 동일한 상태를 재사용하는 행위는 슬픔과 단조로움을
   유발하며 결국 최종 사용자에게 생기를 불어넣는 혼돈을 빼앗아갑니다!

.. container:: cell code

   .. code:: python

      print(random.normal(key, shape=(1,)))
      print(key)
      # No no no!
      print(random.normal(key, shape=(1,)))
      print(key)

   .. container:: output stream stdout

      ::

         [-0.20584226]
         [0 0]
         [-0.20584226]
         [0 0]

.. container:: cell markdown

   대신, 새로운 의사 난수가 필요할 때마다 PRNG를 분할하여 사용 가능한
   하위 키를 얻습니다.

.. container:: cell code

   .. code:: python

      print("old key", key)
      key, subkey = random.split(key)
      normal_pseudorandom = random.normal(subkey, shape=(1,))
      print("    \---SPLIT --> new key   ", key)
      print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)

   .. container:: output stream stdout

      ::

         old key [0 0]
             \---SPLIT --> new key    [4146024105  967050713]
                      \--> new subkey [2718843009 1272950319] --> normal [-1.2515389]

.. container:: cell markdown

   새로운 난수가 필요할 때마다 키를 전파하고 새 하위 키를 만듭니다.

.. container:: cell code

   .. code:: python

      print("old key", key)
      key, subkey = random.split(key)
      normal_pseudorandom = random.normal(subkey, shape=(1,))
      print("    \---SPLIT --> new key   ", key)
      print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)

   .. container:: output stream stdout

      ::

         old key [4146024105  967050713]
             \---SPLIT --> new key    [2384771982 3928867769]
                      \--> new subkey [1278412471 2182328957] --> normal [-0.58665055]

.. container:: cell markdown

   한 번에 둘 이상의 하위키를 만들 수 있습니다.

.. container:: cell code

   .. code:: python

      key, *subkeys = random.split(key, 4)
      for subkey in subkeys:
        print(random.normal(subkey, shape=(1,)))

   .. container:: output stream stdout

      ::

         [-0.37533438]
         [0.98645043]
         [0.14553197]

.. container:: cell markdown

   .. rubric:: **🔪 제어 흐름**
      :name: -제어-흐름

   --------------

   **✔ python control_flow + autodiff ✔**

   Python 함수에 ``grad``\ 를 적용하려는 경우 Autograd(또는 Pytorch 또는
   TF Eager)를 사용하는 것처럼 문제 없이 일반적인 Python 제어 흐름
   구성을 사용할 수 있습니다.

.. container:: cell code

   .. code:: python

      def f(x):
        if x < 3:
          return 3. * x ** 2
        else:
          return -4 * x

      print(grad(f)(2.))  # ok!
      print(grad(f)(4.))  # ok!

   .. container:: output stream stdout

      ::

         12.0
         -4.0

.. container:: cell markdown

   **Python control flow + JIT**

   ``jit``\ 와 함께 제어 흐름을 사용하는 것은 더 복잡하며 기본적으로 더
   많은 제약이 있습니다.

   이 예시는 동작합니다.

.. container:: cell code

   .. code:: python

      @jit
      def f(x):
        for i in range(3):
          x = 2 * x
        return x

      print(f(3))

   .. container:: output stream stdout

      ::

         24

.. container:: cell markdown

   아래 예시도 동작합니다.

.. container:: cell code

   .. code:: python

      @jit
      def g(x):
        y = 0.
        for i in range(x.shape[0]):
          y = y + x[i]
        return y

      print(g(jnp.array([1., 2., 3.])))

   .. container:: output stream stdout

      ::

         6.0

.. container:: cell markdown

   하지만 이 예시는 기본적으로 동작하지 않습니다.

.. container:: cell code

   .. code:: python

      @jit
      def f(x):
        if x < 3:
          return 3. * x ** 2
        else:
          return -4 * x

      # This will fail!
      f(2)

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         UnfilteredStackTrace                      Traceback (most recent call last)
         <ipython-input-31-fe5ae3470df9> in <module>
               8 # This will fail!
         ----> 9 f(2)

         /usr/local/lib/python3.8/dist-packages/jax/_src/traceback_util.py in reraise_with_filtered_traceback(*args, **kwargs)
             161     try:
         --> 162       return fun(*args, **kwargs)
             163     except Exception as e:

         /usr/local/lib/python3.8/dist-packages/jax/_src/api.py in cache_miss(*args, **kwargs)
             621         jax.config.jax_debug_nans or jax.config.jax_debug_infs):
         --> 622       execute = dispatch._xla_call_impl_lazy(fun_, *tracers, **params)
             623       out_flat = call_bind_continuation(execute(*args_flat))

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in _xla_call_impl_lazy(***failed resolving arguments***)
             235     arg_specs = [(None, getattr(x, '_device', None)) for x in args]
         --> 236   return xla_callable(fun, device, backend, name, donated_invars, keep_unused,
             237                       *arg_specs)

         /usr/local/lib/python3.8/dist-packages/jax/linear_util.py in memoized_fun(fun, *args)
             302     else:
         --> 303       ans = call(fun, *args)
             304       cache[key] = (ans, fun.stores)

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in _xla_callable_uncached(fun, device, backend, name, donated_invars, keep_unused, *arg_specs)
             358   else:
         --> 359     return lower_xla_callable(fun, device, backend, name, donated_invars, False,
             360                               keep_unused, *arg_specs).compile().unsafe_call

         /usr/local/lib/python3.8/dist-packages/jax/_src/profiler.py in wrapper(*args, **kwargs)
             313     with TraceAnnotation(name, **decorator_kwargs):
         --> 314       return func(*args, **kwargs)
             315     return wrapper

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in lower_xla_callable(fun, device, backend, name, donated_invars, always_lower, keep_unused, *arg_specs)
             444                         "for jit in {elapsed_time} sec"):
         --> 445     jaxpr, out_type, consts = pe.trace_to_jaxpr_final2(
             446         fun, pe.debug_info_final(fun, "jit"))

         /usr/local/lib/python3.8/dist-packages/jax/_src/profiler.py in wrapper(*args, **kwargs)
             313     with TraceAnnotation(name, **decorator_kwargs):
         --> 314       return func(*args, **kwargs)
             315     return wrapper

         /usr/local/lib/python3.8/dist-packages/jax/interpreters/partial_eval.py in trace_to_jaxpr_final2(fun, debug_info)
            2076     with core.new_sublevel():
         -> 2077       jaxpr, out_type, consts = trace_to_subjaxpr_dynamic2(fun, main, debug_info)
            2078     del fun, main

         /usr/local/lib/python3.8/dist-packages/jax/interpreters/partial_eval.py in trace_to_subjaxpr_dynamic2(fun, main, debug_info)
            2026     in_tracers_ = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
         -> 2027     ans = fun.call_wrapped(*in_tracers_)
            2028     out_tracers = map(trace.full_raise, ans)

         /usr/local/lib/python3.8/dist-packages/jax/linear_util.py in call_wrapped(self, *args, **kwargs)
             166     try:
         --> 167       ans = self.f(*args, **dict(self.params, **kwargs))
             168     except:

         <ipython-input-31-fe5ae3470df9> in f(x)
               2 def f(x):
         ----> 3   if x < 3:
               4     return 3. * x ** 2

         /usr/local/lib/python3.8/dist-packages/jax/core.py in __bool__(self)
             633   def __nonzero__(self): return self.aval._nonzero(self)
         --> 634   def __bool__(self): return self.aval._bool(self)
             635   def __int__(self): return self.aval._int(self)

         /usr/local/lib/python3.8/dist-packages/jax/core.py in error(self, arg)
            1266   def error(self, arg):
         -> 1267     raise ConcretizationTypeError(arg, fname_context)
            1268   return error

         UnfilteredStackTrace: jax._src.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: Traced<ShapedArray(bool[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
         The problem arose with the `bool` function. 
         The error occurred while tracing the function f at <ipython-input-31-fe5ae3470df9>:1 for jit. This concrete value was not available in Python because it depends on the value of the argument 'x'.

         See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError

         The stack trace below excludes JAX-internal frames.
         The preceding is the original exception that occurred, unmodified.

         --------------------

         The above exception was the direct cause of the following exception:

         ConcretizationTypeError                   Traceback (most recent call last)
         <ipython-input-31-fe5ae3470df9> in <module>
               7 
               8 # This will fail!
         ----> 9 f(2)

         <ipython-input-31-fe5ae3470df9> in f(x)
               1 @jit
               2 def f(x):
         ----> 3   if x < 3:
               4     return 3. * x ** 2
               5   else:

         ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: Traced<ShapedArray(bool[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
         The problem arose with the `bool` function. 
         The error occurred while tracing the function f at <ipython-input-31-fe5ae3470df9>:1 for jit. This concrete value was not available in Python because it depends on the value of the argument 'x'.

         See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError

.. container:: cell markdown

   **왜 그럴까!?**

   함수를 ``jit`` 컴파일할 때 일반적으로 컴파일된 코드를 캐시하고 재사용
   할 수 있도록 다양한 인수 값에 대해 작동하는 함수 버전을 컴파일하려고
   합니다. 이러한 방식으로 각 함수 평가마다 다시 컴파일할 필요가
   없습니다.

   예를 들어 ``jnp.array([1., 2., 3.], jnp.float32)`` 배열에서 ``@jit``
   함수를 평가하기 위해 ``jnp.array([4., 5., 6.], jnp.float32)``\ 에서
   사용했던 코드를 컴파일 하여 컴파일 시간을 절약할 수 있습니다.

   -> (2차 검수) 예를 들어 ``jnp.array([1., 2., 3.], jnp.float32)``
   배열에서 ``@jit`` 함수를 평가하기 위해
   ``jnp.array([4., 5., 6.], jnp.float32)``\ 에서 사용했던 컴파일된
   코드를 재사용하여 하여 컴파일에 수행되는 시간을 절약할 수 있습니다.

   Python 코드의 다양한 인수 값에 유효한 뷰를 얻기 위해 JAX는 가능한
   입력 집합을 나타내는 추상 값으로 코드를 추적합니다. 다양한 추상화
   수준이 있으며 서로 다른 변환은 서로 다른 추상화 수준을 사용합니다.

   기본적으로 ``jit``\ 은 ``ShapedArray`` 추상화 수준에서 코드를
   추적합니다. 여기서 각 추상 값은 고정된 모양과 dtype이 있는 모든 배열
   값의 집합을 나타냅니다, 예를 들어 추상 값
   ``ShapedAray((3,), jnp.float32)``\ 를 사용하여 추적하면 해당 배열
   세트의 구체적인 값에 대해 재사용할 수 있는 함수의 뷰를 얻을 수
   있습니다. 즉, 컴파일 시간을 줄일 수 있다는 것을 의미합니다.

   그러나 여기에는 장단점이 있습니다. 특정 구체적인 값이 결정되지 않은
   ``ShapedArray((), jnp.float32)``\ 에서 Python 함수를 추적하는 경우
   ``if x < 3``\ 과 같은 줄에 도달하면 표현식 ``x < 3``\ 은
   ``{True, False}`` 집합을 나타내는 추상
   ``ShapedArray((), jnp.bool_)``\ 로 평가됩니다. Python이 이를 구체적인
   ``True`` 또는 ``False``\ 로 강제하려고 하면 오류가 발생합니다. 어떤
   분기를 선택해야 할지 모르고 추적을 계속할 수 없습니다! 단점은 추상화
   수준이 높을수록 Python 코드에 대한 보다 일반적인 뷰를 얻을 수
   있지만(따라서 재컴파일을 줄일 수 있습니다.) 추적을 완료하려면 Python
   코드에 더 많은 제약이 필요하다는 것입니다.

   좋은 소식은 이 트레이드오프를 직접 제어할 수 있다는 것입니다. 보다
   정밀한 추상 값에 대한 ``jit`` 추적을 통해 추적 가능성 제약을 완화할
   수 있습니다. 예를 들어 ``jit``\ 에 ``static_argnums`` 인수를 사용하여
   일부 인수의 구체적인 값을 추적하도록 지정할 수 있습니다. 다음은 해당
   예제 함수입니다.

.. container:: cell code

   .. code:: python

      def f(x):
        if x < 3:
          return 3. * x ** 2
        else:
          return -4 * x

      f = jit(f, static_argnums=(0,))

      print(f(2.))

   .. container:: output stream stdout

      ::

         12.0

.. container:: cell markdown

   루프를 포함한 또다른 예제입니다.

.. container:: cell code

   .. code:: python

      def f(x, n):
        y = 0.
        for i in range(n):
          y = y + x[i]
        return y

      f = jit(f, static_argnums=(1,))

      f(jnp.array([2., 3., 4.]), 2)

   .. container:: output execute_result

      ::

         DeviceArray(5., dtype=float32)

.. container:: cell markdown

   static_argnums를 이용한 효과로 인해 루프가 정적으로 펼쳐집니다. JAX는
   또한 Unshaped와 같은 더 높은 수준의 추상화에서 추적할 수 있지만 현재
   변환의 기본값은 아닙니다.

   **⚠️ 인수 값에 의존하는 모양을 가진 함수**

   -> (2차 검수) **⚠️ 인수 값에 따라 형태가 바뀌는 함수**

   이러한 제어 흐름 문제는 보다 미묘한 방식으로도 나타납니다. jit 하려는
   수치 함수는 내부 배열의 모양을 인수 값에 따라 특정할 수 없습니다(인수
   모양에 따라 특정하는 것은 괜찮습니다). 간단한 예로 입력 변수
   ``길이``\ 에 따라 출력이 달라지는 함수를 만들어 보겠습니다.

.. container:: cell code

   .. code:: python

      def example_fun(length, val):
        return jnp.ones((length,)) * val
      # un-jit'd works fine
      print(example_fun(5, 4))

   .. container:: output stream stdout

      ::

         [4. 4. 4. 4. 4.]

.. container:: cell code

   .. code:: python

      bad_example_jit = jit(example_fun)
      # this will fail:
      bad_example_jit(10, 4)

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         UnfilteredStackTrace                      Traceback (most recent call last)
         <ipython-input-35-c0b9effea12c> in <module>
               2 # this will fail:
         ----> 3 bad_example_jit(10, 4)

         /usr/local/lib/python3.8/dist-packages/jax/_src/traceback_util.py in reraise_with_filtered_traceback(*args, **kwargs)
             161     try:
         --> 162       return fun(*args, **kwargs)
             163     except Exception as e:

         /usr/local/lib/python3.8/dist-packages/jax/_src/api.py in cache_miss(*args, **kwargs)
             621         jax.config.jax_debug_nans or jax.config.jax_debug_infs):
         --> 622       execute = dispatch._xla_call_impl_lazy(fun_, *tracers, **params)
             623       out_flat = call_bind_continuation(execute(*args_flat))

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in _xla_call_impl_lazy(***failed resolving arguments***)
             235     arg_specs = [(None, getattr(x, '_device', None)) for x in args]
         --> 236   return xla_callable(fun, device, backend, name, donated_invars, keep_unused,
             237                       *arg_specs)

         /usr/local/lib/python3.8/dist-packages/jax/linear_util.py in memoized_fun(fun, *args)
             302     else:
         --> 303       ans = call(fun, *args)
             304       cache[key] = (ans, fun.stores)

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in _xla_callable_uncached(fun, device, backend, name, donated_invars, keep_unused, *arg_specs)
             358   else:
         --> 359     return lower_xla_callable(fun, device, backend, name, donated_invars, False,
             360                               keep_unused, *arg_specs).compile().unsafe_call

         /usr/local/lib/python3.8/dist-packages/jax/_src/profiler.py in wrapper(*args, **kwargs)
             313     with TraceAnnotation(name, **decorator_kwargs):
         --> 314       return func(*args, **kwargs)
             315     return wrapper

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in lower_xla_callable(fun, device, backend, name, donated_invars, always_lower, keep_unused, *arg_specs)
             444                         "for jit in {elapsed_time} sec"):
         --> 445     jaxpr, out_type, consts = pe.trace_to_jaxpr_final2(
             446         fun, pe.debug_info_final(fun, "jit"))

         /usr/local/lib/python3.8/dist-packages/jax/_src/profiler.py in wrapper(*args, **kwargs)
             313     with TraceAnnotation(name, **decorator_kwargs):
         --> 314       return func(*args, **kwargs)
             315     return wrapper

         /usr/local/lib/python3.8/dist-packages/jax/interpreters/partial_eval.py in trace_to_jaxpr_final2(fun, debug_info)
            2076     with core.new_sublevel():
         -> 2077       jaxpr, out_type, consts = trace_to_subjaxpr_dynamic2(fun, main, debug_info)
            2078     del fun, main

         /usr/local/lib/python3.8/dist-packages/jax/interpreters/partial_eval.py in trace_to_subjaxpr_dynamic2(fun, main, debug_info)
            2026     in_tracers_ = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
         -> 2027     ans = fun.call_wrapped(*in_tracers_)
            2028     out_tracers = map(trace.full_raise, ans)

         /usr/local/lib/python3.8/dist-packages/jax/linear_util.py in call_wrapped(self, *args, **kwargs)
             166     try:
         --> 167       ans = self.f(*args, **dict(self.params, **kwargs))
             168     except:

         <ipython-input-34-16ef497a37f9> in example_fun(length, val)
               1 def example_fun(length, val):
         ----> 2   return jnp.ones((length,)) * val
               3 # un-jit'd works fine

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in ones(shape, dtype)
            2104     raise TypeError("expected sequence object with len >= 0 or a single integer")
         -> 2105   shape = canonicalize_shape(shape)
            2106   lax_internal._check_user_dtype_supported(dtype, "ones")

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in canonicalize_shape(shape, context)
              98   else:
         ---> 99     return core.canonicalize_shape(shape, context)  # type: ignore
             100 

         /usr/local/lib/python3.8/dist-packages/jax/core.py in canonicalize_shape(shape, context)
            1898     pass
         -> 1899   raise _invalid_shape_error(shape, context)
            1900 

         UnfilteredStackTrace: TypeError: Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>,).
         If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.

         The stack trace below excludes JAX-internal frames.
         The preceding is the original exception that occurred, unmodified.

         --------------------

         The above exception was the direct cause of the following exception:

         TypeError                                 Traceback (most recent call last)
         <ipython-input-35-c0b9effea12c> in <module>
               1 bad_example_jit = jit(example_fun)
               2 # this will fail:
         ----> 3 bad_example_jit(10, 4)

         <ipython-input-34-16ef497a37f9> in example_fun(length, val)
               1 def example_fun(length, val):
         ----> 2   return jnp.ones((length,)) * val
               3 # un-jit'd works fine
               4 print(example_fun(5, 4))

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in ones(shape, dtype)
            2103   if isinstance(shape, types.GeneratorType):
            2104     raise TypeError("expected sequence object with len >= 0 or a single integer")
         -> 2105   shape = canonicalize_shape(shape)
            2106   lax_internal._check_user_dtype_supported(dtype, "ones")
            2107   return lax.full(shape, 1, _jnp_dtype(dtype))

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in canonicalize_shape(shape, context)
              97     return core.canonicalize_shape((shape,), context)  # type: ignore
              98   else:
         ---> 99     return core.canonicalize_shape(shape, context)  # type: ignore
             100 
             101 # Common docstring additions:

         TypeError: Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>,).
         If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.

.. container:: cell code

   .. code:: python

      # static_argnums tells JAX to recompile on changes at these argument positions:
      good_example_jit = jit(example_fun, static_argnums=(0,))
      # first compile
      print(good_example_jit(10, 4))
      # recompiles
      print(good_example_jit(5, 4))

   .. container:: output stream stdout

      ::

         [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
         [4. 4. 4. 4. 4.]

.. container:: cell markdown

   ``static_argnums``\ 는 예제에서 ``길이``\ 가 거의 변경되지 않는 경우
   편리할 수 있지만 많이 변경되면 재앙이 될 것입니다!

   --> **(1차 검수)** ``static_argnums``\ 는 예제에서 ``길이``\ 의
   변경이 잦지 않은 경우에는 편리할 수 있지만 변경이 잦은 경우 재앙이 될
   수 있습니다!

   마지막으로 함수에 전역적인 부작용이 있는 경우 JAX의 추적 프로그램으로
   인해 이상한 일이 발생할 수 있습니다. 일반적인 문제는 **jit**'d 함수
   내에서 배열을 출력하려고 시도하는 것입니다.

   --> **(1차 검수)** 마지막으로 함수에 전역적인 부수효과들이 있는 경우
   JAX의 추적 프로그램으로 인해 이상한 일이 발생할 수 있습니다. 일반적인
   문제는 **jit**'d 함수 내에서 배열을 출력하려고 시도할 때 발생할 수
   있습니다.

.. container:: cell code

   .. code:: python

      @jit
      def f(x):
        print(x)
        y = 2 * x
        print(y)
        return y
      f(2)

   .. container:: output stream stdout

      ::

         Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
         Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>

   .. container:: output execute_result

      ::

         DeviceArray(4, dtype=int32, weak_type=True)

.. container:: cell markdown

   .. rubric:: 구조적 제어 흐름 프리미티브
      :name: 구조적-제어-흐름-프리미티브

   JAX에는 제어 흐름에 대한 다양한 옵션들이 많이 있습니다. 재컴파일을
   피하고 싶지만 여전히 추적 가능한 제어 흐름을 사용하고 싶고 큰 루프를
   펼치고 싶지 않다고 가정합시다. 그럼 다음 4개의 구조화된 제어 흐름
   프리미티브를 사용할 수 있습니다.

   -> (2차 검수) 예를 들어 재컴파일을 피하고 추적 가능한 제어 흐름을
   사용하면서 큰 루프를 풀고 싶지 않다면 아래의 4가지 구조적 제어 흐름
   기본 구조를 사용할 수 있습니다.

   -  ``lax.cond`` *differentiable*
   -  ``lax.while_loop`` **fwd-mode-differentiable**
   -  ``lax.fori_loop`` **fwd-mode-differentiable** in general; **fwd
      and rev-mode differentiable** if endpoints are static.
   -  ``lax.scan`` *differentiable*

.. container:: cell markdown

   .. rubric:: cond
      :name: cond

.. container:: cell code

   .. code:: python

      def cond(pred, true_fun, false_fun, operand):
        if pred:
          return true_fun(operand)
        else:
          return false_fun(operand)

.. container:: cell code

   .. code:: python

      from jax import lax

      operand = jnp.array([0.])
      lax.cond(True, lambda x: x+1, lambda x: x-1, operand)
      # --> array([1.], dtype=float32)
      lax.cond(False, lambda x: x+1, lambda x: x-1, operand)
      # --> array([-1.], dtype=float32)

   .. container:: output execute_result

      ::

         DeviceArray([-1.], dtype=float32)

.. container:: cell markdown

   ``jax.lax``\ 에는 동적 조건에 따라 분기할 수 있는 다른 두 개의 함수가
   제공됩니다.

   -  ``lax.select``\ 는 ``lax.cond``\ 의 배치 버전이지만, 선택지는
      이전에 계산된 배열로 표현됩니다.
   -  ``lax.switch``\ 는 ``lax.cond``\ 와 유사하지만, 어떤 수의 호출
      가능한 선택지 사이에 전환할 수 있습니다.

   또한, ``jax.numpy``\ 에서는 이러한 함수에 대한 다수의 Numpy 스타일
   인터페이스가 제공됩니다.

   -  ``jnp.where``\ 는 3개의 인수가있는 lax.select의 Numpy 스타일
      래퍼입니다.
   -  ``jnp.piecewise``\ 는 ``lax.switch``\ 의 Numpy 스타일 래퍼이지만,
      단일 스칼라 인덱스 대신에 불리언 조건의 목록에 따라 전환합니다.
   -  ``jnp.select``\ 는 ``jnp.piecewise``\ 와 유사한 API를 가지지만,
      선택지는 사전 계산된 배열로 제공됩니다. 결과적으로
      ``lax.select``\ 의 여러 호출로 구현됩니다.

.. container:: cell markdown

   .. rubric:: while_loop
      :name: while_loop

.. container:: cell code

   .. code:: python

      def while_loop(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
          val = body_fun(val)
        return val

.. container:: cell code

   .. code:: python

      init_val = 0
      cond_fun = lambda x: x<10
      body_fun = lambda x: x+1
      lax.while_loop(cond_fun, body_fun, init_val)
      # --> array(10, dtype=int32)

   .. container:: output execute_result

      ::

         DeviceArray(10, dtype=int32, weak_type=True)

.. container:: cell markdown

   .. rubric:: fori_loop
      :name: fori_loop

.. container:: cell code

   .. code:: python

      def fori_loop(start, stop, body_fun, init_val):
        val = init_val
        for i in range(start, stop):
          val = body_fun(i, val)
        return val

.. container:: cell code

   .. code:: python

      init_val = 0
      start = 0
      stop = 10
      body_fun = lambda i,x: x+i
      lax.fori_loop(start, stop, body_fun, init_val)
      # --> array(45, dtype=int32)

   .. container:: output execute_result

      ::

         DeviceArray(45, dtype=int32, weak_type=True)

.. container:: cell markdown

   .. rubric:: Summary
      :name: summary

.. container:: cell markdown

   .. image:: vertopal_76abf758640444e9a36d970ec61687c4/3402a7b331e93550dbf9c52c26cd865cd6f59a1f.png
      :alt: 스크린샷 2023-02-05 오후 10.08.19.png

.. container:: cell markdown

   .. rubric:: **🔪 동적 형태**
      :name: -동적-형태

   --------------

   ``jax.jit``, ``jax.vmap``, ``jax.grad`` 등과 같은 변환 내에서
   사용되는 JAX 코드는 모든 출력 배열과 중간 배열이 정적 모양을 가져야
   합니다. 즉, 모양은 다른 배열 내의 값에 의존하지 않아야 합니다.

   예를 들어, ``jnp.nansum``\ 의 자체 버전을 구현하는 경우 다음과 같이
   시작할 수 있습니다.

   -> (2차 검수) 예를 들어, ``jnp.nansum``\ 의 버전을 직접 구현하려면
   다음과 같이 시작할 수 있습니다

.. container:: cell code

   .. code:: python

      def nansum(x):
        mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
        x_without_nans = x[mask]
        return x_without_nans.sum()

.. container:: cell markdown

   JIT 및 기타 변환 외부에서는 예상대로 작동합니다.

.. container:: cell code

   .. code:: python

      x = jnp.array([1, 2, jnp.nan, 3, 4])
      print(nansum(x))

   .. container:: output stream stdout

      ::

         10.0

.. container:: cell markdown

   jax.jit 또는 다른 변환을 이 함수에 적용하려고 하면 오류가 발생합니다.

.. container:: cell code

   .. code:: python

      jax.jit(nansum)(x)

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         UnfilteredStackTrace                      Traceback (most recent call last)
         <ipython-input-46-23b331ebedc0> in <module>
         ----> 1 jax.jit(nansum)(x)

         /usr/local/lib/python3.8/dist-packages/jax/_src/traceback_util.py in reraise_with_filtered_traceback(*args, **kwargs)
             161     try:
         --> 162       return fun(*args, **kwargs)
             163     except Exception as e:

         /usr/local/lib/python3.8/dist-packages/jax/_src/api.py in cache_miss(*args, **kwargs)
             621         jax.config.jax_debug_nans or jax.config.jax_debug_infs):
         --> 622       execute = dispatch._xla_call_impl_lazy(fun_, *tracers, **params)
             623       out_flat = call_bind_continuation(execute(*args_flat))

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in _xla_call_impl_lazy(***failed resolving arguments***)
             235     arg_specs = [(None, getattr(x, '_device', None)) for x in args]
         --> 236   return xla_callable(fun, device, backend, name, donated_invars, keep_unused,
             237                       *arg_specs)

         /usr/local/lib/python3.8/dist-packages/jax/linear_util.py in memoized_fun(fun, *args)
             302     else:
         --> 303       ans = call(fun, *args)
             304       cache[key] = (ans, fun.stores)

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in _xla_callable_uncached(fun, device, backend, name, donated_invars, keep_unused, *arg_specs)
             358   else:
         --> 359     return lower_xla_callable(fun, device, backend, name, donated_invars, False,
             360                               keep_unused, *arg_specs).compile().unsafe_call

         /usr/local/lib/python3.8/dist-packages/jax/_src/profiler.py in wrapper(*args, **kwargs)
             313     with TraceAnnotation(name, **decorator_kwargs):
         --> 314       return func(*args, **kwargs)
             315     return wrapper

         /usr/local/lib/python3.8/dist-packages/jax/_src/dispatch.py in lower_xla_callable(fun, device, backend, name, donated_invars, always_lower, keep_unused, *arg_specs)
             444                         "for jit in {elapsed_time} sec"):
         --> 445     jaxpr, out_type, consts = pe.trace_to_jaxpr_final2(
             446         fun, pe.debug_info_final(fun, "jit"))

         /usr/local/lib/python3.8/dist-packages/jax/_src/profiler.py in wrapper(*args, **kwargs)
             313     with TraceAnnotation(name, **decorator_kwargs):
         --> 314       return func(*args, **kwargs)
             315     return wrapper

         /usr/local/lib/python3.8/dist-packages/jax/interpreters/partial_eval.py in trace_to_jaxpr_final2(fun, debug_info)
            2076     with core.new_sublevel():
         -> 2077       jaxpr, out_type, consts = trace_to_subjaxpr_dynamic2(fun, main, debug_info)
            2078     del fun, main

         /usr/local/lib/python3.8/dist-packages/jax/interpreters/partial_eval.py in trace_to_subjaxpr_dynamic2(fun, main, debug_info)
            2026     in_tracers_ = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
         -> 2027     ans = fun.call_wrapped(*in_tracers_)
            2028     out_tracers = map(trace.full_raise, ans)

         /usr/local/lib/python3.8/dist-packages/jax/linear_util.py in call_wrapped(self, *args, **kwargs)
             166     try:
         --> 167       ans = self.f(*args, **dict(self.params, **kwargs))
             168     except:

         <ipython-input-44-e79431cd8284> in nansum(x)
               2   mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
         ----> 3   x_without_nans = x[mask]
               4   return x_without_nans.sum()

         /usr/local/lib/python3.8/dist-packages/jax/core.py in __getitem__(self, idx)
             631   def __rrshift__(self, other): return self.aval._rrshift(self, other)
         --> 632   def __getitem__(self, idx): return self.aval._getitem(self, idx)
             633   def __nonzero__(self): return self.aval._nonzero(self)

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in _rewriting_take(arr, idx, indices_are_sorted, unique_indices, mode, fill_value)
            3814 
         -> 3815   treedef, static_idx, dynamic_idx = _split_index_for_jit(idx, arr.shape)
            3816   return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in _split_index_for_jit(idx, shape)
            3893   # indexing logic to handle them.
         -> 3894   idx = _expand_bool_indices(idx, shape)
            3895 

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in _expand_bool_indices(idx, shape)
            4206         # TODO(mattjj): improve this error by tracking _why_ the indices are not concrete
         -> 4207         raise errors.NonConcreteBooleanIndexError(abstract_i)
            4208       elif _ndim(i) == 0:

         UnfilteredStackTrace: jax._src.errors.NonConcreteBooleanIndexError: Array boolean indices must be concrete; got ShapedArray(bool[5])

         See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError

         The stack trace below excludes JAX-internal frames.
         The preceding is the original exception that occurred, unmodified.

         --------------------

         The above exception was the direct cause of the following exception:

         NonConcreteBooleanIndexError              Traceback (most recent call last)
         <ipython-input-46-23b331ebedc0> in <module>
         ----> 1 jax.jit(nansum)(x)

         <ipython-input-44-e79431cd8284> in nansum(x)
               1 def nansum(x):
               2   mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
         ----> 3   x_without_nans = x[mask]
               4   return x_without_nans.sum()

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in _rewriting_take(arr, idx, indices_are_sorted, unique_indices, mode, fill_value)
            3813         return lax.dynamic_index_in_dim(arr, idx, keepdims=False)
            3814 
         -> 3815   treedef, static_idx, dynamic_idx = _split_index_for_jit(idx, arr.shape)
            3816   return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
            3817                  unique_indices, mode, fill_value)

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in _split_index_for_jit(idx, shape)
            3892   # Expand any (concrete) boolean indices. We can then use advanced integer
            3893   # indexing logic to handle them.
         -> 3894   idx = _expand_bool_indices(idx, shape)
            3895 
            3896   leaves, treedef = tree_flatten(idx)

         /usr/local/lib/python3.8/dist-packages/jax/_src/numpy/lax_numpy.py in _expand_bool_indices(idx, shape)
            4205       if not type(abstract_i) is ConcreteArray:
            4206         # TODO(mattjj): improve this error by tracking _why_ the indices are not concrete
         -> 4207         raise errors.NonConcreteBooleanIndexError(abstract_i)
            4208       elif _ndim(i) == 0:
            4209         raise TypeError("JAX arrays do not support boolean scalar indices")

         NonConcreteBooleanIndexError: Array boolean indices must be concrete; got ShapedArray(bool[5])

         See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError

.. container:: cell markdown

   문제는 ``x_without_nans``\ 의 크기가 ``x`` 내의 값에 의존하기 때문에
   동적이라는 것입니다. 종종 JAX에서는 다른 방법을 통해 동적으로 크기
   조정된 배열의 필요성을 해결할 수 있습니다. 예를 들어 여기에서
   ``jnp.where``\ 의 3개 인수 형식을 사용하여 NaN 값을 0으로 대체할 수
   있으므로 동적 모양을 피하면서 동일한 결과를 계산할 수 있습니다.

.. container:: cell code

   .. code:: python

      @jax.jit
      def nansum_2(x):
        mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
        return jnp.where(mask, x, 0).sum()

      print(nansum_2(x))

   .. container:: output stream stdout

      ::

         10.0

.. container:: cell markdown

   동적 모양의 배열이 발생하는 다른 상황에서도 유사한 트릭을 사용할 수
   있습니다.

.. container:: cell markdown

   .. rubric:: **🔪 NaNs**
      :name: -nans

   --------------

   .. rubric:: NaNs 디버깅
      :name: nans-디버깅

   함수 또는 그래디언트에서 NaN이 발생하는 위치를 추적하려면 다음과 같이
   NaN 검사기를 켤 수 있습니다.

   -  ``JAX_DEBUG_NANS=True`` 환경 변수 설정
   -  메인 파일 상단에
       ``from jax.config import config`` 와 ``config.update("jax_debug_nans",True)`` 를
      추가하세요.
   -  메인
      파일에 ``from jax.config import config`` 와 ``config.parse_flags_with_absl()`` 를
      추가하세요. 그런 다음 명령 줄 플래그에 ``-jax_debug_nans=True`` 을
      이용하여 옵션을 설정하세요.

   이로 인해 NaN 생성 즉시 계산 오류가 발생합니다. 이 옵션을 켜면
   XLA에서 생성된 모든 부동 소수점 유형 값에 nan 검사가 추가됩니다. 즉,
   ``@jit`` 에서 제외되는 모든 기본 작업에 대해 값이 호스트로 다시
   풀백(pulled back)되고 ndarray로 확인됩니다. ``@jit`` 아래에 있는
   코드의 경우 모든 ``@jit`` 함수의 출력을 확인하고 nan이 있으면 한
   수준의 ``@jit``\ 를 제거하며 최적화되지 않은 op-by-op 모드에서 함수를
   다시 실행합니다.

   -> (2차 검수) 즉, ``@jit``\ 에 속하지 않는 모든 기본 작업에 대해 값을
   다시 호스트로 가져와 ndarry로 검사합니다. ``@jit`` 하위에 있는 코드의
   경우 모든 ``@jit``\ 함수의 출력을 검사하고 NaN이 있는 경우 최적화되지
   않은 op-by-op 모드에서 함수를 다시 실행하여 한 번에 한 레벨씩
   ``@jit``\ 을 제거합니다.

   ``@jit``\ 에서만 발생하지만 최적화되지 않은 모드에서는 생성되지 않는
   nan과 같은 까다로운 상황이 발생할 수 있습니다. 이 경우 경고 메시지가
   출력되지만 코드는 계속 실행됩니다.

   그래디언트 평가의 역방향 패스에서 nans가 생성되는 경우 스택 추적에서
   몇 프레임 위로 예외가 발생하면 backward_pass 함수에 있게 됩니다. 이
   함수는 기본적으로 기본 작업 시퀀스를 역순으로 수행하는 간단한 jaxpr
   인터프리터입니다. 아래 예에서 ``env JAX_DEBUG_NANS=True ipython``
   명령줄을 사용하여 ipython repl을 시작한 다음, 다음을 실행했습니다.

   -> (2차 검수) 그래디언트 평가의 역방향 패스에서 nans가 생성되는 경우
   스택 추적에서 몇 프레임 위로 예외가 발생하면 backward_pass 함수
   내부로 진입합니다. 이 함수는 기본적으로 기본 작업 시퀀스를 역순으로
   수행하는 간단한 jaxpr 인터프리터입니다. 아래 예에서
   ``env JAX_DEBUG_NANS=True ipython`` 명령줄을 사용하여 ipython repl을
   시작한 다음, 다음을 실행했습니다.

.. container:: cell code

   .. code:: python

      import jax.numpy as jnp

.. container:: cell code

   .. code:: python

      jnp.divide(0., 0.)

   .. container:: output execute_result

      ::

         DeviceArray(nan, dtype=float32, weak_type=True)

.. container:: cell markdown

   생성된 NaN이 잡혔습니다. ``%debug``\ 를 실행하면 사후 디버거를 얻을
   수 있습니다. 이것은 아래 예제와 같이 ``@jit`` 으로 감싸진 함수에서도
   작동합니다.

.. container:: cell code

   .. code:: python

      from jax import jit

.. container:: cell code

   .. code:: python

      @jit
      def f(x, y):
          a = x * y
          b = (x + y) / (x - y)
          c = a + 2
          return a + b * c

.. container:: cell code

   .. code:: python

      x = jnp.array([2., 0.])
      y = jnp.array([3., 0.])

      f(x, y)

   .. container:: output execute_result

      ::

         DeviceArray([-34.,  nan], dtype=float32)

.. container:: cell markdown

   이 코드는 ``@jit`` 함수의 출력에서 nan을 발견하면 최적화되지 않은
   코드를 호출하므로 여전히 명확한 스택을 추적할 수 있습니다. 그리고
   ``%debug``\ 로 사후 디버거를 실행하여 오류를 파악하기 위해 모든 값을
   검사할 수 있습니다.

   ⚠️ 디버깅하지 않는 경우 NaN 검사기를 켜서는 안 됩니다. 많은
   장치-호스트 왕복 및 성능 저하가 발생할 수 있기 때문입니다!

   ⚠️ NaN 검사기는 pmap에서 작동하지 않습니다. pmap 코드에서 nans를
   디버깅하려면 pmap을 vmap으로 교체해야 합니다.

.. container:: cell markdown

   .. rubric:: **🔪 Double (64bit) 정밀도**
      :name: -double-64bit-정밀도

   --------------

   현재 JAX는 피연산자를 double로 승격시키는 Numpy API의 경향을 완화하기
   위해 기본적으로 단정밀도 숫자를 적용합니다. 이것은 많은 기계 학습
   응용 프로그램에서 원하는 동작이지만, 당신을 놀라게 할 수도 있습니다!

   -> (2차 검수) 현재 JAX는 기본적으로 NumPy API가 피연산자를 강제로
   더블형(double)으로 변환하는 경향을 완화하기 위해
   단정밀도(single-precision) 숫자를 강제로 적용하고 있습니다. 이는 많은
   머신러닝 애플리케이션에서 원하는 동작이지만, 이는 예상치 못한 결과를
   초래할 수 있습니다!

.. container:: cell code

   .. code:: python

      x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
      x.dtype

   .. container:: output execute_result

      ::

         dtype('float32')

.. container:: cell markdown

   Double 정밀도의 숫자를 사용하려면, 시작 시 ``jax_enable_x64`` 구성
   변수를 설정해야 합니다.

   이를 수행하기 위한 몇가지 방법이 있습니다.

   #. ``JAX_ENABLE_X64=True``\ 로 설정하여 64비트 모드를 사용 가능하게
      할 수 있습니다.
   #. 시작 시에 ``jax_enable_x64`` 구성 플래그를 수동으로 설정할 수
      있습니다:

.. container:: cell code

   .. code:: python

      # again, this only works on startup!
      from jax.config import config
      config.update("jax_enable_x64", True)

.. container:: cell markdown

   #. ``absl.app.run(main)``\ 을 사용하여 명령줄 플래그를 파싱할 수
      있습니다.

.. container:: cell code

   .. code:: python

      from jax.config import config
      config.config_with_absl()

.. container:: cell markdown

   JAX가 absl 파싱을 대신 수행하려면, 즉,\ ``absl.app.run(main)``\ 을
   수행하지 않으려면 다음을 사용할 수 있습니다:

   ->(2차 검수) ``absl.app.run(main)``\ 를 사용하지 않고 JAX가 absl
   파싱을 수행하게 하려면 다음과 사용하면 됩니다:

.. container:: cell code

   .. code:: python

      from jax.config import config
      if __name__ == '__main__':
        # calls config.config_with_absl() *and* runs absl parsing
        config.parse_flags_with_absl()

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         DuplicateFlagError                        Traceback (most recent call last)
         <ipython-input-59-5c374bc4b618> in <module>
               2 if __name__ == '__main__':
               3   # calls config.config_with_absl() *and* runs absl parsing
         ----> 4   config.parse_flags_with_absl()

         /usr/local/lib/python3.8/dist-packages/jax/_src/config.py in parse_flags_with_absl(self)
             172 
             173       import absl.flags
         --> 174       self.config_with_absl()
             175       absl.flags.FLAGS(jax_argv, known_only=True)
             176       self.complete_absl_config(absl.flags)

         /usr/local/lib/python3.8/dist-packages/jax/_src/config.py in config_with_absl(self)
             155     for name, val in self.values.items():
             156       flag_type, meta_args, meta_kwargs = self.meta[name]
         --> 157       absl_defs[flag_type](name, val, *meta_args, **meta_kwargs)
             158     app.call_after_init(lambda: self.complete_absl_config(absl_flags))
             159 

         /usr/local/lib/python3.8/dist-packages/absl/flags/_defines.py in DEFINE_integer(name, default, help, lower_bound, upper_bound, flag_values, required, **args)
             423   parser = _argument_parser.IntegerParser(lower_bound, upper_bound)
             424   serializer = _argument_parser.ArgumentSerializer()
         --> 425   result = DEFINE(
             426       parser,
             427       name,

         /usr/local/lib/python3.8/dist-packages/absl/flags/_defines.py in DEFINE(parser, name, default, help, flag_values, serializer, module_name, required, **args)
              98     a handle to defined flag.
              99   """
         --> 100   return DEFINE_flag(
             101       _flag.Flag(parser, serializer, name, default, help, **args), flag_values,
             102       module_name, required)

         /usr/local/lib/python3.8/dist-packages/absl/flags/_defines.py in DEFINE_flag(flag, flag_values, module_name, required)
             134   # Copying the reference to flag_values prevents pychecker warnings.
             135   fv = flag_values
         --> 136   fv[flag.name] = flag
             137   # Tell flag_values who's defining the flag.
             138   if module_name:

         /usr/local/lib/python3.8/dist-packages/absl/flags/_flagvalues.py in __setitem__(self, name, flag)
             430         # module is simply being imported a subsequent time.
             431         return
         --> 432       raise _exceptions.DuplicateFlagError.from_flag(name, self)
             433     short_name = flag.short_name
             434     # If a new flag overrides an old one, we need to cleanup the old flag's

         DuplicateFlagError: The flag 'jax_tracer_error_num_traceback_frames' is defined twice. First from jax._src.config, Second from jax._src.config.  Description from first occurrence: Set the number of stack frames in JAX tracer error messages.

.. container:: cell markdown

   #2-#4는 JAX의 모든 구성 옵션에서 작동합니다.

   그런 다음 x64 모드가 활성화되었는지 확인할 수 있습니다.

.. container:: cell code

   .. code:: python

      import jax.numpy as jnp
      from jax import random
      x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
      x.dtype # --> dtype('float64')

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         UnparsedFlagAccessError                   Traceback (most recent call last)
         <ipython-input-60-bea7f3d9c65d> in <module>
               1 import jax.numpy as jnp
               2 from jax import random
         ----> 3 x = random.uniform(random.PRNGKey(0), (1000,), dtype=jnp.float64)
               4 x.dtype # --> dtype('float64')

         /usr/local/lib/python3.8/dist-packages/jax/_src/random.py in PRNGKey(seed)
             126 
             127   """
         --> 128   impl = default_prng_impl()
             129   if np.ndim(seed):
             130     raise TypeError("PRNGKey accepts a scalar seed, but was given an array of"

         /usr/local/lib/python3.8/dist-packages/jax/_src/random.py in default_prng_impl()
             104   ``jax.prng.PRNGImpl`` instance.
             105   """
         --> 106   impl_name = config.jax_default_prng_impl
             107   assert impl_name in PRNG_IMPLS, impl_name
             108   return PRNG_IMPLS[impl_name]

         /usr/local/lib/python3.8/dist-packages/jax/_src/config.py in get_state(self)
             281     def get_state(self):
             282       val = getattr(_thread_local_state, name, unset)
         --> 283       return val if val is not unset else self._read(name)
             284     setattr(Config, name, property(get_state))
             285 

         /usr/local/lib/python3.8/dist-packages/jax/_src/config.py in _read(self, name)
             100   def _read(self, name):
             101     if self.use_absl:
         --> 102       return getattr(self.absl_flags.FLAGS, name)
             103     else:
             104       self.check_exists(name)

         /usr/local/lib/python3.8/dist-packages/absl/flags/_flagvalues.py in __getattr__(self, name)
             479       return fl[name].value
             480     else:
         --> 481       raise _exceptions.UnparsedFlagAccessError(
             482           'Trying to access flag --%s before flags were parsed.' % name)
             483 

         UnparsedFlagAccessError: Trying to access flag --jax_default_prng_impl before flags were parsed.

.. container:: cell markdown

   .. rubric:: 주의사항
      :name: 주의사항

   ⚠️ XLA는 모든 백엔드에서 64비트 컨볼루션을 지원하지 않습니다!

.. container:: cell markdown

   .. rubric:: **🔪 NumPy에서 유래된 여러가지 파생들**
      :name: -numpy에서-유래된-여러가지-파생들

   --------------

   ``jax.numpy``\ 는 Numpy API 동작을 유사하게 하기 위한 시도들을 하지만
   동작이 다른 코너케이스들이 있습니다. 이러한 많은 경우들은 위 섹션에서
   자세히 설명합니다. 여기에는 API가 분기되는 다른 여러 위치들을
   나엻합니다.

   -  바이너리 작업의 경우 JAX의 유형 승격 규칙은 NumPy에서 사용하는
      규칙과 다소 다릅니다. 자세한 내용은 Type Promotion Semantics를
      참조하십시오.
   -  안전하지 않은 유형 캐스팅(즉, 대상 dtype이 입력 값을 나타낼 수
      없는 캐스팅)를 수행할 때 JAX의 동작은 백엔드에 따라 다를 수 있으며
      일반적으로 NumPy의 동작과 다를 수 있습니다. Numpy는 캐스팅 인수를
      통해 이러한 시나리오에서 결과를 제어할 수
      있습니다(np.ndarray.astype 참조). JAX는 XLA:ConvertElementType의
      동작을 직접 상속하는 대신 이러한 구성을 제공하지 않습니다.

   여기에 NumPy와 JAX의 안전하지 않은 캐스팅에 따른 다른 결과에 대한
   예제입니다.

.. container:: cell code

   .. code:: python

      np.arange(254.0, 258.0).astype('uint8')
      array([254, 255,   0,   1], dtype=uint8)

      jnp.arange(254.0, 258.0).astype('uint8')
      DeviceArray([254, 255, 255, 255], dtype=uint8)

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         NameError                                 Traceback (most recent call last)
         <ipython-input-61-08beec3232bb> in <module>
               1 np.arange(254.0, 258.0).astype('uint8')
         ----> 2 array([254, 255,   0,   1], dtype=uint8)
               3 
               4 jnp.arange(254.0, 258.0).astype('uint8')
               5 DeviceArray([254, 255, 255, 255], dtype=uint8)

         NameError: name 'uint8' is not defined

.. container:: cell markdown

   이러한 종류의 불일치는 일반적으로 부동에서 정수 유형으로 또는 그
   반대로 극단적인 값을 캐스팅할 때 발생합니다.

.. container:: cell markdown

   .. rubric:: Fin.
      :name: fin

   --------------

   여기에서 다루지 않은 몇몇 열받는 원인이 있는 경우 이 입문 조언
   페이지를 더 확장하도록 하겠습니다.

   --> 여기에서 다루지 않은 몇몇 당신을 화나게 하는 원인들을
   제보해주시면 해당 튜토리얼 페이지에 반영하겠습니다.

.. container:: cell code

   .. code:: python
