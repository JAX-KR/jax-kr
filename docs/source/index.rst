.. rtd-tutorial-ybeen documentation master file, created by
   sphinx-quickstart on Mon Feb 27 20:45:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

JAX KR 문서 번역 프로젝트
==============================================

JAX는 고성능 수치 컴퓨팅을 위해 `Autograd <https://github.com/hips/autograd>`_ 와 `XLA <https://www.tensorflow.org/xla>`_ 를 결합한 것입니다.

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: 친숙한 API
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX는 연구자와 엔지니어들이 쉽게 적용할 수 있도록 친숙한 NumPy 스타일의 API를 제공합니다.

   .. grid-item-card:: 변환
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX는 컴파일, 배치 처리, 자동 미분, 병렬화를 위한 구성 가능한 함수 변환을 포함합니다.

   .. grid-item-card:: 어디에서든 실행 가능
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      동일한 코드가 CPU, GPU 및 TPU를 포함한 여러 백엔드에서 실행됩니다.


설치
------------
.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install "jax[cpu]"

    .. tab-item:: GPU (CUDA)

       .. code-block:: bash

          pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    .. tab-item:: TPU (Google Cloud)

       .. code-block:: bash

          pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

JAX 프로젝트 README의 `설치 가이드 <https://github.com/google/jax#installation>`_에서 지원되는 가속기와 플랫폼에 대한 자세한 정보 및 기타 설치 옵션을 확인할 수 있습니다.

본 연구(프로젝트)는 모두의연구소 K-디지털 플랫폼의 지원 받아 진행하고 있습니다.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   JAX101/(최종본)🔪JAX_The_Sharp_bits🔪.md


.. toctree::
   :hidden:
   :maxdepth: 1

   JAX101/index

.. toctree::
   :hidden:
   :maxdepth: 1

   Flax_tutorial/index

.. toctree::
   :hidden:
   :maxdepth: 1

   JAX_Keras/index


