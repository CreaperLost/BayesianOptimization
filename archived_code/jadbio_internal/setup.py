#!/usr/bin/env python
from distutils.core import setup

setup(name='jadbio_internal',
      version='1.0',
      description='jad internal python jadbio_internal',
      author='pkatsogr',
      author_email='pkatsogr@gmail.com',
      packages=[
            'jadbio_internal',
            'jadbio_internal/api',
            'jadbio_internal/ml',
            'jadbio_internal/ml/algo',
            'jadbio_internal/ml/algo/dt',
            'jadbio_internal/ml/algo/gb',
            'jadbio_internal/ml/algo/rf',
            'jadbio_internal/ml/fs',
            'jadbio_internal/ml/pp',
            'jadbio_internal/ml/tuning',
      ],
)