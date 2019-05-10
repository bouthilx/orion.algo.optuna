# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.optuna -- TODO
====================================

.. module:: optuna
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
from ._version import get_versions

VERSIONS = get_versions()
del get_versions

__descr__ = 'TODO'
__version__ = VERSIONS['version']
__license__ = 'BSD 3-Clause'
__author__ = u'Xavier Bouthillier'
__author_short__ = u'Xavier Bouthillier'
__author_email__ = 'xavier.bouthillier@umontreal.ca'
__copyright__ = u'2019, Xavier Bouthillier'
__url__ = 'https://github.com/bouthilx/orion.algo.optuna'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
