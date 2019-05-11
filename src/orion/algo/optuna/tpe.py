# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.optuna.tpe -- TODO
==============================================

.. module:: tpe
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
import copy

import numpy

import optuna
from optuna.samplers.tpe.sampler import TPESampler
from optuna.distributions import UniformDistribution

from orion.algo.base import BaseAlgorithm
from orion.algo.space import pack_point, unpack_point


def iterdims(orion_space):
    for key, dimension in orion_space.items():

        shape = dimension.shape

        assert not shape or len(shape) == 1
        if not shape:
            shape = (1,)

        # Unpack dimension
        if shape[0] > 1:
            for i in range(shape[0]):
                yield key + '_' + str(i), dimension
        else:
            yield key, dimension


def convert_orion_space_to_optuna_dimensions(orion_space):
    """Convert Oríon's definition of problem's domain to a hyperopt compatible."""

    dimensions = {}
    for key, dimension in iterdims(orion_space):
        #  low = dimension._args[0]
        #  high = low + dimension._args[1]
        low, high = dimension.interval()
        dimensions[key] = UniformDistribution(low, high)

    return dimensions


def default_gamma(x):
    # type: (int) -> int

    return min(int(numpy.ceil(0.25 * numpy.sqrt(x))), 25)


def default_weights(x):
    # type: (int) -> np.ndarray

    if x == 0:
        return numpy.asarray([])
    elif x < 25:
        return numpy.ones(x)
    else:
        ramp = numpy.linspace(1.0 / x, 1.0, num=x - 25)
        flat = numpy.ones(25)
        return numpy.concatenate([ramp, flat], axis=0)


class TPEOptimizer(BaseAlgorithm):
    """
    TODO: Class docstring
    """

    requires = 'linear'

    def __init__(
            self, space,
            consider_prior=True,  # type: bool
            prior_weight=1.0,  # type: float
            consider_magic_clip=True,  # type: bool
            consider_endpoints=False,  # type: bool
            n_startup_trials=10,  # type: int
            n_ei_candidates=24,  # type: int
            gamma=default_gamma,  # type: Callable[[int], int]
            weights=default_weights,  # type: Callable[[int], np.ndarray]
            seed=None  # type: Optional[int]
            ):
        """
        TODO: init docstring
        """
        super(TPEOptimizer, self).__init__(
            space,
            consider_prior=True,  # type: bool
            prior_weight=1.0,  # type: float
            consider_magic_clip=True,  # type: bool
            consider_endpoints=False,  # type: bool
            n_startup_trials=10,  # type: int
            n_ei_candidates=24,  # type: int
            gamma=default_gamma,  # type: Callable[[int], int]
            weights=default_weights,  # type: Callable[[int], np.ndarray]
            seed=None  # type: Optional[int]
            )

        self.study = optuna.create_study(sampler=TPESampler())
        self.dimensions = convert_orion_space_to_optuna_dimensions(space)

    def suggest(self, num=1):
        """Suggest a `num`ber of new sets of parameters.

        TODO: document how suggest work for this algo

        """

        # Make a copy so that all sampled trials are lost afterwards. The will be observed with fake
        # results anyway, so they will be re-introduced in the algo before the next call to
        # `suggest`. We only copy the storage however, otherwise the RNG stote increment inside the
        # samplers would lost.
        storage = self.study.storage
        self.study.storage = copy.deepcopy(storage)

        points = []
        for i in range(num):
            trial_id = self.study.storage.create_new_trial_id(self.study.study_id)
            trial = optuna.trial.Trial(self.study, trial_id)

            params = []
            for param_name, _ in iterdims(self.space):
                distribution = self.dimensions[param_name]
                params.append(trial._suggest(param_name, distribution))

            # TODO: Re-pack points

            points.append(pack_point(params, self.space))

        self.study.storage = storage

        return points

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        TODO: document how observe work for this algo

        """

        for point, result in zip(points, results):

            params = unpack_point(point, self.space)

            # Create a trial
            trial_id = self.study.storage.create_new_trial_id(self.study.study_id)
            trial = optuna.trial.Trial(self.study, trial_id)

            # Set the params
            for i, (param_name, _) in enumerate(iterdims(self.space)):
                distribution = self.dimensions[param_name]
                param_value_internal = distribution.to_internal_repr(params[i])
                self.study.storage.set_trial_param(
                    trial_id, param_name, param_value_internal, distribution)

            # Report the objective
            trial.report(result['objective'])
            self.study.storage.set_trial_state(trial_id, optuna.structs.TrialState.COMPLETE)
