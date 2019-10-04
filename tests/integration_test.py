#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.optuna`."""
import os

import numpy
import pytest

from orion.algo.space import Integer, Real, Space
import orion.core.cli
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.worker.primary_algo import PrimaryAlgo


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    from pymongo import MongoClient
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Integer('yolo1', 'uniform', -3, 6)
    space.register(dim1)
    dim2 = Real('yolo2', 'uniform', 0, 1)
    space.register(dim2)

    return space


def test_seeding(space):
    """Verify that seeding makes sampling deterministic"""
    tpe_optimizer = PrimaryAlgo(space, 'tpeoptimizer')

    tpe_optimizer.seed_rng(1)
    a = tpe_optimizer.suggest(1)[0]
    assert not numpy.allclose(a, tpe_optimizer.suggest(1)[0])

    tpe_optimizer.seed_rng(1)
    assert numpy.allclose(a, tpe_optimizer.suggest(1)[0])


def test_set_state(space):
    """Verify that resetting state makes sampling deterministic"""
    tpe_optimizer = PrimaryAlgo(space, 'tpeoptimizer')

    tpe_optimizer.seed_rng(1)
    state = tpe_optimizer.state_dict
    a = tpe_optimizer.suggest(1)[0]
    assert not numpy.allclose(a, tpe_optimizer.suggest(1)[0])

    tpe_optimizer.set_state(state)
    assert numpy.allclose(a, tpe_optimizer.suggest(1)[0])


@pytest.mark.usefixtures("clean_db")
def test_tpe_optimizer(database, monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for single shaped dimension."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["hunt", "--config",
                         "./orion_config_tpe.yaml", "./rosenbrock.py",
                         "-x~uniform(-5, 5)"])


@pytest.mark.usefixtures("clean_db")
def test_tpe_optimizer_two_inputs(database, monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for 2 dimensions."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["hunt", "--config",
                         "./orion_config_tpe.yaml", "./rosenbrock.py",
                         "-x~uniform(-5, 5)", "-y~uniform(-10, 10)"])


@pytest.mark.usefixtures("clean_db")
def test_tpe_optimizer_actually_optimize(database, monkeypatch):
    """Check if TPE Optimizer has better optimization than random search."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    best_random_search = 23.403275057472825
    orion.core.cli.main(["hunt", "--max-trials", "40", "--config",
                         "./orion_config_tpe.yaml", "./black_box.py",
                         "-x~uniform(-50, 50)"])

    with open("./orion_config_tpe.yaml", "r") as f:
        exp = ExperimentBuilder().build_view_from({'config': f})

    objective = exp.stats['best_evaluation']

    assert best_random_search > objective
