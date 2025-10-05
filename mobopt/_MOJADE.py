# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 18:43:39 2025

@author: gcaje
"""

import numpy as np
from numpy.random import default_rng

import random
import array

from scipy.stats import cauchy

from deap import base
from deap import creator
from deap import tools

FirstCall = True

def uniform(bounds):
    return [random.uniform(b[0], b[1]) for b in bounds]

def MOJADE(NObj, objective, pbounds, seed=None, NGEN=100, MU=100, p=0.1, c=0.1, archive_size_rate=2.0):

    random.seed(seed)
    generator = default_rng(seed=seed)

    mu_CR = 0.5
    mu_F = 0.5
    archive = []
    archive_size = int(MU * archive_size_rate)

    NDIM = len(pbounds)
    LOWER_BOUNDS = pbounds[:, 0]
    UPPER_BOUNDS = pbounds[:, 1]

    global FirstCall
    if FirstCall:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)*NObj)
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
        FirstCall = False
    toolbox = base.Toolbox()
    toolbox.register("attr_float", uniform, pbounds)
    toolbox.register("individual",
                      tools.initIterate,
                      creator.Individual,
                      toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective)
    toolbox.register("select", tools.selNSGA2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    #print(logbook.stream)

    for g in range(1, NGEN):
        SF = []
        SCR = []
        offspring = []
        offspring_params_map = {}

        for i, parent_ind in enumerate(pop):
            cr = generator.normal(mu_CR, 0.1)
            cr = np.clip(cr, 0, 1)

            while True:
              f = cauchy.rvs(mu_F, 0.1)
              if f < 0:
                continue
              elif f > 1:
                f = 1
              break

            top_p = int(p * MU)
            if top_p < 1: top_p = 1

            pop_u_archive = pop + archive

            pbest_idx = random.randint(0, top_p - 1)
            while pbest_idx == i:
                pbest_idx = random.randint(0, top_p - 1)

            r1_idx = random.randint(0, len(pop_u_archive) - 1)
            while r1_idx == i:
                r1_idx = random.randint(0, len(pop_u_archive) - 1)

            r2_idx = random.randint(0, len(pop_u_archive) - 1)
            while r2_idx == i or r2_idx == r1_idx:
                r2_idx = random.randint(0, len(pop_u_archive) - 1)

            x_pbest = pop[pbest_idx]
            x_r1 = pop_u_archive[r1_idx]
            x_r2 = pop_u_archive[r2_idx]

            x_i_arr = np.array(parent_ind)
            x_pbest_arr = np.array(x_pbest)
            x_r1_arr = np.array(x_r1)
            x_r2_arr = np.array(x_r2)

            mutant_v = x_i_arr + f * (x_pbest_arr - x_i_arr) + f * (x_r1_arr - x_r2_arr)

            u = np.array(parent_ind)

            jrand = random.randrange(NDIM)

            for j in range(NDIM):
                if random.random() < cr or j == jrand:
                    u[j] = mutant_v[j]

            u = np.clip(u, LOWER_BOUNDS, UPPER_BOUNDS)

            offspring_ind = creator.Individual(u)
            del offspring_ind.fitness.values
            offspring.append(offspring_ind)
            offspring_params_map[id(offspring_ind)] = {'cr': cr, 'f': f, 'parent': parent_ind}

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, MU)

        parents_to_archive = []
        for ind in pop:
            if id(ind) in offspring_params_map:
                params = offspring_params_map[id(ind)]
                SF.append(params['f'])
                SCR.append(params['cr'])
                parents_to_archive.append(params['parent'])

        archive.extend(parents_to_archive)
        if len(archive) > archive_size:
            archive = random.sample(archive, archive_size)

        if SF:
          mu_F = (1 - c)*mu_F + c*((np.sum(np.array(SF)**2))/np.sum(SF))
        if SCR:
          mu_CR = (1 - c)*mu_CR + c*(np.mean(SCR))

        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        print(logbook.stream)

    front = np.array([ind.fitness.values for ind in pop])
    return pop, logbook, front
