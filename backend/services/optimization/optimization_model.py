from abc import ABC, abstractmethod

import numpy as np
from deap import base, creator, tools
from loguru import logger
from sklearn.preprocessing import OneHotEncoder

from backend.models import BuildingType, ECMParameters
from backend.services.optimization.surrogate_model import ISurrogateModel
from backend.utils.config import ConfigManager


class IOptimizationModel(ABC):
    @abstractmethod
    def optimize(self, building_type: BuildingType) -> tuple[ECMParameters, float]:
        pass


class GeneticAlgorithmModel(IOptimizationModel):
    def __init__(
        self,
        config: ConfigManager,
        surrogate_model: ISurrogateModel,
        encode_model: OneHotEncoder,
        code: str,
    ):
        self._config = config
        self._population_size = config.optimization.genetic.population_size
        self._generations = config.optimization.genetic.generations
        self._crossover_prob_start = config.optimization.genetic.crossover_prob_start
        self._crossover_prob_end = config.optimization.genetic.crossover_prob_end
        self._mutation_prob_start = config.optimization.genetic.mutation_prob_start
        self._mutation_prob_end = config.optimization.genetic.mutation_prob_end
        self._gene_crossover_prob = config.optimization.genetic.gene_crossover_prob
        self._gene_mutation_prob = config.optimization.genetic.gene_mutation_prob
        self._seed = config.optimization.seed
        self._ecm_parameters_names = config.ecm_parameters.keys
        self._ecm_parameters = config.ecm_parameters.model_dump()
        self._max_indices = [
            len(self._ecm_parameters[name]) - 1 for name in self._ecm_parameters_names
        ]
        np.random.seed(self._seed)

        self._surrogate_model = surrogate_model
        self._encode_model = encode_model
        self._code = code


    def _decode_chromosome(
        self, individual: list, building_type: BuildingType
    ) -> ECMParameters:
        params = {"building_type": building_type}
        for i, name in enumerate(self._ecm_parameters_names):
            idx = individual[i]
            value = self._ecm_parameters[name][idx]
            params[name] = value
        return ECMParameters(**params)  # type: ignore[arg-type]

    def _encode_to_features(self, ecm_parameters: ECMParameters) -> np.ndarray:
        code_encoded = self._encode_model.transform([[self._code]]) # type: ignore
        features = [ecm_parameters.model_dump().get(name, 0.0) for name in self._ecm_parameters_names]

        features = np.concatenate([[features], code_encoded], axis=1) # type: ignore[arg-type]
        return features


    def _create_individual(self, icls: type[list]) -> list:
        individual = icls(
            np.random.randint(0, max_idx + 1) for max_idx in self._max_indices
        )
        return individual

    def _evaluate_fitness(self, individual: list) -> tuple[float,]:
        try:
            ecm_parameters = self._decode_chromosome(
                individual,
                self._building_type,
            )

            features = self._encode_to_features(ecm_parameters)

            predictions = self._surrogate_model.predict(features)

            fitness_value = float(predictions[0, 0])

            return (fitness_value,)
        except Exception as e:
            logger.error(f"Error evaluating fitness: {e}")
            return (float("inf"),)

    def _discrete_mutation(self, individual: list, indpb: float) -> tuple[list]:
        for i in range(len(individual)):
            if np.random.random() < indpb:
                individual[i] = np.random.randint(0, self._max_indices[i] + 1)
        return (individual,)

    def _get_adaptive_params(self, gen: int, max_gen: int) -> tuple[float, float]:
        progress = gen / max_gen

        crossover_prob = self._crossover_prob_start - (
            self._crossover_prob_start - self._crossover_prob_end
        ) * progress

        mutation_prob = self._mutation_prob_start - (
            self._mutation_prob_start - self._mutation_prob_end
        ) * progress

        return crossover_prob, mutation_prob

    def optimize(self, building_type: BuildingType) -> tuple[ECMParameters, float]:
        self._building_type = building_type

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)  # type: ignore[attr-defined]

        toolbox = base.Toolbox()
        toolbox.register("attr_int", np.random.randint, 0, 1)
        toolbox.register("individual", self._create_individual, creator.Individual)  # type: ignore[attr-defined]
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore[attr-defined]

        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=self._gene_crossover_prob)
        toolbox.register("mutate", self._discrete_mutation)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=self._population_size)  # type: ignore[attr-defined]

        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", *stats.fields] # type: ignore[arg-type]

        logger.info(f"Starting genetic algorithm optimization for {building_type}")

        fitness = list(map(toolbox.evaluate, population))  # type: ignore[attr-defined]
        for ind, fit in zip(population, fitness, strict=False):
            ind.fitness.values = fit

        hof.update(population)

        for gen in range(self._generations):
            adaptive_cx_prob, adaptive_mut_prob = self._get_adaptive_params(
                gen, self._generations
            )

            offspring = toolbox.select(population, len(population))  # type: ignore[attr-defined]
            offspring = list(map(toolbox.clone, offspring))  # type: ignore[attr-defined]

            for child1, child2 in zip(offspring[::2], offspring[1::2], strict=False):
                if np.random.random() < adaptive_cx_prob:
                    toolbox.mate(child1, child2)  # type: ignore[attr-defined]
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.random() < adaptive_mut_prob:
                    toolbox.mutate(mutant, indpb=self._gene_mutation_prob)  # type: ignore[attr-defined]
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)  # type: ignore[attr-defined]
            for ind, fit in zip(invalid_ind, fitnesses, strict=False):
                ind.fitness.values = fit

            hof.update(offspring)
            if hof:
                worst_ind_idx = np.argmax([ind.fitness.values[0] for ind in offspring])
                offspring[worst_ind_idx] = toolbox.clone(hof[0]) # type: ignore[attr-defined]

            population[:] = offspring

            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            logger.info(f"Generation {gen + 1}: {record}")

            if gen % 10 == 0:
                logger.info(
                    f"Generation {gen}: min={record['min']:.2f}, avg={record['avg']:.2f}"
                )

        best_ind = tools.selBest(population, k=1)[0]
        best_ecm = self._decode_chromosome(best_ind, building_type)

        logger.info(
            f"Optimization completed. Best fitness: {best_ind.fitness.values[0]:.2f}"
        )
        logger.info(f"Best ECM parameters: {best_ecm}")

        return best_ecm, best_ind.fitness.values[0]
