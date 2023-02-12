from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark

b = XGBoostBenchmark(task_id=167149)
config = b.get_configuration_space(seed=1).sample_configuration()
result_dict = b.objective_function(configuration=config,
                                   fidelity={"n_estimators": 128, "dataset_fraction": 0.5}, rng=1)
print(result_dict)


"""
Multiple Optimizers on SVMSurrogate
=======================================
This example shows how to run SMAC-HB and SMAC-random-search on SVMSurrogate
Please install the necessary dependencies via ``pip install .`` and singularity (v3.5).
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
"""
import logging
from pathlib import Path
from time import time

import numpy as np
from smac.facade.smac_mf_facade import SMAC4MF
from smac.facade.roar_facade import ROAR
from smac.intensification.hyperband import Hyperband
from smac.scenario.scenario import Scenario
from smac.callbacks import IncorporateRunResultCallback

from hpobench.container.benchmarks.surrogates.svm_benchmark import SurrogateSVMBenchmark
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.example_utils import set_env_variables_to_use_only_one_core

logger = logging.getLogger("minicomp")
logging.basicConfig(level=logging.INFO)
set_env_variables_to_use_only_one_core()


class Callback(IncorporateRunResultCallback):
    def __init__(self):
        self.budget = 10

    def __call__(self, smbo, run_info, result, time_left) -> None:
        self.budget -= run_info.budget
        if self.budget < 0:
            # No budget left
            raise ValueError

def create_smac_rs(benchmark, output_dir: Path, seed: int):
    # Set up SMAC-HB
    cs = benchmark.get_configuration_space(seed=seed)

    scenario_dict = {"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 60,
                     "cs": cs,
                     "deterministic": "true",
                     "runcount-limit": 200,
                     "limit_resources": True,  # Uses pynisher to limit memory and runtime
                     "cutoff": 1800,  # runtime limit for target algorithm
                     "memory_limit": 10000,  # adapt this to reasonable value for your hardware
                     "output_dir": output_dir,
                     "abort_on_first_run_crash": True,
                     }

    scenario = Scenario(scenario_dict)
    def optimization_function_wrapper(cfg, seed, **kwargs):
        """ Helper-function: simple wrapper to use the benchmark with smac """
        result_dict = benchmark.objective_function(cfg, rng=seed)
        cs.sample_configuration()
        return result_dict['function_value']

    smac = ROAR(scenario=scenario,
                   rng=np.random.RandomState(seed),
                   tae_runner=optimization_function_wrapper,
                   )
    return smac

def create_smac_hb(benchmark, output_dir: Path, seed: int):
    # Set up SMAC-HB
    cs = benchmark.get_configuration_space(seed=seed)

    scenario_dict = {"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 60,
                     "cs": cs,
                     "deterministic": "true",
                     "runcount-limit": 200,
                     "limit_resources": True,  # Uses pynisher to limit memory and runtime
                     "cutoff": 1800,  # runtime limit for target algorithm
                     "memory_limit": 10000,  # adapt this to reasonable value for your hardware
                     "output_dir": output_dir,
                     "abort_on_first_run_crash": True,
                     }

    scenario = Scenario(scenario_dict)
    def optimization_function_wrapper(cfg, seed, instance, budget):
        """ Helper-function: simple wrapper to use the benchmark with smac """
        result_dict = benchmark.objective_function(cfg, rng=seed,
                                                   fidelity={"dataset_fraction": budget})
        cs.sample_configuration()
        return result_dict['function_value']

    smac = SMAC4MF(scenario=scenario,
                   rng=np.random.RandomState(seed),
                   tae_runner=optimization_function_wrapper,
                   intensifier=Hyperband,
                   intensifier_kwargs={'initial_budget': 0.1, 'max_budget': 1, 'eta': 3}
                   )
    return smac


def run_experiment(out_path: str, on_travis: bool = False):

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    hb_res = []
    rs_res = []
    for i in range(4):
        benchmark = SurrogateSVMBenchmark(rng=i)
        smac = create_smac_hb(benchmark=benchmark, seed=i, output_dir=out_path)
        callback = Callback()
        smac.register_callback(callback)
        try:
            smac.optimize()
        except ValueError:
            print("Done")
        incumbent = smac.solver.incumbent
        inc_res = benchmark.objective_function(configuration=incumbent)
        hb_res.append(inc_res["function_value"])

        benchmark = SurrogateSVMBenchmark(rng=i)
        smac = create_smac_rs(benchmark=benchmark, seed=i, output_dir=out_path)
        callback = Callback()
        smac.register_callback(callback)
        try:
            smac.optimize()
        except ValueError:
            print("Done")
        incumbent = smac.solver.incumbent
        inc_res = benchmark.objective_function(configuration=incumbent)
        rs_res.append(inc_res["function_value"])

    print("SMAC-HB", hb_res, np.median(hb_res))
    print("SMAC-RS", rs_res, np.median(rs_res))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='HPOBench - SVM comp',
                                     description='Run different opts on SVM Surrogate',
                                     usage='%(prog)s --out_path <string>')
    parser.add_argument('--out_path', default='./svm_comp', type=str)
    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')
    args = parser.parse_args()

    run_experiment(out_path=args.out_path, on_travis=args.on_travis)


"""
BOHB on Cartpole
==========================
This example shows the usage of an Hyperparameter Tuner, such as BOHB on the cartpole benchmark.
BOHB is a combination of Bayesian optimization and Hyperband.
**Note**: This is a raw benchmark, i.e. it actually runs an algorithms, and will take some time
Please install the necessary dependencies via ``pip install .[examples]`` and singularity (v3.5).
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
"""
import logging
import pickle
from pathlib import Path

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

from hpobench.container.benchmarks.rl.cartpole import CartpoleReduced as Benchmark
from hpobench.util.example_utils import get_travis_settings, set_env_variables_to_use_only_one_core
from hpobench.util.rng_helper import get_rng

logger = logging.getLogger('BOHB on cartpole')
set_env_variables_to_use_only_one_core()


class CustomWorker(Worker):
    def __init__(self, seed, max_budget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.max_budget = max_budget

    # pylint: disable=arguments-differ
    def compute(self, config, budget, **kwargs):
        b = Benchmark(rng=self.seed)
        # Old API ---- NO LONGER SUPPORTED ---- This will simply ignore the fidelities
        # result_dict = b.objective_function(config, budget=int(budget))
        
        # New API ---- Use this
        result_dict = b.objective_function(config, fidelity={"budget": int(budget)})
        return {'loss': result_dict['function_value'],
                'info': {'cost': result_dict['cost'],
                         'budget': result_dict['budget']}}


def run_experiment(out_path, on_travis):

    settings = {'min_budget': 1,
                'max_budget': 9,  # number of repetitions; this is the fidelity for this bench
                'num_iterations': 10,  # Set this to a low number for demonstration
                'eta': 3,
                'output_dir': Path(out_path)
                }
    if on_travis:
        settings.update(get_travis_settings('bohb'))

    b = Benchmark(rng=1)

    b.get_configuration_space(seed=1)
    settings.get('output_dir').mkdir(exist_ok=True)

    cs = b.get_configuration_space()
    seed = get_rng(rng=0)
    run_id = 'BOHB_on_cartpole'

    result_logger = hpres.json_result_logger(directory=str(settings.get('output_dir')), overwrite=True)

    ns = hpns.NameServer(run_id=run_id, host='localhost', working_directory=str(settings.get('output_dir')))
    ns_host, ns_port = ns.start()

    worker = CustomWorker(seed=seed,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          run_id=run_id,
                          max_budget=settings.get('max_budget'))
    worker.run(background=True)

    master = BOHB(configspace=cs,
                  run_id=run_id,
                  host=ns_host,
                  nameserver=ns_host,
                  nameserver_port=ns_port,
                  eta=settings.get('eta'),
                  min_budget=settings.get('min_budget'),
                  max_budget=settings.get('max_budget'),
                  result_logger=result_logger)

    result = master.run(n_iterations=settings.get('num_iterations'))
    master.shutdown(shutdown_workers=True)
    ns.shutdown()

    with open(settings.get('output_dir') / 'results.pkl', 'wb') as f:
        pickle.dump(result, f)

    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()
    inc_value = result.get_runs_by_id(incumbent)[-1]['loss']
    inc_cfg = id2config[incumbent]['config']

    logger.info(f'Inc Config:\n{inc_cfg}\n'
                f'with Performance: {inc_value:.2f}')

    if not on_travis:
        benchmark = Benchmark(container_source='library://phmueller/automl')
        incumbent_result = benchmark.objective_function_test(configuration=inc_cfg,
                                                             fidelity={"budget": settings['max_budget']})
        print(incumbent_result)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='HPOBench - BOHB',
                                     description='HPOBench with BOHB on Cartpole',
                                     usage='%(prog)s --out_path <string>')
    parser.add_argument('--out_path', default='./cartpole_smac_hb', type=str)
    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')
    args = parser.parse_args()

    run_experiment(out_path=args.out_path, on_travis=args.on_travis)