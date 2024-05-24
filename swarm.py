
import numpy as np
from tqdm import tqdm

from controllers import PIDController
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from pathlib import Path

from swarming.utils.argument_parser import parse_arguments
from swarming.core.pso import PSO, ParallelPSO

class PIDRollout:
    def __init__(self):
        self.tinyphysicsmodel = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
        self.controller = PIDController(p=0, i=0, d=0)
        self.data_path = Path('./data')
        # The parameter search space is already big enough. Restricting to a few segs
        self.num_segments = 10
        self.data_files = sorted(self.data_path.iterdir())[:self.num_segments]
        
    def tuning_cost(self, pid):
        self.controller.reset()
        self.controller.tune(pid[0], pid[1], pid[2])
        costs = []
        for data_file in tqdm(self.data_files, total=len(self.data_files)):
            sim = TinyPhysicsSimulator(self.tinyphysicsmodel, str(data_file), controller=self.controller, debug=False)
            costs.append(sim.rollout()['total_cost'])
        return np.mean(costs)

if __name__ == "__main__":
    args = parse_arguments()

    parallel = args.parallel
    swarm_size = args.swarm_size
    iterations = args.iterations
    executions = args.executions

    dimension = 3
    lower_bounds = [0, 0, -0.05]
    upper_bounds = [1.0, 1.0, 0.05]

    if not parallel:
        PSOClass = PSO
    else:
        # Need to instantiate several TinySims for this
        exit("Parallel PSO not supported yet")
        PSOClass = ParallelPSO

    rollout = PIDRollout()
    pso = PSOClass(swarm_size, dimension, rollout.tuning_cost, lower_bounds, upper_bounds)
    pso.optimize(iterations=iterations, executions=executions)

