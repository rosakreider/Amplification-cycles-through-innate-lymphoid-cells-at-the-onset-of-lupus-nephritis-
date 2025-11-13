from acute_inflammation_model import LupusSimulator
from lupus_params import tele_params
from sklearn.utils import Bunch
from tqdm import tqdm
import numpy as np
import pandas as pd

class GillespieSimulator(LupusSimulator):

    def __init__(self, params=None, runs=None):
        if params is None:
            self.params = tele_params
        else:
            self.params = params
        super(GillespieSimulator, self).__init__(params=params, runs=runs)
        self.acute_inflammation = None
        self.rng = np.random.default_rng()
        self.use_progressbar = False
        self.runs = runs

    def gillespie(self):
        times = [0.0]
        y = [self.state]
        steps = 0

        def generator():
            while True:  # while finish time has not been reached
                yield

        if self.use_progressbar:
            container = tqdm(generator())
        else:
            container = generator()

        for _ in container:
            if times[-1] >= self.params.t_max:
                times.pop()
                y.pop()
                times.append(self.params.t_max)
                y.append(y[-1])
                break

            steps += 1
            state = y[-1].copy()
            rates = [propensity_function(*state) for propensity_function in self.propensities]

            if all(r == 0 for r in rates):
                break  # stop loop if no transitions available

            probabilities = [r / sum(rates) for r in rates]
            transition_index = self.rng.choice(len(self.stoichiometry), p=probabilities)  # randomly draw one transition
            transition = self.stoichiometry[transition_index]
            dt = self.rng.exponential(1 / sum(rates))  # np.random.exponential uses scale parameter, not rate parameter
            times.append(times[-1] + dt)  # draw next time increment from random exponential distribution
            next_state = [a + b for a, b in zip(state, transition)]  # update state
            y.append(next_state)

        if times[-1] < self.params.t_max:
            # in case the loop has stopped since no transitions were available, add additional point if final time is not reached
            times.append(self.params.t_max)
            y.append(y[-1])

        return Bunch(t=np.array(times), y=np.array(y).T)

    def simulate_multiple_runs(self, progressbar=False):
        print(f"(GillespieSimulator)  Generating data for {self.runs} Gillespie runs.")  # with model parameters")
        print('k_on: ' + str(self.params.k_on), 'k_off: ' + str(self.params.k_off), 'n: ' + str(self.params.n))

        solutions = []
        runs = range(self.runs)
        if progressbar:
            runs = tqdm(runs)
        for _ in runs:
            solution = self.save_gillespie()
            solutions.append(solution)

        new_labels = self.labels + ["status"]
        data = Bunch(params=self.params, solutions=solutions, labels=new_labels)  # Store data (serialize)
        data.n = self.runs
        return data

    def save_gillespie(self):
        solution = self.gillespie()
        d = {'t': solution.t}
        df = pd.DataFrame(data=d)
        for i in range(len(self.state)):
            df[self.labels[i]] = solution.y[i]

        my_list = solution.y[self.inflammation_on_index:]
        y = [sum(x) for x in zip(*my_list)]
        df['status'] = y
        return df