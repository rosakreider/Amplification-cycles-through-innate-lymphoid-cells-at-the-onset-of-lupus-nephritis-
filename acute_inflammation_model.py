from lupus_params import tele_params

class LupusSimulator:

    def __init__(self, params=None, runs=None):
        if params is None:
            self.params = tele_params
        else:
            self.params = params

        self.labels=[]
        self.inflammation_off_index = None
        self.inflammation_on_index = None
        self.initialize_labels()

        self.state = None
        self.state = self.initialize_state()

        self.stoichiometry = None
        self.stoichiometry = self.initialize_stoichiometries()

        self.propensities = None
        self.propensities = self.initialize_propensities()

        self.format = '.svg'

    def initialize_labels(self):

        self.labels += ["inflammation_off"]
        self.inflammation_off_index = 0
        self.labels.append("inflammation_on")
        self.inflammation_on_index = len(self.labels) - 1  # end of response time modelling of pDCs
        self.labels += [f"y_{i}" for i in range(self.params.n)]

        return self.labels

    def initialize_state(self):
        self.state = [0 for _ in range(len(self.labels))]
        self.state[self.inflammation_off_index] = 1
        return self.state

    def initialize_stoichiometries(self):

        # initialize stoichiometry matrix
        matrix_dim_x = len(self.labels)
        matrix_dim_y = self.params.n + 2
        self.stoichiometry = [[0] * matrix_dim_x for _ in range(matrix_dim_y)]

        # inflammation_off -> inflammation_on
        next_reaction = 0
        self.stoichiometry[next_reaction][self.inflammation_off_index] = -1
        self.stoichiometry[next_reaction][self.inflammation_on_index] = 1

        # inflammation_on -> x_1, x_1 -> x_2, ... x_n-1 -> x_n
        for i in range(self.params.n):
            next_reaction += 1
            self.stoichiometry[next_reaction][self.inflammation_on_index + i] = -1
            self.stoichiometry[next_reaction][self.inflammation_on_index + i + 1] = 1

        next_reaction += 1
        self.stoichiometry[next_reaction][len(self.labels) -1] = -1
        self.stoichiometry[next_reaction][self.inflammation_off_index] = 1

        return self.stoichiometry

    def initialize_propensities(self):
        self.propensities = []
        self.propensities.append(lambda *state: self.params.k_on * state[self.inflammation_off_index])

        for i in range(self.params.n + 1):
            self.propensities.append(lambda *state, i_arg=i, start_index_arg=self.inflammation_on_index:
                                    self.params.k_off * state[start_index_arg + i_arg])

        return self.propensities