import numpy as np
from scipy.integrate import solve_ivp
from lupus_params import tele_params


class LupusIncIC(object):

    def __init__(self):
        entities = ['prtILC', 'tILC', 'MOMA', 'iPEC', 'vNK', 'MO', 'EC', 'Damage', 'IC', 'IFN', 'C34', 'C43', 'C67', 'C76']
        self.annotation = dict(zip(entities, [*range(0, len(entities))]))
        self.colors = {
            'prtILC': '#095AA7',
            'tILC': '#095AA7',
            'vNK': '#A70909',
            'MOMA': '#6BAED6',
            'MO': '#E98E83',
            'iPEC': '#C6DBEF',
            'EC': '#faceb9',
            'Damage': 'black',
            'IFN': '#fec44f',
            'IC': '#fe8324',
            'C34': '#6BAED6',
            'C43': '#C6DBEF',
            'C67': '#E98E83',
            'C76': '#faceb9'
        }
        self.acute_inflammation = None


    @staticmethod
    def h(x, param, k):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            x_n = np.power(x, param.n)
            k_n = np.power(k, param.n)
            out = x_n / (k_n + x_n)
            # Wenn out ein Skalar ist:
            if np.isscalar(out):
                return 0 if np.isnan(out) else out
            # Wenn out ein Array ist:
            out[np.isnan(out)] = 0
            return out

    def ode(self, t, y, param):

        x1, x2, x3, x4, x5, x6, x7, x8, x9, c34, c43, c67, c76, x9_1, x9_2, x9_3 = y

        # Lupus trigger (poly I:C or random)
        if t < param.t_start_trigger or t > param.t_end_trigger:
            bf1 = 0
        else:
            bf1 = param.f_trigger

        # ILC knockdown
        if t < param.t_start_kd or t > param.t_end_kd:
            r = 1
            r_nk = 1
        else:
            r = 1 - param.r
            r_nk = 1 - param.r_nk

        h_x2 = self.h(x2, param, param.k)
        h_x6 = self.h(x6, param, param.k3)

        h_x8 = self.h(x8, param, param.k) * param.disruption['F1'] * \
        param.disruption['F1/2'] * \
        param.disruption['F1/3'] * \
        param.disruption['F1/4'] * \
        param.disruption['F1/5'] * \
        param.disruption['F1/2/3'] * \
        param.disruption['F1/2/4'] * \
        param.disruption['F1/2/5'] * \
        param.disruption['F1/3/4'] * \
        param.disruption['F1/3/5'] * \
        param.disruption['F1/4/5'] * \
        param.disruption['F1/2/3/4'] * \
        param.disruption['F1/2/3/5'] * \
        param.disruption['F1/2/4/5'] * \
        param.disruption['F1/3/4/5'] * \
        param.disruption['F1/2/3/4/5']

        h_x8_act = self.h(x8, param, param.k) * param.disruption['F2'] * \
        param.disruption['F1/2'] * \
        param.disruption['F2/3'] * \
        param.disruption['F2/4'] * \
        param.disruption['F2/5'] * \
        param.disruption['F1/2/3'] * \
        param.disruption['F1/2/4'] * \
        param.disruption['F1/2/5'] * \
        param.disruption['F2/3/4'] * \
        param.disruption['F2/3/5'] * \
        param.disruption['F2/4/5'] * \
        param.disruption['F1/2/3/4'] * \
        param.disruption['F1/2/3/5'] * \
        param.disruption['F1/2/4/5'] * \
        param.disruption['F2/3/4/5'] * \
        param.disruption['F1/2/3/4/5']

        h_x9_MOMA = self.h(x9, param, param.k) * param.disruption['F3'] * \
        param.disruption['F1/3'] * \
        param.disruption['F2/3'] * \
        param.disruption['F3/4'] * \
        param.disruption['F3/5'] * \
        param.disruption['F1/2/3'] * \
        param.disruption['F1/3/4'] * \
        param.disruption['F1/3/5'] * \
        param.disruption['F2/3/4'] * \
        param.disruption['F2/3/5'] * \
        param.disruption['F3/4/5'] * \
        param.disruption['F1/2/3/4'] * \
        param.disruption['F1/2/3/5'] * \
        param.disruption['F1/3/4/5'] * \
        param.disruption['F2/3/4/5'] * \
        param.disruption['F1/2/3/4/5']

        h_x9_cf = self.h(x9, param, param.k) * param.disruption['F4'] * \
        param.disruption['F1/4'] * \
        param.disruption['F2/4'] * \
        param.disruption['F3/4'] * \
        param.disruption['F4/5'] * \
        param.disruption['F1/2/4'] * \
        param.disruption['F1/3/4'] * \
        param.disruption['F1/4/5'] * \
        param.disruption['F2/3/4'] * \
        param.disruption['F2/4/5'] * \
        param.disruption['F3/4/5'] * \
        param.disruption['F1/2/3/4'] * \
        param.disruption['F1/2/4/5'] * \
        param.disruption['F1/3/4/5'] * \
        param.disruption['F2/3/4/5'] * \
        param.disruption['F1/2/3/4/5']

        h_x9_MO = self.h(x9, param, param.k) * param.disruption['F5'] * \
        param.disruption['F1/5'] * \
        param.disruption['F2/5'] * \
        param.disruption['F3/5'] * \
        param.disruption['F4/5'] * \
        param.disruption['F1/2/5'] * \
        param.disruption['F1/3/5'] * \
        param.disruption['F1/4/5'] * \
        param.disruption['F2/3/5'] * \
        param.disruption['F2/4/5'] * \
        param.disruption['F3/4/5'] * \
        param.disruption['F1/2/3/5'] * \
        param.disruption['F1/2/4/5'] * \
        param.disruption['F1/3/4/5'] * \
        param.disruption['F2/3/4/5'] * \
        param.disruption['F1/2/3/4/5']

        h_c34 = self.h(c34, param, param.k)
        h_c43 = self.h(c43, param, param.k)
        h_c67 = self.h(c67, param, param.k)
        h_c76 = self.h(c76, param, param.k)

        cf = np.min(np.array([param.bfbase + bf1 + param.bf2 * (h_x9_cf + h_x8), 1]))  # IFN
        h_f = self.h(cf, param, param.k2)

        # Cells
        dydt_x1 = param.l * (1 - (x1 + x2) / (param.w * r)) * (x1 + x2) - (param.m + param.p * np.min(np.array([h_f + h_x8_act, 1]))) * x1  # prtILC
        dydt_x2 = param.p * np.min(np.array([h_f + h_x8_act, 1])) * x1 - param.m * x2  # tILC

        dydt_x3 = param.d * np.min(np.array([h_x2 + param.depl * h_x9_MOMA, 1])) + (param.l * h_c43 - param.m) * x3  # MOMA
        dydt_x4 = (param.d + param.l * h_c34 * x4) * (1 - x4 / param.w) - param.m * x4  # iPEC

        dydt_x5 = param.d * h_x6 * r_nk - param.m * x5  # NK
        dydt_x6 = param.d * np.min(np.array([param.depl * h_x9_MO + param.nk * h_f, 1])) + (param.l * h_c76 - param.m) * x6  # MO
        dydt_x7 = (param.d + param.l * h_c67 * x7) * (1 - x7 / param.w) - param.m * x7  # EC

        dydt_x8 = param.d8 * (x3 + x5 + x6) - param.m8 * x8  # Damage
        dydt_x9_1 = param.d9 * x8 - param.d9 * x9_1
        dydt_x9_2 = param.d9 * x9_1 - param.d9 * x9_2
        dydt_x9_3 = param.d9 * x9_2 - param.d9 * x9_3
        try:
            infl_value = self.acute_inflammation_status(t).iloc[0]
        except (AttributeError, IndexError, TypeError):
            infl_value = 0
        dydt_x9 = param.d9 * (tele_params.q_ic * infl_value + x9_3) - param.m9 * x9  # IC

        # Growth factors
        dydt_c34 = param.b * x3 - param.a * x4 * h_c34 - param.g * c34
        dydt_c43 = param.b * x4 - param.a * x3 * h_c43 - param.g * c43

        dydt_c67 = param.b * x6 - param.a * x7 * h_c67 - param.g * c67
        dydt_c76 = param.b * x7 - param.a * x6 * h_c76 - param.g * c76

        return [dydt_x1, dydt_x2, dydt_x3, dydt_x4, dydt_x5, dydt_x6, dydt_x7, dydt_x8, dydt_x9, dydt_c34, dydt_c43,
                dydt_c67, dydt_c76, dydt_x9_1, dydt_x9_2, dydt_x9_3]

    def ode_simulator(self, params, t_eval=None, event_func=None):
        sol = solve_ivp(self.ode, (0, params.t_max), params.y0,  args=(params,), t_eval=t_eval, method='LSODA',
                        events=event_func)
        x8 = sol.y[self.annotation['Damage'], :]
        x9 = sol.y[self.annotation['IC'], :]
        t = sol.t

        h_x9 = self.h(x9, params, params.k)
        h_x8 = self.h(x8, params, params.k)
        bf1 = np.where((t >= params.t_start_trigger) & (t <= params.t_end_trigger), params.f_trigger, 0)

        # calculate IFN-values
        cf_values = params.bfbase + bf1 + params.bf2 * (h_x9 + h_x8)
        cf = np.minimum(cf_values, 1.0)
        return sol, cf

    def acute_inflammation_status(self, t):
        acute_inflammation = self.acute_inflammation[self.acute_inflammation['t'] <= t]
        acute_inflammation = acute_inflammation.iloc[-1:]
        return acute_inflammation['status']


