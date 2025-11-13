import copy

class ParameterSet(object):
    def __init__(self, name, *initial_data, **kwargs):
        self.name = name
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @property
    def param_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != 'name'
        }

# NZBW + pIC
nzbw_pIC = ParameterSet('nzbw_pIC', dict(
    t_max=54,  # maximal time span [days]
    t_start_trigger=19,  # Lupus trigger
    t_end_trigger=23,
    # ILC knockdown
    t_start_kd=0,
    t_end_kd=0,
    r=0,
    r_nk=0,
    n=3,  # Hill variables

    l=1e-1 * 24 * 7,  # Cell proliferation rates
    m=1e-2 * 24 * 7,  # Cell removal rates
    d=1e-1 * 24 * 7,  # Cell inflitration rate
    p=2e-1 * 24 * 7,  # Cell activation rate
    w=1,  # Carrying capacities cells

    bfbase=0.05,  # basis IFN level
    f_trigger=1,  # IFN by lupus trigger
    bf2=0.4,  # IFN by ICs and damage
    k2=0.4,  # half saturation IFN
    k=1.5,
    k3=8,

    g=1e-2 * 24 * 7,  # Cytokines decay
    a=1e-1 * 60 * 24 * 7,  # endocytosis
    b=1e-2 * 60 * 24 * 7,  # production

    d8=1e-3 * 24 * 7,   # damage growth
    m8=0.5 * 1e-2 * 24 * 7,   # damage decay

    d9=1e-2 * 24 * 7,  # IC growth
    m9=1e-2 * 24 * 7,   # IC decay

    depl=1,
    nk=0.5,

    # Initial parameters
    y0=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    disruption={
        'ref': 1,
        # Single-Knockouts
        'F1': 1,
        'F2': 1,
        'F3': 1,
        'F4': 1,
        'F5': 1,
        # Double-Knockouts
        'F1/2': 1,
        'F1/3': 1,
        'F1/4': 1,
        'F1/5': 1,
        'F2/3': 1,
        'F2/4': 1,
        'F2/5': 1,
        'F3/4': 1,
        'F3/5': 1,
        'F4/5': 1,
        # Three-Knockouts
        'F1/2/3': 1,
        'F1/2/4': 1,
        'F1/2/5': 1,
        'F1/3/4': 1,
        'F1/3/5': 1,
        'F1/4/5': 1,
        'F2/3/4': 1,
        'F2/3/5': 1,
        'F2/4/5': 1,
        'F3/4/5': 1,
        # Four-Knockouts
        'F1/2/3/4': 1,
        'F1/2/3/5': 1,
        'F1/2/4/5': 1,
        'F1/3/4/5': 1,
        'F2/3/4/5': 1,
        # Five-Knockout
        'F1/2/3/4/5': 1
    }
))

## NZBW
nzbw_inc_ic = ParameterSet(
    'nzbw_inc_ic',
    {
        **copy.deepcopy(nzbw_pIC.param_dict),
        't_start_trigger': 0,
        't_end_trigger': 0
    }
)

nzbw_on = ParameterSet(
    'nzbw_on',
    {
        **copy.deepcopy(nzbw_pIC.param_dict),
        't_end_trigger': 54
    }
)

## Wild type + pIC
wt_pIC = ParameterSet(
    'wt_pIC',
    {
        **copy.deepcopy(nzbw_pIC.param_dict),
        'd9': 0
    }
)


## NZBW + pIC + kd
nzbw_pIC_red = ParameterSet(
    'nzbw_pIC_kd',
    {
        **copy.deepcopy(nzbw_pIC.param_dict),
        'depl': 0.2
    }
)

## NZBW + pIC + kd
nzbw_pIC_kd = ParameterSet(
    'nzbw_pIC_kd',
    {
        **copy.deepcopy(nzbw_pIC.param_dict),
        'r': 0.8,
        'r_nk': 0.8,
        't_start_kd': 19,
        't_end_kd': 25,
        'depl': 0.2
    }
)

## NZBW + pIC + kd
nzbw_ILC_kd = ParameterSet(
    'nzbw_pIC_kd',
    {
        **copy.deepcopy(nzbw_pIC.param_dict),
        'r': 0.8,
        't_start_kd': 19,
        't_end_kd': 25,
        'depl': 0.2
    }
)

## NZBW + pIC + kd
nzbw_NK_kd = ParameterSet(
    'nzbw_pIC_kd',
    {
        **copy.deepcopy(nzbw_pIC.param_dict),
        'r_nk': 0.8,
        't_start_kd': 19,
        't_end_kd': 25,
        'depl': 0.2
    }
)

# Acute Inflammation Parameters
tele_params = ParameterSet('tele_params', dict(
    k_on=0.5,
    k_off=6,
    q_ic=0.1,
    n=3,
    t_max=nzbw_pIC.t_max
))

class Params:
    def __init__(self):
        self.nzbw_pIC = nzbw_pIC
        self.nzbw_pIC_red = nzbw_pIC_red
        self.nzbw_inc_ic = nzbw_inc_ic
        self.wt_pIC = wt_pIC
        self.nzbw_pIC_kd = nzbw_pIC_kd
        self.nzbw_ILC_kd = nzbw_ILC_kd
        self.nzbw_NK_kd = nzbw_NK_kd
        self.tele_params = tele_params
        self.nzbw_on = nzbw_on