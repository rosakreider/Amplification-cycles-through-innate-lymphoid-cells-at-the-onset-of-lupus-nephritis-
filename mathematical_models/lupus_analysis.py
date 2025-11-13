from gillespie_simulator import GillespieSimulator
from lupus_models import LupusIncIC
from lupus_params import Params

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import copy
import pandas as pd
import numpy as np
import csv
import os
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
import time
from tqdm import tqdm


# Fonts eingebettet als Text, nicht als Pfade
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42

sns.set_style("ticks")
plt.rcParams.update({
    'axes.linewidth': 2.5,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'legend.fontsize': 16,
})

class Analysis:
    def __init__(self, model: LupusIncIC, params: Params):
        """
        Analysis object for LupusIncIC model and ParameterSet.

        Parameters:
        - model: an instance of LupusIncIC (e.g., lupus_models)
        - params: an instance of ParameterSet (e.g., lupus_params)
        """
        self.model = model
        self.params = params
        self.annotation = model.annotation  # shorthand for access
        self.colors = model.colors

    #############
    # Analysis functions
    def format_ax(self, ax, xlim=(17, 40), color='gray'):
        """
        Apply consistent styling to axis.
        """
        #ax.set_xlim(*xlim)
        #ax.set_ylim(bottom=-0.1)
        ax.tick_params(axis='both', direction='out', length=6, width=2, color=color, labelcolor='black')
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    def norm_fct(self, paramset=None):
        """
        End-stage norm (ESN) function.
        Calculates the time until 95% of final damage value is reached and
        normalizes all variables to their values at that time point.

        Parameters
        ----------
        paramset : optional
            Parameter set to simulate. If None, defaults to self.params.nzbw_pIC.

        Returns
        -------
        dmg_95 : float
            Damage value at 95% threshold.
        dmg_t95 : float
            Time when 95% damage is reached.
        norm : list[float]
            Values of all state variables at dmg_t95 (normalization factors).
        """
        # choose default paramset if none provided
        if paramset is None:
            paramset = self.params.nzbw_pIC

        ref_sim, cf = self.model.ode_simulator(paramset)

        dmg_id = self.annotation.get('Damage')
        dmg_curve = ref_sim.y[dmg_id]
        max_dmg = max(dmg_curve)

        # threshold at 95% of maximum damage
        dmg_95 = dmg_curve[(dmg_curve / max_dmg) >= 0.95][0]
        dmg_t95 = ref_sim.t[dmg_curve >= dmg_95][0]

        # values of all state variables at dmg_t95
        idx_t95 = (ref_sim.t == dmg_t95)
        norm = [float(ref_sim.y[i][idx_t95]) for i in range(len(ref_sim.y))]

        return dmg_95, dmg_t95, norm

    def norm_fct_red(self, paramset=None):
        """
        End-stage norm (ESN) function.
        Calculates the time until 95% of final damage value is reached and
        normalizes all variables to their values at that time point.

        Parameters
        ----------
        paramset : optional
            Parameter set to simulate. If None, defaults to self.params.nzbw_pIC_red.

        Returns
        -------
        dmg_95 : float
            Damage value at 95% threshold.
        dmg_t95 : float
            Time when 95% damage is reached.
        norm : list[float]
            Values of all state variables at dmg_t95 (normalization factors).
        """
        # choose default paramset if none provided
        if paramset is None:
            paramset = self.params.nzbw_pIC_red

        ref_sim, cf = self.model.ode_simulator(paramset)

        dmg_id = self.annotation.get('Damage')
        dmg_curve = ref_sim.y[dmg_id]
        max_dmg = max(dmg_curve)

        # threshold at 95% of maximum damage
        dmg_95 = dmg_curve[(dmg_curve / max_dmg) >= 0.95][0]
        dmg_t95 = ref_sim.t[dmg_curve >= dmg_95][0]

        # values of all state variables at dmg_t95
        idx_t95 = (ref_sim.t == dmg_t95)
        norm = [float(ref_sim.y[i][idx_t95]) for i in range(len(ref_sim.y))]

        return dmg_95, dmg_t95, norm

    def time2esn(self, sim, dmg_95, nan=True):
        """
        Computes interpolated time (in weeks) at which Damage crosses dmg_95.

        Parameters:
        - sim: simulation result with attributes t (time) and y (solution array)
        - dmg_95: damage threshold
        - nan: whether to return NaN if threshold not reached (else return 100)

        Returns:
        - Time in weeks when damage reaches dmg_95 (interpolated), or fallback value.
        """
        dmg_id = self.annotation['Damage']
        y_dmg = sim.y[dmg_id]

        if np.any(y_dmg >= dmg_95):
            above_idx = np.argmax(y_dmg >= dmg_95)
            below_idx = np.where(y_dmg <= dmg_95)[0]
            below_idx = below_idx[below_idx < above_idx]

            if len(below_idx) == 0:
                return sim.t[above_idx]   # first time threshold reached

            last_below = below_idx[-1]

            x_vals = [y_dmg[last_below], y_dmg[above_idx]]
            t_vals = [sim.t[last_below] , sim.t[above_idx]]

            return np.interp(dmg_95, x_vals, t_vals)
        else:
            return float('nan') if nan else 100

    def get_t_esn_distribution(self, params=None, runs=None):

        dmg_95, dmg_t95, norm = self.norm_fct()
        ts = []
        simulator = GillespieSimulator(params=params, runs=runs)
        data = simulator.simulate_multiple_runs()
        for sol in data.solutions:
            self.model.acute_inflammation = sol
            sim = self.model.ode_simulator(self.params.nzbw_inc_ic)[0]
            dmg_t95 = self.time2esn(sim, dmg_95)
            ts.append(dmg_t95)
        return ts


    def sample_lognormal(self, val, rel_std=0.1):
        sigma = np.sqrt(np.log(1 + rel_std ** 2))
        mu = np.log(val) - 0.5 * sigma ** 2
        return np.random.lognormal(mu, sigma)

    def network_disruption_screen(self):
        """
        Perform a network disruption screen by disabling specific feedbacks defined in the parameter set.

        Returns:
            results (dict): Dictionary of simulation results for each disrupted interaction.
        """
        results = {}
        for key in self.params.nzbw_pIC.disruption:
            print(key)
            tmp_param = copy.deepcopy(self.params.nzbw_pIC)
            tmp_param.disruption[key] = 0  # disable the interaction
            sol = self.model.ode_simulator(tmp_param)[0]
            results[key] = sol
        return results

    def sens_analysis(self, pset=None):
        """
        Perform sensitivity analysis for a predefined set of parameters
        (decay and migration rates) by varying each parameter ±2-fold
        and observing the change in time to ESRD.

        Parameters:
        - pset: ParameterSet instance used for simulation

        Returns:
        - A pandas DataFrame with sensitivity results (parameter, factor, delta in weeks)
        """
        dmg_95, dmg_t95, norm = self.norm_fct()

        if pset==None:
            pset = self.params.nzbw_pIC

        # Define parameters to perturb (migration and decay rates)
        pvar = [k for k in vars(pset) if not k.startswith('t_') and k not in ['r', 'r_nk', 'name', 'n', 'y0', 'k', 'k2', 'k3', 'w', 'f_trigger', 'disruption', 'm8', 'd8', 'bfbase', 'g', 'a', 'b', 'depl', 'nk']]
        names = ['cell prolif.', 'cell death', 'cell infilt.', 'cell activ.', 'IFN-I prod.', 'IC prod.', 'IC decay']

        # Initialize result list
        results = [['name', 'var', 'fac', 'delta_weeks']]

        # Run baseline simulation
        sim = self.model.ode_simulator(pset)[0]
        delta_weeks = self.time2esn(sim, dmg_95)
        results.append(['ref', 'ref', 1.0, delta_weeks])

        # Perform sensitivity runs (factor 0.5 and 2.0)
        for i, pname in enumerate(pvar):
            for fac in [0.8, 1.2]:
                tmp_pset = copy.deepcopy(pset)
                setattr(tmp_pset, pname, getattr(tmp_pset, pname) * fac)
                sim = self.model.ode_simulator(tmp_pset)[0]
                delta_weeks = self.time2esn(sim, dmg_95, nan=False)
                results.append([names[i], pname, fac, delta_weeks])

        return pd.DataFrame(results[1:], columns=results[0])

    def sens_analysis_2(self, pset=None):
        """
        Perform sensitivity analysis for a predefined set of parameters
        (decay and migration rates) by varying each parameter ±2-fold
        and observing the change in time to ESRD.

        Parameters:
        - pset: ParameterSet instance used for simulation

        Returns:
        - A pandas DataFrame with sensitivity results (parameter, factor, delta in weeks)
        """
        dmg_95, dmg_t95, norm = self.norm_fct()

        if pset==None:
            pset = self.params.nzbw_pIC

        # Define parameters to perturb (migration and decay rates)
        pvar = [k for k in ['k', 'k2', 'k3', 'w', 'bfbase', 'f_trigger', 'm8', 'd8',  'b', 'a', 'g']]
        names = ['K1', 'K2', 'K3', 'carrying cap', 'IFN base', 'IFN pIC', 'damage decay', 'damage grwoth',  'cyt prod', 'cyt endo', 'cyt decay']

        # Initialize result list
        results = [['name', 'var', 'fac', 'delta_weeks']]

        # Run baseline simulation
        sim = self.model.ode_simulator(pset)[0]
        delta_weeks = self.time2esn(sim, dmg_95)
        results.append(['ref', 'ref', 1.0, delta_weeks])

        # Perform sensitivity runs (factor 0.5 and 2.0)
        for i, pname in enumerate(pvar):
            for fac in [0.5, 2.0]:
                tmp_pset = copy.deepcopy(pset)
                setattr(tmp_pset, pname, getattr(tmp_pset, pname) * fac)
                sim = self.model.ode_simulator(tmp_pset)[0]
                delta_weeks = self.time2esn(sim, dmg_95, nan=False)
                results.append([names[i], pname, fac, delta_weeks])

        return pd.DataFrame(results[1:], columns=results[0])

    def knockdown_analysis(self):
        """
        Analyze depletion strength effect on ESRD time for three knockdown types:
        tILC, NK, and NKp46+.
        """
        dmg_95, dmg_t95, norm = self.norm_fct_red()

        # Liste mit (Label, ParamSet, zu variierende(r) Parametername(n))
        kd_sets = [
            ('NZB/W', self.params.nzbw_pIC_red, []),
            ('tILC depl', self.params.nzbw_ILC_kd, ['r']),
            ('vNK depl', self.params.nzbw_NK_kd, ['r_nk']),
            ('tILC/vNK depl', self.params.nzbw_pIC_kd, ['r', 'r_nk']),
        ]

        all_results = []

        for label, base_pset, vars_to_vary in kd_sets:
            for j in np.linspace(0, 1, num=50):
                tmp_pset = copy.deepcopy(base_pset)
                for var in vars_to_vary:
                    setattr(tmp_pset, var, j)
                sim = self.model.ode_simulator(tmp_pset)[0]
                delta_weeks = self.time2esn(sim, dmg_95)
                all_results.append({
                    'fac': j,
                    'delta_weeks': delta_weeks,
                    'condition': label
                })

        return pd.DataFrame(all_results)

    def screen_nkp46_timing_multi(self, param_list, labels, out_basepath='csv/'):
        """
        Run NKp46 knockdown timing screens for multiple parameter settings.

        Args:
            param_list: list of ParameterSet objects (each with its own kd definition)
            labels: list of labels (used for filenames)
            out_basepath: directory to save the CSVs
        Returns:
            List of DataFrames
        """
        dmg_95, dmg_t95, norm = self.norm_fct_red()
        sim_time = 45
        results = []

        for param, label in zip(param_list, labels):
            tmp = np.empty((0, 3), float)
            tmp1 = copy.deepcopy(param)

            for i in range(0, min(2 * sim_time, 31)):  # start max 30
                for j in range(i, min(i + 31, 2 * sim_time)):  # stop - start max 30
                    tmp1.t_start_kd = i
                    tmp1.t_end_kd = j
                    sim = self.model.ode_simulator(tmp1)[0]
                    delta_weeks = self.time2esn(sim, dmg_95)
                    tmp = np.append(tmp, np.array([[i, j, delta_weeks]]), axis=0)

            df = pd.DataFrame(tmp, columns=['start', 'stop', 't50'])
            df['t_treatment'] = df['stop'] - df['start']
            df.to_csv(f'{out_basepath}screen_nkp46_timing_{label}.tsv', sep='\t', index=False)
            results.append(df)

        return results

    def load_nkp46_timing_from_csv(self, labels, path='csv/'):
        """
        Load precomputed depletion timing data from TSV files.

        Args:
            labels: list of label strings (used in filenames: screen_nkp46_timing_{label}.tsv)
            path: path to the directory containing the CSV files

        Returns:
            List of DataFrames
        """
        import os
        dfs = []
        for label in labels:
            file_path = os.path.join(path, f'screen_nkp46_timing_{label}.tsv')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            df = pd.read_csv(file_path, sep='\t')
            dfs.append(df)
        return dfs

    def _plot_inflammation_damage(self, acute_infl, t, damage, filename):

        fig, ax = plt.subplots(
            figsize=(3, 2.5)
        )

        self.model.acute_inflammation = None
        dmg_95, dmg_t95, norm = self.norm_fct()

        # --- Hybrid model damage ---
        dmg_hybrid = damage
        t_hybrid = t

        mask_hybrid = dmg_hybrid <= 1.0
        if not np.all(mask_hybrid):
            t_hybrid = t_hybrid[mask_hybrid]
            dmg_hybrid = dmg_hybrid[mask_hybrid]
            ax.plot(t_hybrid, dmg_hybrid, linewidth=3, color='black', label='hybrid model')
            # Endpunkt markieren
            ax.plot(t_hybrid[-1], dmg_hybrid[-1], 'o', color='black', markersize=5)
        else:
            ax.plot(t_hybrid, dmg_hybrid, linewidth=3, color='black', label='hybrid model')

        # --- Deterministic model (baseline) ---
        sim_det, _ = self.model.ode_simulator(self.params.nzbw_inc_ic)
        dmg_det = sim_det.y[self.annotation['Damage']] / norm[self.annotation['Damage']]
        t_det = sim_det.t

        mask_det = dmg_det <= 1.0
        if not np.all(mask_det):
            t_det = t_det[mask_det]
            dmg_det = dmg_det[mask_det]
            ax.plot(t_det, dmg_det, color='grey', linewidth=3, label='deterministic model')
            ax.plot(t_det[-1], dmg_det[-1], 'o', color='grey', markersize=5)
        else:
            ax.plot(t_det, dmg_det, color='grey', linewidth=3, label='deterministic model')

        # --- Deterministic model (pIC, nzbw_on) ---
        sim_det_pic, _ = self.model.ode_simulator(self.params.nzbw_on)
        dmg_det_pic = sim_det_pic.y[self.annotation['Damage']] / norm[self.annotation['Damage']]
        t_det_pic = sim_det_pic.t

        mask_det_pic = dmg_det_pic <= 1.0
        if not np.all(mask_det_pic):
            t_det_pic = t_det_pic[mask_det_pic]
            dmg_det_pic = dmg_det_pic[mask_det_pic]
            ax.plot(t_det_pic, dmg_det_pic, color='steelblue', linewidth=3, label='deterministic model (pIC)')
            ax.plot(t_det_pic[-1], dmg_det_pic[-1], 'o', color='steelblue', markersize=5)
        else:
            ax.plot(t_det_pic, dmg_det_pic, color='steelblue', linewidth=3, label='deterministic model (pIC)')

        # --- Styling ---
        ax.set_ylabel('damage (norm)', color='black')
        ax.tick_params(axis='y', colors='black')

        ax.set_ylim(0, 1.5)
        ax.set_xlim(18, 36)

        ax.set_xlabel('time (weeks)')
        ax.tick_params(axis='both')

        self.format_ax(ax)
        ax.legend(loc='upper left', frameon=False, fontsize=10)

        plt.tight_layout()
        plt.savefig(filename + '_damage.pdf', bbox_inches='tight')
        plt.close()

    def _plot_inflammation_zoom(self, acute_infl, t, damage, ic, filename):

        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, ncols=1,
            figsize=(3.5, 2.7),  # slightly taller
            height_ratios=[1, 1.5, 1],
            sharex=True
        )

        fs = 14

        # --- Upper subplot: Inflammation signal ---
        t_infl = np.array(acute_infl['t'])
        y_infl = np.array(acute_infl['status'])
        ax1.fill_between(t_infl, y_infl, step="post", color='black', alpha=0.7, linewidth=0)
        ax1.set_ylabel("")
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['off', 'on'])
        ax1.set_xlim(0, 20)

        self.format_ax(ax1)

        # --- Middle subplot: IC ---
        ax2.plot(t, ic, linewidth=3, color=self.colors['IC'])
        ax2.set_ylabel('IC\n(norm)', color='black', fontsize=fs)
        ax2.tick_params(axis='y', colors='black')
        ax2.set_ylim(0, 0.2)

        self.format_ax(ax2)

        # --- Lower subplot: Damage ---
        ax3.plot(t, damage, linewidth=3, color=self.colors['Damage'])
        ax3.set_ylabel('damage\n(norm)', color='black', fontsize=fs)
        ax3.tick_params(axis='y', colors='black')
        # optionally adjust limits:
        ax3.set_ylim(0, 0.2)

        ax3.set_xlabel('time (weeks)', fontsize=fs)
        ax3.tick_params(axis='both')

        self.format_ax(ax3)

        plt.tight_layout(h_pad=0.25)
        plt.savefig(filename + '.pdf', bbox_inches='tight')
        plt.close()

    def run_heterogeneity_analysis_mu(self, mu_values, fixed_std, runs_per_file, n_repeats,
                                      folder_base_path, summary_csv_path):

        def make_params(n, k_off):
            new_paramset = copy.deepcopy(self.params.tele_params)
            setattr(new_paramset, "n", int(n))
            setattr(new_paramset, "k_off", k_off)
            return new_paramset

        summary_records = []

        for mu in mu_values:
            k_float = (mu / fixed_std) ** 2
            n = int(np.ceil(k_float))
            k_off = n / mu

            print(f"[mu={mu:.2f}] -> n = {n}, k_off = {k_off:.4f}")

            paramset = make_params(n, k_off)
            per_run_percentiles = []

            folder_path = os.path.join(folder_base_path, f"mu_{mu:.2f}".replace(".", "_"))
            os.makedirs(folder_path, exist_ok=True)

            for i in range(n_repeats):
                t_values = np.array(self.get_t_esn_distribution(params=paramset, runs=runs_per_file))
                t_values = t_values[~np.isnan(t_values)]

                if len(t_values) == 0:
                    print(
                        f"⚠️  Warnung: Leeres t_ESRD-Array für mu = {mu:.2f}, Wiederholung {i + 1}, überspringe Perzentile.")
                    heterogeneity = np.nan
                else:
                    percentiles = np.percentile(t_values, [5, 95])
                    heterogeneity = percentiles[1] - percentiles[0]
                    per_run_percentiles.append(heterogeneity)

                df = pd.DataFrame({
                    'mu': mu,
                    't_ESRD': t_values,
                    'reached_esrd': t_values <= 54 if len(t_values) > 0 else []
                })
                df.to_csv(os.path.join(folder_path, f"replicate_{i + 1:03d}.csv"), index=False)

            if len(per_run_percentiles) == 0:
                print(f"⚠️  Keine gültigen Perzentil-Werte für mu = {mu:.2f}.")
                heterogeneity_mean = np.nan
                heterogeneity_sem = np.nan
            else:
                per_run_percentiles = np.array(per_run_percentiles)
                heterogeneity_mean = np.mean(per_run_percentiles)
                heterogeneity_sem = np.std(per_run_percentiles, ddof=1) / np.sqrt(len(per_run_percentiles))

            summary_records.append({
                'mu': mu,
                'std': fixed_std,
                'n': n,
                'k_off': k_off,
                'heterogeneity_95_5': heterogeneity_mean,
                'heterogeneity_SEM': heterogeneity_sem
            })

        pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
        print(f"Zusammenfassung gespeichert in {summary_csv_path}")

    def summarize_mu_scan(self, mu_values, fixed_std, folder_base_path, summary_csv_path,
                          bins=50, ignore_outliers=True):
        """
        For each mu in mu_values:
          - collect all t_ESRD values from replicate CSVs
          - shift them so that deterministic nzbw_on ESRD = 0
          - compute histogram, mean, std, median, IQR
          - save results into one summary CSV
        """

        import os
        import numpy as np
        import pandas as pd

        summary_records = []

        dmg_95, dmg_t95, norm = self.norm_fct()

        self.acute_inflammation = None
        sim_det, _ = self.model.ode_simulator(self.params.nzbw_on)

        t_det_esrd=self.time2esn(sim_det, dmg_95)

        print(f"Referenz ESRD (NZBW_on) = {t_det_esrd:.2f} weeks")

        for mu in mu_values:
            folder_name = f"mu_{mu:.2f}".replace(".", "_")
            folder_path = os.path.join(folder_base_path, folder_name)
            k_float = (mu / fixed_std) ** 2
            n = int(np.ceil(k_float))
            k_off = n / mu

            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found for mu={mu:.2f}: {folder_path}")
                continue

            all_t = []
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and fname.startswith("replicate_"):
                    df = pd.read_csv(os.path.join(folder_path, fname))
                    if "t_ESRD" in df.columns:
                        all_t.extend(df["t_ESRD"].dropna().values)

            all_t = np.array(all_t, dtype=float)
            if len(all_t) == 0:
                print(f"⚠️ No t_ESRD values for mu={mu:.2f}")
                summary_records.append({
                    "mu": mu,
                    "k_off": k_off,
                    "mean_shifted_t_ESRD": np.nan,
                    "std_shifted_t_ESRD": np.nan,
                    "median_shifted_t_ESRD": np.nan,
                    "iqr_shifted_t_ESRD": np.nan,
                    "n_values": 0
                })
                continue

            # optional: remove extreme outliers
            if ignore_outliers:
                low, high = np.percentile(all_t, [1, 99])
                all_t = all_t[(all_t >= low) & (all_t <= high)]

            # --- 3. Shift mit deterministischem ESRD ---
            shifted_t = all_t - t_det_esrd

            mean_val = float(np.mean(shifted_t))
            std_val = float(np.std(shifted_t, ddof=1))
            median_val = float(np.median(shifted_t))
            q1, q3 = np.percentile(shifted_t, [25, 75])
            iqr_val = float(q3 - q1)

            summary_records.append({
                "mu": mu,
                "k_off": k_off,
                "mean_shifted_t_ESRD": mean_val,
                "std_shifted_t_ESRD": std_val,
                "median_shifted_t_ESRD": median_val,
                "iqr_shifted_t_ESRD": iqr_val,
                "n_values": len(shifted_t)
            })

            print(f"[mu={mu:.2f}] n={len(shifted_t)}, mean={mean_val:.2f}, "
                  f"std={std_val:.2f}, median={median_val:.2f}, IQR={iqr_val:.2f}")

        pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
        print(f"Summary saved to {summary_csv_path}")

    def run_heterogeneity_analysis_cv(self, cv_values, fixed_mu, runs_per_file, n_repeats,
                                      folder_base_path, summary_csv_path):
        """
        Erweiterte Version:
        - Zeigt Fortschritt an
        - Überspringt bereits vorhandene Ergebnisse
        - Speichert Zwischenergebnisse nach jedem CV-Schritt
        """

        def make_params(n, k_off):
            new_paramset = copy.deepcopy(self.params.tele_params)
            setattr(new_paramset, "n", int(n))
            setattr(new_paramset, "k_off", k_off)
            return new_paramset

        # <<< Falls schon ein teilweise fertiges Summary existiert, lade es
        if os.path.exists(summary_csv_path):
            summary_df = pd.read_csv(summary_csv_path)
            done_cvs = set(summary_df['cv'].round(5))
            summary_records = summary_df.to_dict('records')
            print(f"⚙️  Fortsetzung erkannt. Bereits erledigte CVs: {sorted(done_cvs)}")
        else:
            summary_records = []
            done_cvs = set()

        total_tasks = len(cv_values) * n_repeats
        completed_tasks = 0

        start_time = time.time()

        for cv in cv_values:
            if round(cv, 5) in done_cvs:
                print(f"⏭ CV={cv:.2f} bereits abgeschlossen – überspringe.")
                completed_tasks += n_repeats
                continue

            k_float = 1 / (cv ** 2)
            n = int(np.ceil(k_float))
            k_off = n / fixed_mu

            print(f"\n[CV={cv:.2f}] -> n = {n}, k_off = {k_off:.4f}")

            paramset = make_params(n, k_off)
            per_run_percentiles = []

            folder_path = os.path.join(folder_base_path, f"cv_{cv:.2f}".replace(".", "_"))
            os.makedirs(folder_path, exist_ok=True)

            # <<< Prüfe, welche Replikate bereits existieren
            existing_files = {f for f in os.listdir(folder_path) if f.startswith("replicate_")}
            existing_ids = {int(f.split("_")[1].split(".")[0]) for f in existing_files}
            print(f"➡️  Bereits vorhandene Replikate: {sorted(existing_ids)}")

            for i in tqdm(range(n_repeats), desc=f"Simuliere CV={cv:.2f}", unit="rep"):
                replicate_id = i + 1
                if replicate_id in existing_ids:
                    completed_tasks += 1
                    continue  # <<< Überspringe bereits vorhandene CSVs

                t_values = np.array(self.get_t_esn_distribution(params=paramset, runs=runs_per_file))
                t_values = t_values[~np.isnan(t_values)]

                if len(t_values) == 0:
                    print(f"⚠️  Leeres t_ESRD-Array (CV={cv:.2f}, Rep={replicate_id}), überspringe Perzentile.")
                    heterogeneity = np.nan
                else:
                    percentiles = np.percentile(t_values, [5, 95])
                    heterogeneity = percentiles[1] - percentiles[0]
                    per_run_percentiles.append(heterogeneity)

                df = pd.DataFrame({
                    'cv': cv,
                    'mu': fixed_mu,
                    't_ESRD': t_values,
                    'reached_esrd': t_values <= 54 if len(t_values) > 0 else []
                })
                df.to_csv(os.path.join(folder_path, f"replicate_{replicate_id:03d}.csv"), index=False)

                completed_tasks += 1
                progress = completed_tasks / total_tasks * 100
                print(f"📊 Fortschritt: {progress:.1f}% ({completed_tasks}/{total_tasks})", end='\r')

            # <<< Nach jedem CV-Schritt Zwischenspeicherung
            if len(per_run_percentiles) == 0:
                heterogeneity_mean = np.nan
                heterogeneity_sem = np.nan
            else:
                per_run_percentiles = np.array(per_run_percentiles)
                heterogeneity_mean = np.mean(per_run_percentiles)
                heterogeneity_sem = np.std(per_run_percentiles, ddof=1) / np.sqrt(len(per_run_percentiles))

            summary_records.append({
                'cv': cv,
                'mu': fixed_mu,
                'n': n,
                'k_off': k_off,
                'heterogeneity_95_5': heterogeneity_mean,
                'heterogeneity_SEM': heterogeneity_sem
            })

            pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
            print(f"\n💾 Zwischenspeicherung: {summary_csv_path}")

        print(f"\n✅ Alle Simulationen abgeschlossen ({(time.time() - start_time) / 60:.1f} min).")

    def summarize_cv_scan(self, cv_values, folder_base_path, summary_csv_path,
                          bins=50, ignore_outliers=True):
        """
        For each CV in cv_values:
          - collect all t_ESRD values from replicate CSVs
          - shift them so that deterministic nzbw_on ESRD = 0
          - compute histogram, mean, std, median, IQR
          - save results into one summary CSV
        """

        import os
        import numpy as np
        import pandas as pd

        summary_records = []

        dmg_95, dmg_t95, norm = self.norm_fct()

        # --- 1. ESRD-Zeitpunkt für deterministische nzbw_on ermitteln ---
        self.acute_inflammation = None
        sim_det, _ = self.model.ode_simulator(self.params.nzbw_on)

        t_det_esrd = self.time2esn(sim_det, dmg_95)

        print(f"Referenz ESRD (NZBW_on) = {t_det_esrd:.2f} weeks")

        # --- 2. Scan über CV ---
        for cv in cv_values:
            folder_name = f"cv_{cv:.2f}".replace(".", "_")
            folder_path = os.path.join(folder_base_path, folder_name)

            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found for CV={cv:.2f}: {folder_path}")
                continue

            all_t = []
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and fname.startswith("replicate_"):
                    df = pd.read_csv(os.path.join(folder_path, fname))
                    if "t_ESRD" in df.columns:
                        all_t.extend(df["t_ESRD"].dropna().values)

            all_t = np.array(all_t, dtype=float)
            if len(all_t) == 0:
                print(f"⚠️ No t_ESRD values for CV={cv:.2f}")
                summary_records.append({
                    "cv": cv,
                    "mean_shifted_t_ESRD": np.nan,
                    "std_shifted_t_ESRD": np.nan,
                    "median_shifted_t_ESRD": np.nan,
                    "iqr_shifted_t_ESRD": np.nan,
                    "n_values": 0
                })
                continue

            # optional: remove extreme outliers
            if ignore_outliers:
                low, high = np.percentile(all_t, [1, 99])
                all_t = all_t[(all_t >= low) & (all_t <= high)]

            # --- 3. Shift mit deterministischem ESRD ---
            shifted_t = all_t - t_det_esrd

            # Statistik
            mean_val = float(np.mean(shifted_t))
            std_val = float(np.std(shifted_t, ddof=1))
            median_val = float(np.median(shifted_t))
            q1, q3 = np.percentile(shifted_t, [25, 75])
            iqr_val = float(q3 - q1)

            summary_records.append({
                "cv": cv,
                "mean_shifted_t_ESRD": mean_val,
                "std_shifted_t_ESRD": std_val,
                "median_shifted_t_ESRD": median_val,
                "iqr_shifted_t_ESRD": iqr_val,
                "n_values": len(shifted_t)
            })

            print(f"[CV={cv:.2f}] n={len(shifted_t)}, mean={mean_val:.2f}, "
                  f"std={std_val:.2f}, median={median_val:.2f}, IQR={iqr_val:.2f}")

        pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
        print(f"Summary saved to {summary_csv_path}")

    def run_heterogeneity_analysis_mu_fixed_ratio(self, mu_values, fixed_std, runs_per_file, n_repeats,
                                                  folder_base_path, summary_csv_path):
        """
        Erweiterte Version:
        - Zeigt Fortschritt an
        - Überspringt bereits vorhandene Ergebnisse
        - Speichert Zwischenergebnisse nach jedem mu-Schritt
        """

        # Referenzverhältnis berechnen
        mu_off_ref = self.params.tele_params.n / self.params.tele_params.k_off
        mu_on_ref = 1 / getattr(self.params.tele_params, "k_on", 1.0)
        r = mu_off_ref / mu_on_ref
        print(f"Referenzverhältnis mu_off / mu_on = {r:.3f}")

        def make_params(n, k_off, k_on):
            new_paramset = copy.deepcopy(self.params.tele_params)
            setattr(new_paramset, "n", int(n))
            setattr(new_paramset, "k_off", k_off)
            setattr(new_paramset, "k_on", k_on)
            return new_paramset

        # <<< Falls bereits ein Summary existiert → fortsetzen
        if os.path.exists(summary_csv_path):
            summary_df = pd.read_csv(summary_csv_path)
            done_mus = set(summary_df['mu'].round(5))
            summary_records = summary_df.to_dict('records')
            print(f"⚙️  Fortsetzung erkannt. Bereits erledigte μ-Werte: {sorted(done_mus)}")
        else:
            summary_records = []
            done_mus = set()

        total_tasks = len(mu_values) * n_repeats
        completed_tasks = 0
        start_time = time.time()

        for mu in mu_values:
            if round(mu, 5) in done_mus:
                print(f"⏭ μ={mu:.2f} bereits abgeschlossen – überspringe.")
                completed_tasks += n_repeats
                continue

            # Erlang-Parameter berechnen
            k_float = (mu / fixed_std) ** 2
            n = int(np.ceil(k_float))
            k_off = n / mu
            k_on = r / mu
            mu_on = 1 / k_on

            print(f"\n[μ_off={mu:.2f}] -> n = {n}, k_off = {k_off:.4f}, k_on = {k_on:.4f} (μ_on = {mu_on:.2f})")

            paramset = make_params(n, k_off, k_on)
            per_run_percentiles = []

            folder_path = os.path.join(folder_base_path, f"mu_{mu:.2f}".replace(".", "_"))
            os.makedirs(folder_path, exist_ok=True)

            # <<< Prüfe, welche Replikate bereits existieren
            existing_files = {f for f in os.listdir(folder_path) if f.startswith("replicate_")}
            existing_ids = {int(f.split("_")[1].split(".")[0]) for f in existing_files}
            print(f"➡️  Bereits vorhandene Replikate: {sorted(existing_ids)}")

            for i in tqdm(range(n_repeats), desc=f"Simuliere μ={mu:.2f}", unit="rep"):
                replicate_id = i + 1
                if replicate_id in existing_ids:
                    completed_tasks += 1
                    continue  # <<< Überspringe bereits vorhandene CSVs

                # Simulation
                t_values = np.array(self.get_t_esn_distribution(params=paramset, runs=runs_per_file))
                t_values = t_values[~np.isnan(t_values)]

                if len(t_values) == 0:
                    print(f"⚠️  Leeres t_ESRD-Array (μ={mu:.2f}, Rep={replicate_id}), überspringe Perzentile.")
                    heterogeneity = np.nan
                else:
                    percentiles = np.percentile(t_values, [5, 95])
                    heterogeneity = percentiles[1] - percentiles[0]
                    per_run_percentiles.append(heterogeneity)

                # Ergebnisse pro Run speichern
                df = pd.DataFrame({
                    'mu': mu,
                    'std': fixed_std,
                    'r': r,
                    't_ESRD': t_values,
                    'reached_esrd': t_values <= 54 if len(t_values) > 0 else []
                })
                df.to_csv(os.path.join(folder_path, f"replicate_{replicate_id:03d}.csv"), index=False)

                completed_tasks += 1
                progress = completed_tasks / total_tasks * 100
                print(f"📊 Fortschritt: {progress:.1f}% ({completed_tasks}/{total_tasks})", end='\r')

            # <<< Nach jedem μ-Schritt Zwischenspeicherung
            if len(per_run_percentiles) == 0:
                heterogeneity_mean = np.nan
                heterogeneity_sem = np.nan
            else:
                per_run_percentiles = np.array(per_run_percentiles)
                heterogeneity_mean = np.mean(per_run_percentiles)
                heterogeneity_sem = np.std(per_run_percentiles, ddof=1) / np.sqrt(len(per_run_percentiles))

            summary_records.append({
                'mu': mu,
                'std': fixed_std,
                'n': n,
                'k_off': k_off,
                'k_on': k_on,
                'mu_on': mu_on,
                'mu_off': mu,
                'r': r,
                'heterogeneity_95_5': heterogeneity_mean,
                'heterogeneity_SEM': heterogeneity_sem
            })

            pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
            print(f"\n💾 Zwischenspeicherung: {summary_csv_path}")

        print(f"\n✅ Alle Simulationen abgeschlossen ({(time.time() - start_time) / 60:.1f} min).")

    def summarize_mu_fixed_ratio_scan(self, mu_values, folder_base_path, summary_csv_path,
                                      bins=50, ignore_outliers=True):
        """
        For each mu (with fixed mu_off/mu_on ratio):
          - collect all t_ESRD values from replicate CSVs
          - shift them so that deterministic nzbw_on ESRD = 0
          - compute histogram, mean, std, median, IQR
          - save results into one summary CSV
        """

        import os
        import numpy as np
        import pandas as pd

        summary_records = []

        dmg_95, dmg_t95, norm = self.norm_fct()

        # --- 1. ESRD-Zeitpunkt für deterministische nzbw_on ermitteln ---
        self.acute_inflammation = None
        sim_det, _ = self.model.ode_simulator(self.params.nzbw_on)

        t_det_esrd = self.time2esn(sim_det, dmg_95)

        print(f"Referenz ESRD (NZBW_on) = {t_det_esrd:.2f} weeks")

        # --- 2. Scan über mu ---
        for mu in mu_values:
            folder_name = f"mu_{mu:.2f}".replace(".", "_")
            folder_path = os.path.join(folder_base_path, folder_name)

            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found for mu={mu:.2f}: {folder_path}")
                continue

            all_t = []
            all_r = None
            all_std = None

            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and fname.startswith("replicate_"):
                    df = pd.read_csv(os.path.join(folder_path, fname))
                    if "t_ESRD" in df.columns:
                        all_t.extend(df["t_ESRD"].dropna().values)
                    if all_r is None and "r" in df.columns:
                        all_r = float(df["r"].iloc[0])
                    if all_std is None and "std" in df.columns:
                        all_std = float(df["std"].iloc[0])

            all_t = np.array(all_t, dtype=float)
            if len(all_t) == 0:
                print(f"⚠️ No t_ESRD values for mu={mu:.2f}")
                summary_records.append({
                    "mu": mu,
                    "std": all_std,
                    "r": all_r,
                    "mean_shifted_t_ESRD": np.nan,
                    "std_shifted_t_ESRD": np.nan,
                    "median_shifted_t_ESRD": np.nan,
                    "iqr_shifted_t_ESRD": np.nan,
                    "n_values": 0
                })
                continue

            # optional: remove extreme outliers
            if ignore_outliers:
                low, high = np.percentile(all_t, [1, 99])
                all_t = all_t[(all_t >= low) & (all_t <= high)]

            # --- 3. Shift mit deterministischem ESRD ---
            shifted_t = all_t - t_det_esrd

            # Statistik
            mean_val = float(np.mean(shifted_t))
            std_val = float(np.std(shifted_t, ddof=1))
            median_val = float(np.median(shifted_t))
            q1, q3 = np.percentile(shifted_t, [25, 75])
            iqr_val = float(q3 - q1)

            summary_records.append({
                "mu": mu,
                "std": all_std,
                "r": all_r,
                "mean_shifted_t_ESRD": mean_val,
                "std_shifted_t_ESRD": std_val,
                "median_shifted_t_ESRD": median_val,
                "iqr_shifted_t_ESRD": iqr_val,
                "n_values": len(shifted_t)
            })

            print(f"[mu={mu:.2f}] n={len(shifted_t)}, mean={mean_val:.2f}, "
                  f"std={std_val:.2f}, median={median_val:.2f}, IQR={iqr_val:.2f}")

        pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
        print(f"Summary saved to {summary_csv_path}")

    #############
    # Plots Figure 1

    def fig1B(self, data_path: str, figure_format: str = '.pdf'):
        """
        Generate Figure 1 with side-by-side subplots:
        (1) Bar plot of proteinuria incidence (%) with absolute counts,
        (2) Point plot (mean ± SEM) of onset times for proteinuria:
            - mean = colored point
            - SEM = black error bars
        """
        print("(Analysis) Generating Figure 1 - Experimental proteinuria data.")

        # Load and prepare data
        df = pd.read_csv(data_path)
        df['time[weeks]'] = pd.to_numeric(df['time[weeks]'], errors='coerce').round().astype('Int64')

        condition_map = {
            'pIC': 'NZB/W',
            'aAGM1': 'aAGM1',
            'nzbw': 'NZB/W w/o pIC'
        }

        color_map = {
            'NZB/W': 'red',
            'aAGM1': 'steelblue',
            'NZB/W w/o pIC': 'black'
        }

        df = df[df['condition'].isin(condition_map.keys())].copy()
        df['group'] = df['condition'].map(condition_map)

        group_order = ['NZB/W', 'aAGM1', 'NZB/W w/o pIC']
        colors = [color_map[g] for g in group_order]

        # Incidence stats
        percentages, counts = [], []
        for group in group_order:
            subset = df[df['group'] == group]
            n_total = subset['time[weeks]'].notna().sum()
            n_positive = (subset['time[weeks]'] > 0).sum()
            percent = (n_positive / n_total) * 100 if n_total > 0 else 0
            percentages.append(percent)
            counts.append((n_positive, n_total))

        # Onset data
        df_nonzero = df[df['time[weeks]'] > 0].copy()
        df_nonzero['group'] = pd.Categorical(df_nonzero['group'], categories=group_order, ordered=True)
        means_onset = df_nonzero.groupby('group')['time[weeks]'].mean().reindex(group_order)
        print(means_onset)
        sems_onset = df_nonzero.groupby('group')['time[weeks]'].sem().reindex(group_order)
        print(sems_onset)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4), sharey=False)

        # --- Left: Incidence barplot
        bars = ax1.bar(group_order, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('proteinuria (%)')
        ax1.set_ylim(0, 100)
        ax1.set_yticks(range(0, 110, 20))
        ax1.set_xticklabels(group_order, rotation=90)

        for bar, (n_pos, n_total) in zip(bars, counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height - 5,
                f'{n_pos}/{n_total}',
                ha='center',
                va='top',
                fontsize=10,
                fontweight='bold',
                color='white'
            )
        self.format_ax(ax1)

        # --- Right: Onset plot with colored mean, black SEM
        positions = np.arange(len(group_order))

        for i, group in enumerate(group_order):
            mean = means_onset[group]
            sem = sems_onset[group]
            ax2.errorbar(i, mean, yerr=sem, fmt='o',
                         color=color_map[group],  # mean point in color
                         ecolor='black',  # SEM in black
                         elinewidth=2, capsize=5, markersize=8)

        ax2.set_ylabel('onset proteinuria (weeks)')
        ax2.set_ylim(19, means_onset.max() + sems_onset.max()+2)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(group_order, rotation=90)
        ax2.margins(x=0.2)  # Add horizontal whitespace like in barplot
        # Horizontal reference line at y = 1
        ax2.axhline(y=min(df_nonzero['lifespan[weeks]']), color='grey', linestyle='--', linewidth=1, zorder=3)
        ax2.axhline(y=max(df_nonzero['lifespan[weeks]']), color='grey', linestyle='--', linewidth=1, zorder=3)
        #ax2.fill_between(min(df_nonzero['lifespan[weeks]']), max(df_nonzero['lifespan[weeks]']), step="post", alpha=0.2, color='black')
        self.format_ax(ax2)

        plt.tight_layout()
        plt.savefig(f'results/subfigures/fig1B', dpi=800, bbox_inches='tight')
        plt.close()

    #############
    # Plots Figure 2
    def fig2C(self, filename):
        """
        Plot Damage (oben), IFN (Mitte) und IC (unten) Kinetiken für NZB/W, WT und Knockdown mit poly(I:C),
        inklusive ESRD-Markern, pIC- und Depletion-Markierungen. Jeder Plot hat eine eigene Legende.
        """
        print('(Analysis) Generating Fig IFN, Damage und IC kinetics')

        dmg_95, dmg_t95, norm = self.norm_fct()
        linewidth = 3

        fig, (ax_ifn, ax_dmg, ax_ic) = plt.subplots(3, 1, figsize=(4, 7), sharex=True, gridspec_kw={'hspace': 0.1})

        def mark_esrd(ax, t, y, color):
            ax.plot(t, y, 'o', color=color, markersize=6, label='ESRD reached')

        # NZB/W + pIC
        sim, cf = self.model.ode_simulator(self.params.nzbw_pIC)
        mask = sim.t <= dmg_t95
        t_trunc = sim.t[mask]
        cf_trunc = cf[mask]
        y_trunc = sim.y[:, mask]
        idx_damage = self.annotation['Damage']
        idx_ic = self.annotation['IC']

        y_damage = y_trunc[idx_damage] / norm[idx_damage]
        y_ic = y_trunc[idx_ic] / norm[idx_ic]

        ax_dmg.plot(t_trunc, y_damage, color=self.colors['Damage'], linewidth=linewidth, label='NZB/W')
        ax_ifn.plot(t_trunc, cf_trunc, color=self.colors['IFN'], linewidth=linewidth, label='NZB/W')
        ax_ic.plot(t_trunc, y_ic, color=self.colors['IC'], linewidth=linewidth, label='NZB/W')

        # WT + pIC
        sim, cf = self.model.ode_simulator(self.params.wt_pIC)
        ax_dmg.plot(sim.t, sim.y[idx_damage] / norm[idx_damage], color=self.colors['Damage'],
                    linewidth=linewidth, label='WT', alpha=0.5)
        ax_ifn.plot(sim.t, cf, color=self.colors['IFN'], linewidth=linewidth, label='WT', alpha=0.5)
        ax_ic.plot(sim.t, sim.y[idx_ic] / norm[idx_ic], color=self.colors['IC'],
                   linewidth=linewidth, label='WT', alpha=0.5)

        if not np.all(mask):
            mark_esrd(ax_dmg, t_trunc[-1], y_damage[-1], self.colors['Damage'])
            mark_esrd(ax_ifn, t_trunc[-1], cf_trunc[-1], self.colors['IFN'])
            mark_esrd(ax_ic, t_trunc[-1], y_ic[-1], self.colors['IC'])

        # pIC-Markierung (über dem obersten Plot)
        pic_start = 19
        pic_end = self.params.wt_pIC.t_end_trigger
        ax_ifn.axvspan(pic_start, pic_end, ymin=1.04, ymax=1.06, color='gray', alpha=0.8, clip_on=False)
        ax_ifn.text((pic_start + pic_end) / 2, 1.11, 'pIC', ha='center', transform=ax_ifn.get_xaxis_transform(),
                    fontsize=11, color='gray')

        # Achsen und Layout
        for ax, label in zip([ax_dmg, ax_ifn, ax_ic], ['damage (norm)', 'IFN-I (norm)', 'IC (norm)']):
            ax.set_xlim(18, 28)
            ax.set_ylim(bottom=-0.1)
            ax.set_ylabel(label)
            self.format_ax(ax)

        ax_ic.set_xlabel('time (weeks)')

        # Einzelne Legenden
        for ax in [ax_dmg, ax_ifn, ax_ic]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # nur wenn was gezeichnet wurde
                ax.legend(handles, labels, loc='upper left', fontsize=9, frameon=False)

        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def fig2E(self, disruption_results, save_path):
        """
        Create a bar plot comparing the normalized damage value at t=25
        for feedback-disrupted systems.

        Args:
            disruption_results (dict): Simulations from the network_disruption_screen method.
            save_path (str): Path to save the plot.
        """
        print('(LupusAnalysis) Generate - Damage at t=25 analysis.')

        # Get normalization factors
        _, _, norm = self.norm_fct()

        # Collect damage values at t=25
        damage_at_t25 = {}
        for key, sim in disruption_results.items():
            t = sim.t
            damage_traj = sim.y[self.model.annotation['Damage']] / norm[self.annotation['Damage']]

            # Find the index closest to t = 25
            idx_t25 = np.argmin(np.abs(t - 25))
            value_t25 = damage_traj[idx_t25]

            damage_at_t25[key] = value_t25

        # Convert to DataFrame
        damage_df = pd.DataFrame(list(damage_at_t25.items()), columns=['Disruption', 'DamageAtT25'])

        # Selection and sorting
        selected_keys = ['ref',
                         # Einzel-Knockouts
                         'F1', 'F2', 'F3', 'F4', 'F5',
                         # Doppel-Knockouts
                         'F1/3', 'F1/5', 'F3/5',
                         # Dreifach-Knockouts
                         'F1/3/5', 'F3/4/5',
                         # Fünffach-Knockout
                         'F1/2/3/4/5']

        damage_df = damage_df[damage_df['Disruption'].isin(selected_keys)]
        damage_df['SortKey'] = damage_df['Disruption'].apply(lambda x: selected_keys.index(x))
        damage_df = damage_df.sort_values('SortKey').drop(columns='SortKey')

        # Plot
        fig, ax = plt.subplots(figsize=(6, 3.5))

        disruptions = damage_df['Disruption'].tolist()
        values = damage_df['DamageAtT25'].tolist()

        # Farben
        gray_color = 'gray'
        black_color = 'black'

        # Farbwahl pro Balken
        colors = []
        for d in disruptions:
            if d in ['ref', 'F3/4/5']:
                colors.append(gray_color)
            else:
                colors.append(black_color)

        # Balken zeichnen
        bars = ax.bar(disruptions, values, color=colors, edgecolor='white',
                      linewidth=1.5, zorder=2, alpha=0.85)

        # Linien für ref und F3/4/5
        y_ref = values[disruptions.index('ref')]
        y_wt = values[disruptions.index('F3/4/5')]

        ax.axhline(y=y_ref, color=gray_color, linestyle='--', linewidth=1.5, zorder=3)
        ax.axhline(y=y_wt, color=gray_color, linestyle='--', linewidth=1.5, zorder=3)

        # Achsenbeschriftungen
        ax.set_ylabel('damage (norm)')

        # Angepasste X-Achsen-Labels
        labels = []
        for d in disruptions:
            if d == 'ref':
                labels.append('NZB/W')
            elif d == 'F3/4/5':
                labels.append('WT (F3/4/5)')
            else:
                labels.append(d)
        ax.set_xticks(range(len(disruptions)))
        ax.set_xticklabels(labels, rotation=90)

        # Formatierung
        ax.set_ylim(0, 1)
        self.format_ax(ax)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def fig2F(self, screen_res, name):
        """
        Create a barplot showing the results of the sensitivity analysis
        for each tested parameter at factor 0.5 and 2, excluding the reference bar.

        Parameters:
        - screen_res: DataFrame produced by `sens_analysis`
        - name: filename to save the figure
        """
        print('(LupusAnalysis) Generate Fig2G - Sensitivity analysis (without reference bar).')

        fig, ax = plt.subplots(1, 1, figsize=(6.3, 2.5))  # Single plot now
        self.format_ax(ax)

        ax.axhline(y=screen_res['delta_weeks'][0], linestyle='--', linewidth=1.5, color='black', zorder=1)
        ax.tick_params(axis='x', labelrotation=90)

        # Plot sensitivity results (excluding the first row = baseline)
        sns.barplot(
            data=screen_res[1:], x='name', y='delta_weeks', hue='fac',
            palette=['white', 'black'], edgecolor='black', linewidth=2, ax=ax, zorder=2, alpha=0.8
        )

        # Legend
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, ["down", "up"], frameon=False, bbox_to_anchor=(1.04, 1), loc="upper right", ncols=2, fontsize=14)

        # Axis labels and limits
        ax.set_xlabel('')
        ax.set_ylabel(r'$t_{ESRD}$ (weeks)', fontsize=16)
        ax.set_ylim(2, 35)

        # Add red asterisks for bars > 40 weeks
        for bar in ax.patches:
            height = bar.get_height()
            if height > 40:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    29,
                    '*',
                    ha='center',
                    va='bottom',
                    fontsize=16,
                    color='red',
                    zorder=3
                )

        # Text legend
        fig.text(0.4, 0.9, '* = ESRD not reached', ha='center', fontsize=14, color='red')

        # Save figure
        plt.savefig(name, bbox_inches='tight')
        plt.close()

    # Plots Figure S2
    def figS2A(self, variables, filename: str):
        """
        Generate Figure 2B: End-Stage Norm (ESN) explanation using damage and cell trajectories.
        Both y-axes on the left, cell kinetics stop at t_ESN, fontsize adjusted.
        """
        print("(Analysis) Generating Figure 2B - Explanation of End-Stage Norm (ESN).")

        dmg_95, dmg_t95, norm = self.norm_fct()
        sim = self.model.ode_simulator(self.params.nzbw_pIC)[0]
        sol_wt, cf_wt = self.model.ode_simulator(self.params.wt_pIC)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        linewidth = 3
        vline_width = 1.5
        markersize = 50

        # --- LINKER PLOT: DAMAGE ---
        dmg_index = self.annotation['Damage']
        t_cut_idx = np.searchsorted(sim.t, dmg_t95)
        t_cut = sim.t[:t_cut_idx + 1]
        y_cut = sim.y[dmg_index][:t_cut_idx + 1]

        #print(sol_wt.t[dmg_index], sol_wt.y[dmg_index])

        axes[0].plot(t_cut, y_cut, color='black', linewidth=linewidth, label='NZB/W')
        axes[0].plot(sol_wt.t, sol_wt.y[dmg_index], alpha=0.5, color='black', linewidth=linewidth, label='WT')
        axes[0].scatter([dmg_t95], [norm[dmg_index]], color='black', s=markersize, zorder=5)
        axes[0].axvline(x=dmg_t95, color='black', linestyle='--', linewidth=vline_width, ymax=0.9)

        # --- RECHTER PLOT: ZELLKINETIK (endet bei t_ESN) ---
        for var in variables:
            idx = self.annotation[var]
            clr = self.colors[var]
            y_cut = sim.y[idx][:t_cut_idx + 1]  # Cut at t_ESN
            axes[1].plot(t_cut, y_cut, color=clr, linewidth=linewidth)
            axes[1].scatter([dmg_t95], [norm[idx]], color=clr, marker='o', s=markersize,
                            linewidth=linewidth, zorder=6, label=f'norm {var}')

        # Vertikale Linie bei t_ESN
        axes[1].axvline(x=dmg_t95, color='black', linestyle='--', linewidth=vline_width)

        # --- ACHSENFORMATIERUNG ---
        for ax in axes:
            ax.set_xlim(18.5, 28)
            ax.set_ylim(bottom=-0.1)
            ax.set_xlabel('time (weeks)', fontsize=14)
            self.format_ax(ax)

            # y-Achsen links
            ax.yaxis.tick_left()
            ax.yaxis.set_label_position("left")

        axes[0].set_ylabel('damage (a.u.)', fontsize=14)
        axes[1].set_ylabel('cells (a.u.)', fontsize=14)

        axes[0].text(dmg_t95 - 5, norm[dmg_index] + 0.5, '95% max. damage', fontsize=12)

        # --- pIC-MARKIERUNG OBEN ---
        t0 = self.params.wt_pIC.t_start_trigger
        t1 = self.params.wt_pIC.t_end_trigger
        for ax in axes:
            ax.axvspan(t0, t1, ymin=1.04, ymax=1.06, color='gray', alpha=0.8, clip_on=False)
            ax.text((t0 + t1) / 2, 1.11, 'pIC', ha='center',
                    transform=ax.get_xaxis_transform(), fontsize=12, color='gray')

        axes[0].legend(fontsize=11, frameon=False, loc='upper left')
        axes[1].legend(fontsize=11, frameon=False, loc='upper left')

        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def figS2BC(self, filename: str):
        """
        Vergleicht NZB/W vs WT Kinetiken:
        - Hauptfigur: alle Zustände
        - Separate Figur: Damage & IC
        - Separate Figur: Cytokines (C34, C43, C67, C76)
        """
        print("(Analysis) Generating NZB/W vs WT Kinetics Plot.")

        plot_vars = [
            'IFN', 'prtILC', 'tILC', 'vNK',
            'MOMA', 'MO', 'iPEC', 'EC',
            'C34', 'C43', 'C67', 'C76',
            'Damage', 'IC'
        ]

        dmg_95, dmg_t95, norm = self.norm_fct()

        sol_nzbw, cf_nzbw = self.model.ode_simulator(self.params.nzbw_pIC)
        sol_wt, cf_wt = self.model.ode_simulator(self.params.wt_pIC)

        linewidth = 3
        alpha_wt = 0.5

        # ---------- FIGURE 2: DAMAGE & IC ----------
        print("(Analysis) Generating Damage & IC comparison plot.")
        dmg_ic_vars = ['MO', 'Damage']
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.7), sharex=True)
        linewidth = 3

        for idx, var in enumerate(dmg_ic_vars):
            ax = axes[idx]
            i = self.annotation[var]
            ax.plot(sol_nzbw.t, sol_nzbw.y[i]/norm[i], color=self.colors[var], linewidth=linewidth, label='NZB/W')
            ax.plot(sol_wt.t, sol_wt.y[i]/norm[i], color=self.colors[var], linewidth=linewidth, alpha=alpha_wt, label='WT')
            ax.set_xlim(0, 18)
            ax.set_ylim(-0.01, 0.2)
            ax.set_ylabel(f"{var} (norm)", fontsize=14)
            ax.set_xlabel("time (weeks)", fontsize=14)
            self.format_ax(ax)
            ax.legend(fontsize=11, frameon=False)

        plt.tight_layout()
        plt.savefig(filename + "B.pdf", bbox_inches='tight')
        plt.close()

        # ---------- FIGURE 3: CYTOKINES ----------
        print("(Analysis) Generating Cytokine comparison plot.")
        cytokine_vars = ['C34', 'C43', 'C67', 'C76']
        fig, axes = plt.subplots(1, 4, figsize=(12.2, 2.6), sharex=True)
        linewidth = 3

        for idx, var in enumerate(cytokine_vars):
            ax = axes[idx]
            i = self.annotation[var]
            ax.plot(sol_nzbw.t, sol_nzbw.y[i]/norm[i], color=self.colors[var], linewidth=linewidth, label='NZB/W')
            ax.plot(sol_wt.t, sol_wt.y[i]/norm[i], color=self.colors[var], linewidth=linewidth, alpha=alpha_wt, label='WT')
            ax.set_xlim(18.5, 27)
            ax.set_ylim(-0.1, 2.1)
            if var in ['C34']:
                ax.set_ylabel('C1 (norm)')
            if var in ['C43']:
                ax.set_ylabel('C2 (norm)')
            if var in ['C67']:
                ax.set_ylabel('C3 (norm)')
            if var in ['C76']:
                ax.set_ylabel('C4 (norm)')
            #ax.set_ylabel(f"{var} (norm)", fontsize=18)
            ax.set_xlabel("time (weeks)")
            self.format_ax(ax)
            ax.legend(fontsize=11, frameon=False)

        plt.tight_layout()
        plt.savefig(filename + "C.pdf", bbox_inches='tight')
        plt.close()

    def figS2D(self, disruption_results, save_path):
        """
        Create a bar plot comparing the normalized damage value at t=25
        for feedback-disrupted systems.

        Args:
            disruption_results (dict): Simulations from the network_disruption_screen method.
            save_path (str): Path to save the plot.
        """
        print('(LupusAnalysis) Generate - Damage at t=25 analysis.')

        # Get normalization factors
        _, _, norm = self.norm_fct()

        # Collect damage values at t=25
        damage_at_t25 = {}
        for key, sim in disruption_results.items():
            t = sim.t
            damage_traj = sim.y[self.model.annotation['Damage']] / norm[self.annotation['Damage']]

            # Find the index closest to t = 25
            idx_t25 = np.argmin(np.abs(t - 25))
            value_t25 = damage_traj[idx_t25]

            damage_at_t25[key] = value_t25

        # Convert to DataFrame
        damage_df = pd.DataFrame(list(damage_at_t25.items()), columns=['Disruption', 'DamageAtT25'])

        # Selection and sorting
        selected_keys = ['ref',
                         # Einzel-Knockouts
                         'F1', 'F2', 'F3', 'F4', 'F5',
                         # Doppel-Knockouts
                         'F1/2', 'F1/3', 'F1/4', 'F1/5',
                         'F2/3', 'F2/4', 'F2/5',
                         'F3/4', 'F3/5', 'F4/5',
                         # Dreifach-Knockouts
                         'F1/2/3', 'F1/2/4', 'F1/2/5',
                         'F1/3/4', 'F1/3/5', 'F1/4/5',
                         'F2/3/4', 'F2/3/5', 'F2/4/5',
                         'F3/4/5',
                         # Vierfach-Knockouts
                         'F1/2/3/4', 'F1/2/3/5', 'F1/2/4/5',
                         'F1/3/4/5', 'F2/3/4/5',
                         # Fünffach-Knockout
                         'F1/2/3/4/5']

        damage_df = damage_df[damage_df['Disruption'].isin(selected_keys)]
        damage_df['SortKey'] = damage_df['Disruption'].apply(lambda x: selected_keys.index(x))
        damage_df = damage_df.sort_values('SortKey').drop(columns='SortKey')

        # Plot
        fig, ax = plt.subplots(figsize=(6, 3.5))

        disruptions = damage_df['Disruption'].tolist()
        values = damage_df['DamageAtT25'].tolist()

        # Farben
        gray_color = 'gray'
        black_color = 'black'

        # Farbwahl pro Balken
        colors = []
        for d in disruptions:
            if d in ['ref', 'F3/4/5']:
                colors.append(gray_color)
            else:
                colors.append(black_color)

        # Balken zeichnen
        bars = ax.bar(disruptions, values, color=colors, edgecolor='white',
                      linewidth=1.5, zorder=2, alpha=0.85)

        # Linien für ref und F3/4/5
        y_ref = values[disruptions.index('ref')]
        y_wt = values[disruptions.index('F3/4/5')]

        ax.axhline(y=y_ref, color=gray_color, linestyle='--', linewidth=1.5, zorder=3)
        ax.axhline(y=y_wt, color=gray_color, linestyle='--', linewidth=1.5, zorder=3)

        # Achsenbeschriftungen
        ax.set_ylabel('damage (norm)')

        # Angepasste X-Achsen-Labels
        labels = []
        for d in disruptions:
            if d == 'ref':
                labels.append('NZB/W')
            elif d == 'F3/4/5':
                labels.append('WT (F3/4/5)')
            else:
                labels.append(d)
        ax.set_xticks(range(len(disruptions)))
        ax.set_xticklabels(labels, rotation=90)

        # Formatierung
        ax.set_ylim(0, 1)
        self.format_ax(ax)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    #############
    # Plots Figure 3

    def fig3B(self, name: str):
        """
        Plot input distributions: Exponential (k_off) and Erlang (n, k_on).
        y-axis: input distributions
        x-axis: time (weeks)
        """

        from scipy.stats import expon, erlang

        # time grid
        t = np.linspace(0, 2, 100)

        # exponential distribution
        pdf_exp = expon(scale=1.0 / self.params.tele_params.k_on).pdf(t)

        # erlang distribution
        pdf_erlang = erlang(a=self.params.tele_params.n, scale=1.0 / self.params.tele_params.k_off).pdf(t)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(3, 2.5))

        ax.plot(t, pdf_exp, color='black', linewidth=3, label=r'Exp$(\tau_{off})$')
        ax.plot(t, pdf_erlang, color=self.colors['tILC'], linewidth=3, label=r'Erlang$(\tau_{on}, \sigma_{on})$')

        # styling
        ax.set_ylabel('input distribution', color='black', fontsize=14)
        ax.set_xlabel('time (weeks)', fontsize=14)
        ax.set_xlim(0, 2)
        ax.set_xticks([0,1,2])
        ax.set_yticks([0, 1.5])
        #ax.set_ylim(0, max(pdf_exp.max(), pdf_erlang.max()) * 1.2)

        ax.legend(loc='upper right', frameon=False, fontsize=10)
        self.format_ax(ax)

        plt.tight_layout()
        plt.savefig(name + '_input_distributions.pdf', bbox_inches='tight')
        plt.close()

    def fig3C(self, runs: int, name: str):
        """
        Plots acute inflammation signal (Gillespie) together with IC and Damage kinetics (ODE).
        Includes a zoom-in inset for inflammation and IC.
        """
        print('(HybridLupusModel) Generate Fig4B - Acute inflammation signal and IC kinetics.')
        dmg_95, dmg_t95, norm = self.norm_fct()

        for j in range(runs):
            simulator = GillespieSimulator(runs=runs)
            self.model.acute_inflammation = simulator.save_gillespie()
            sim = self.model.ode_simulator(self.params.nzbw_inc_ic)[0]
            print(sim)

            self._plot_inflammation_zoom(
                acute_infl=self.model.acute_inflammation,
                t=sim.t,
                damage=sim.y[self.annotation['Damage']] / norm[self.annotation['Damage']],
                ic=sim.y[self.annotation['IC']] / norm[self.annotation['IC']],
                filename=f"{name}_{j}"
            )
            self._plot_inflammation_damage(acute_infl=self.model.acute_inflammation,
                t=sim.t,
                damage=sim.y[self.annotation['Damage']] / norm[self.annotation['Damage']],
                filename=f"{name}_{j}")
        self.model.acute_inflammation = None

    def fig3D(self, runs: int, name: str):
        """
        Generate stochastic inflammation signals (hybrid simulations) multiple times
        and plot mean ± std damage kinetics for the hybrid model.
        Each kinetic runs until damage reaches 1 and is then fixed at 1.
        The mean and std are truncated when all trajectories have reached 1.
        """

        print('(HybridLupusModel) Generate mean ± std hybrid damage kinetics (terminated when all runs reached 1).')

        import numpy as np
        import matplotlib.pyplot as plt

        dmg_95, dmg_t95, norm = self.norm_fct()

        # --- Hybrid simulations ---
        hybrid_curves = []
        hybrid_times = None

        for j in range(runs):
            simulator = GillespieSimulator(runs=1)
            self.model.acute_inflammation = simulator.save_gillespie()
            sim = self.model.ode_simulator(self.params.nzbw_inc_ic)[0]

            t = sim.t
            dmg = sim.y[self.annotation['Damage']] / norm[self.annotation['Damage']]

            # truncate until first time damage >= 1
            if np.any(dmg >= 1.0):
                idx_end = np.argmax(dmg >= 1.0)
                t = t[:idx_end + 1]
                dmg = dmg[:idx_end + 1]
                dmg[-1] = 1.0

            # create common grid
            if hybrid_times is None:
                hybrid_times = np.linspace(t[0], t[-1], 500)

            # interpolate to common grid
            dmg_interp = np.interp(hybrid_times, t, dmg)

            # fix at 1 after reaching
            idx_1 = np.argmax(dmg_interp >= 1.0)
            if dmg_interp[idx_1] >= 1.0:
                dmg_interp[idx_1:] = 1.0

            hybrid_curves.append(dmg_interp)

        self.model.acute_inflammation = None

        # --- Stack to array ---
        hybrid_curves = np.vstack(hybrid_curves)

        # --- Find global endpoint (when all curves reached 1) ---
        idx_ends = [np.argmax(curve >= 1.0) for curve in hybrid_curves]
        idx_end_all = np.max(idx_ends)

        # truncate arrays up to the latest reaching time
        hybrid_times = hybrid_times[:idx_end_all + 1]
        hybrid_curves = hybrid_curves[:, :idx_end_all + 1]

        # --- Mean and std ---
        dmg_mean = np.mean(hybrid_curves, axis=0)
        dmg_std = np.std(hybrid_curves, axis=0)

        # ensure std ends at 0 when all reached 1
        if np.allclose(dmg_mean[-1], 1.0):
            dmg_std[-1] = 0.0

        # --- Deterministic reference simulations ---
        sim_det, _ = self.model.ode_simulator(self.params.nzbw_inc_ic)
        dmg_det = sim_det.y[self.annotation['Damage']] / norm[self.annotation['Damage']]
        t_det = sim_det.t
        mask_det = dmg_det <= 1.0
        if not np.all(mask_det):
            t_det = t_det[mask_det]
            dmg_det = dmg_det[mask_det]
        dmg_det[dmg_det >= 1.0] = 1.0

        sim_det_pic, _ = self.model.ode_simulator(self.params.nzbw_on)
        dmg_det_pic = sim_det_pic.y[self.annotation['Damage']] / norm[self.annotation['Damage']]
        t_det_pic = sim_det_pic.t
        mask_det_pic = dmg_det_pic <= 1.0
        if not np.all(mask_det_pic):
            t_det_pic = t_det_pic[mask_det_pic]
            dmg_det_pic = dmg_det_pic[mask_det_pic]
        dmg_det_pic[dmg_det_pic >= 1.0] = 1.0

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(4, 3))

        # hybrid mean ± std
        ax.plot(hybrid_times, dmg_mean, color='black', linewidth=3, label='hybrid (mean ± std)')
        ax.fill_between(hybrid_times, dmg_mean - dmg_std, dmg_mean + dmg_std, color='black', alpha=0.6)

        # endpoint marker
        ax.plot(hybrid_times[-1], dmg_mean[-1], 'o', color='black', markersize=5)

        # deterministic baseline
        ax.plot(t_det, dmg_det, color='lightgrey', linewidth=3, label='deterministic')
        ax.plot(t_det[-1], dmg_det[-1], 'o', color='lightgrey', markersize=5)

        # deterministic pIC
        ax.plot(t_det_pic, dmg_det_pic, color='grey', linewidth=3, label='deterministic (pIC)')
        ax.plot(t_det_pic[-1], dmg_det_pic[-1], 'o', color='grey', markersize=5)

        # styling
        ax.set_ylabel('damage (norm)', color='black', fontsize=14)
        ax.set_xlabel('time (weeks)', fontsize=14)
        ax.set_ylim(0, 1.5)
        ax.set_xlim(18, 36)
        self.format_ax(ax)
        ax.legend(loc='upper left', frameon=False, fontsize=10)

        plt.tight_layout()
        plt.savefig(name, bbox_inches='tight')
        plt.close()

    def fig3E(self, runs: int = 50,filename: str = "results/subfigures/fig3E"):
        """
        Simuliert 'runs' akute Inflammations-Signale und plottet die normierten Damage-Kinetiken,
        jeweils abgeschnitten bei Erreichen von ESRD (dmg_95). Zusätzlich wird die mittlere Kinetik
        (mean damage) über alle gültigen Verläufe dargestellt.
        """

        print(f"(HybridLupusModel) Plotting ESRD-truncated damage kinetics ({runs} runs)...")

        dmg_95, dmg_t95, norm = self.norm_fct()

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.set_xlim(19, 35)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("time (weeks)")
        ax.set_ylabel("damage (ESRD)")
        self.format_ax(ax)

        all_interp_damages = []

        # Einheitliche Zeitachse für Mittelwertbildung
        common_time = np.linspace(19, 54, 500)

        collected = 0
        attempts = 0
        max_attempts = runs * 3  # falls manche Runs keine ESRD erreichen

        while collected < runs and attempts < max_attempts:
            attempts += 1

            simulator = GillespieSimulator(runs=runs)
            self.model.acute_inflammation = simulator.save_gillespie()

            sim = self.model.ode_simulator(self.params.nzbw_inc_ic)[0]
            t = sim.t
            y_dmg = sim.y[self.annotation['Damage']] / norm[self.annotation['Damage']]

            t_esrd = self.time2esn(sim, dmg_95, nan=True)
            if np.isnan(t_esrd):
                continue

            mask = sim.t <= t_esrd
            t_trunc = sim.t[mask]
            y_trunc = y_dmg[mask]

            # Exakten Endpunkt ergänzen
            if t_trunc[-1] < t_esrd and len(t_trunc) >= 1:
                y_last = np.interp(t_esrd, [t_trunc[-1], sim.t[mask.sum()]],
                                   [y_trunc[-1], y_dmg[mask.sum()]])
                t_trunc = np.append(t_trunc, t_esrd)
                y_trunc = np.append(y_trunc, y_last)

            # Interpolation auf gemeinsame Zeitachse
            y_interp = np.interp(common_time, t_trunc, y_trunc, left=np.nan, right=np.nan)
            all_interp_damages.append(y_interp)

            # Einzelkurve
            ax.plot(t_trunc, y_trunc, color="black", linewidth=0.8, alpha=0.6)
            collected += 1

        if all_interp_damages:
            damage_array = np.array(all_interp_damages)
            with np.errstate(invalid='ignore'):
                mean_damage = np.nanmean(damage_array, axis=0)
            ax.plot(common_time, mean_damage, color=self.colors['tILC'], linewidth=2, label="mean damage")

        plt.tight_layout()
        plt.savefig(filename + ".pdf", bbox_inches='tight', dpi=800)
        plt.close()
        self.model.acute_inflammation = None

    def fig3F(
            self,
            mu_values, fixed_std, folder_base_path, summary_csv_path,
            bins=50, ignore_outliers=True, hist_folder="histograms",
            ncols=1, xlim=None
    ):
        """
        Collect t_ESRD from replicate CSVs for each mu, shift times by the
        deterministic pIC time (t_det_pic[-1]), compute histogram, mean, std,
        save combined figure and a summary CSV. x-axis shows t_stoch = t_ESRD - t_shift.
        """

        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        os.makedirs(hist_folder, exist_ok=True)

        # --- determine deterministic pIC time (t_shift) exactly as in your plot function ---
        # use same normalization as in plotting to be consistent
        try:
            _, _, norm = self.norm_fct()
        except Exception:
            # fallback if norm_fct returns only one item
            norm = self.norm_fct()[-1]

        sim_det_pic, _ = self.model.ode_simulator(self.params.nzbw_on)
        t_det_pic = np.asarray(sim_det_pic.t)
        dmg_det_pic = np.asarray(sim_det_pic.y[self.annotation['Damage']]) / norm[self.annotation['Damage']]

        # apply mask (keep times where dmg <= 1.0) and take the last time point of that masked array
        mask_det_pic = dmg_det_pic <= 1.0
        if not np.all(mask_det_pic):
            t_det_pic_masked = t_det_pic[mask_det_pic]
            if len(t_det_pic_masked) > 0:
                t_shift = float(t_det_pic_masked[-1])
            else:
                t_shift = float(t_det_pic[-1])
        else:
            # if all values are <= 1.0, use last time point
            t_shift = float(t_det_pic[-1])

        print(f"→ Using t_shift = {t_shift:.4f} weeks (deterministic pIC time, t_det_pic[-1])")

        summary_records = []
        all_hist_data = {}

        for mu in mu_values:
            folder_name = f"mu_{mu:.2f}".replace(".", "_")
            folder_path = os.path.join(folder_base_path, folder_name)
            k_float = (mu / fixed_std) ** 2
            n = int(np.ceil(k_float))
            k_off = n / mu

            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found for mu={mu:.2f}: {folder_path}")
                continue

            all_t = []
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and fname.startswith("replicate_"):
                    df = pd.read_csv(os.path.join(folder_path, fname))
                    if "t_ESRD" in df.columns:
                        all_t.extend(df["t_ESRD"].dropna().values)

            all_t = np.array(all_t, dtype=float)
            if len(all_t) == 0:
                print(f"⚠️ No t_ESRD values for mu={mu:.2f}")
                summary_records.append({
                    "mu": mu,
                    "k_off": k_off,
                    "mean_t_stoch": np.nan,
                    "std_t_stoch": np.nan,
                    "n_values": 0
                })
                continue

            # --- Shift relative to deterministic pIC time (no interpolation) ---
            all_t_shifted = all_t - t_shift

            # optional: remove extreme outliers after shifting
            if ignore_outliers and len(all_t_shifted) > 2:
                low, high = np.percentile(all_t_shifted, [1, 99])
                all_t_shifted = all_t_shifted[(all_t_shifted >= low) & (all_t_shifted <= high)]

            mean_val = float(np.mean(all_t_shifted)) if len(all_t_shifted) > 0 else np.nan
            std_val = float(np.std(all_t_shifted, ddof=1)) if len(all_t_shifted) > 1 else np.nan

            summary_records.append({
                "mu": mu,
                "k_off": k_off,
                "mean_t_stoch": mean_val,
                "std_t_stoch": std_val,
                "n_values": len(all_t_shifted)
            })

            print(f"[mu={mu:.2f}] n={len(all_t_shifted)}, mean={mean_val:.2f}, std={std_val:.2f}")

            all_hist_data[mu] = (all_t_shifted, mean_val, std_val)

        # save summary
        pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
        print(f"Summary saved to {summary_csv_path}")

        # --- gemeinsame Figure mit Subplots ---
        n = len(all_hist_data)
        if n == 0:
            print("No data to plot.")
            return

        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4 * ncols, 3 * nrows),
                                 sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)

        for i, (mu, (all_t_shifted, mean_val, std_val)) in enumerate(sorted(all_hist_data.items())):
            ax = axes[i]

            # Farbwahl: Standard schwarz, außer 2. (i==1) grau, vorletztes gelb (IFN)
            if i == 1:
                bar_color = "gray"
            elif i == len(all_hist_data) - 2:
                bar_color = self.colors.get('IFN', 'gold')
            else:
                bar_color = "black"

            # Histogramm
            ax.hist(all_t_shifted, bins=bins, color=bar_color, edgecolor=bar_color, alpha=0.8)

            # Mean- und Std-Linien
            if not np.isnan(mean_val):
                ax.axvline(mean_val, color='grey', linestyle="-", linewidth=2, label="mean")
            if not np.isnan(std_val):
                ic_color = self.colors.get('IC', 'tab:blue')
                ax.axvline(mean_val - std_val, color=ic_color, linestyle=":", linewidth=2)
                ax.axvline(mean_val + std_val, color=ic_color, linestyle=":", linewidth=2)

            if xlim:
                ax.set_xlim(*xlim)

            # y-label nur links
            if i % ncols == 0:
                ax.set_ylabel("# runs", fontsize=14)
            # x-label nur unten
            if i // ncols == nrows - 1:
                ax.set_xlabel(r"$t_{\mathrm{stoch}}$ (weeks)", fontsize=14)

            self.format_ax(ax)

            ax.set_yticks([0,150,300])

        # überzählige Achsen unsichtbar machen
        for j in range(len(all_hist_data), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(hist_folder, "fig3F.pdf")
        plt.savefig(out_path)
        plt.close(fig)

        print(f"Combined shifted histogram figure saved to {out_path}")


    def fig3G(self, filename):
        """
        Plot shifted mean (black) and std (IC color) of t_ESRD
        for τ_on, CV, and switching frequency (1/τ_on).
        Adds grey arrows exactly under data points and black explanatory text below legends.
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        summary_paths = {
            r"$\tau_{on}$": "data/heterogeneity_mu_shifted_summary.csv",
            "CV": "data/heterogeneity_cv_shifted_summary.csv",
            "switching": "data/heterogeneity_mu_ratio_shifted_summary.csv"
        }

        x_labels = {
            r"$\tau_{on}$": r"$\tau_{on}$ (weeks)",
            "CV": r"CV $(\sigma_{on} / \tau_{on})$",
            "switching": r"switching frequency (weeks$^{-1}$)"
        }

        param_cols = {
            r"$\tau_{on}$": "mu",
            "CV": "cv",
            "switching": "mu"
        }

        fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 3), sharey=True)
        axes = axes.flatten()
        linewidth = 2
        capsize = 3
        vertical_offset = 0.7  # controls arrow length

        for ax, (label, path) in zip(axes, summary_paths.items()):
            df = pd.read_csv(path)
            col = param_cols[label]
            x = df[col].values
            y_mean = df["mean_shifted_t_ESRD"].values
            y_std = df["std_shifted_t_ESRD"].values

            # Drop first point for τ_on
            if label == r"$\tau_{on}$":
                x, y_mean, y_std = x[1:], y_mean[1:], y_std[1:]

            # --- switch to frequency scale if switching ---
            if label == "switching":
                x = 1 / x  # transform to frequency
                order = np.argsort(x)
                x, y_mean, y_std = x[order], y_mean[order], y_std[order]

            # --- mean (black) ---
            ax.errorbar(
                x, y_mean,
                fmt='-o', markersize=8,
                markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=2, linewidth=linewidth,
                color='black', ecolor='black', capsize=capsize,
                label="stochastic mean"
            )

            # --- std (IC color) ---
            ax.errorbar(
                x, y_std,
                fmt='-o', markersize=8,
                markerfacecolor='white', markeredgecolor=self.colors['IC'],
                markeredgewidth=2, linewidth=linewidth,
                color=self.colors['IC'], ecolor=self.colors['IC'], capsize=capsize,
                label=r"stochastic s.d."
            )

            # --- grey arrow style ---
            arrow_style = dict(
                arrowstyle='-|>,head_width=0.5,head_length=0.8',
                color=self.colors['tILC'], linewidth=2.5
            )

            # --- Arrow positions ---
            if label in [r"$\tau_{on}$", "switching"]:
                target_mu = 0.5
                if np.any(np.isclose(df["mu"].values, target_mu, atol=1e-6)):
                    idx = np.where(np.isclose(df["mu"].values, target_mu, atol=1e-6))[0][0]
                    x_arrow = (1 / df["mu"].iloc[idx]) if label == "switching" else df["mu"].iloc[idx]
                    y_arrow = y_mean[np.argmin(np.abs(x - x_arrow))]
                    ax.annotate(
                        '',
                        xy=(x_arrow, y_arrow - 0.08),
                        xytext=(x_arrow, y_arrow - vertical_offset),
                        arrowprops=arrow_style,
                        annotation_clip=False
                    )

            elif label == "CV":
                cv_target = np.sqrt(3) / 3  # ≈ 0.577
                if np.any(np.isclose(df["cv"].values, cv_target, atol=0.02)):
                    idx = np.where(np.isclose(df["cv"].values, cv_target, atol=0.02))[0][0]
                    x_arrow = df["cv"].iloc[idx]
                    y_arrow = y_mean[idx]
                    ax.annotate(
                        '',
                        xy=(x_arrow, y_arrow - 0.08),
                        xytext=(x_arrow, y_arrow - vertical_offset),
                        arrowprops=arrow_style,
                        annotation_clip=False
                    )

            # --- formatting ---
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

            # Gleichmäßig verteilte Ticks auf der X-Achse (3 pro Subplot)
            xticks = np.linspace(x.min(), x.max(), 3)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{t:.2f}" for t in xticks])
            ax.set_xlabel(x_labels[label], fontsize=14)

            self.format_ax(ax)

            # --- Legend ---
            handles, leg_labels = ax.get_legend_handles_labels()
            ax.legend(handles, leg_labels, frameon=False, loc='upper right', fontsize=10)

            # --- black explanatory text BELOW legend ---
            #text_ypos = 0.70
            #if label == r"$\tau_{on}$":
            #    ax.text(
            #        0.98, text_ypos,
            #        r"fixed std: $\sigma_{on} = \sqrt{3}/6$",
            #        transform=ax.transAxes,
            #        fontsize=10, va='top', ha='right', color='black'
            #    )
            #elif label == "CV":
            #    ax.text(
            #        0.98, text_ypos,
            #        r"fixed mean: $\tau_{on} = 0.5$",
            #        transform=ax.transAxes,
            #        fontsize=10, va='top', ha='right', color='black'
            #    )
            #elif label == "switching":
            #    ax.text(
            #        0.98, text_ypos,
            #        r"fixed std: $\sigma_{on} = \sqrt{3}/6$"'\n'
            #        r"fixed ratio: $\tau_{on}/\tau_{off}=0.5$",
            #        transform=ax.transAxes,
            #        fontsize=10, va='top', ha='right', color='black'
            #    )

        # --- shared y label ---
        axes[0].set_ylabel(r"$t_{\mathrm{stoch}}$ (weeks)", fontsize=14)
        axes[0].set_ylim(0,4)
        axes[0].set_yticks([0, 2, 4])
        axes[2].set_xlim(0.5, 10.5)
        axes[2].set_xticks([1,5.5,10])
        axes[2].set_xticklabels(["1.0", "5.5", "10.0"])


        plt.tight_layout()
        plt.savefig(filename + ".pdf", bbox_inches='tight')
        plt.close()

    # Plots Figure S3

    def figS3A(self, save=True, save_path="histogram_with_kde.svg"):
        from scipy.stats import gaussian_kde

        # Zeitintervalle in Tagen + Häufigkeiten
        intervals_days = [(151, 200), (201, 250), (251, 300), (301, 350), (351, 400)]
        counts = [7, 58, 53, 16, 4]

        # Intervallgrenzen in Wochen
        intervals_weeks = [(start / 7, end / 7) for start, end in intervals_days]
        widths = [end - start for start, end in intervals_weeks]
        midpoints = [(start + end) / 2 for start, end in intervals_weeks]

        # Künstliche Rohdaten für KDE
        data = []
        for (start, end), count in zip(intervals_weeks, counts):
            midpoint = (start + end) / 2
            data.extend([midpoint] * count)
        data = np.array(data)

        # KDE mit glatterer Bandbreite
        kde = gaussian_kde(data, bw_method='scott')
        kde.set_bandwidth(bw_method=kde.factor * 2)

        x_vals = np.linspace(21, 57, 300)
        kde_vals = kde(x_vals) * len(data)  # Skalierung zur Histogrammhöhe

        # Statistiken berechnen
        mean = np.sum(x_vals * kde_vals) / np.sum(kde_vals)
        std = np.sqrt(np.sum((x_vals - mean) ** 2 * kde_vals) / np.sum(kde_vals))

        cdf = np.cumsum(kde_vals)
        cdf /= cdf[-1]
        x_5 = x_vals[np.searchsorted(cdf, 0.05)]
        x_95 = x_vals[np.searchsorted(cdf, 0.95)]

        # Terminalausgabe
        print(f"Mean: {mean:.2f} weeks")
        print(f"Std: {std:.2f} weeks")
        print(f"5th percentile: {x_5:.2f} weeks")
        print(f"95th percentile: {x_95:.2f} weeks")

        # Plot
        fig, ax = plt.subplots(figsize=(3, 2.7))
        for (start, end), count in zip(intervals_weeks, counts):
            ax.bar(x=(start + end) / 2, height=count, width=(end - 0.5 - start),
                   align='center', color='black', edgecolor='black')

        # KDE
        ax2 = ax.twinx()
        ax2.plot(x_vals, kde_vals, color=self.colors['IC'], lw=3, label='KDE (scaled)')

        # Mittelwert- und Std-Linien
        ax2.axvline(mean, color='gray', lw=2, linestyle='-', label='mean')
        ax2.axvline(mean - std, color=self.colors['IC'], lw=1.5, linestyle='--', label='±1 SD')
        ax2.axvline(mean + std, color=self.colors['IC'], lw=1.5, linestyle='--')

        # Achsen
        ax.set_xlabel("time (weeks)")
        ax.set_ylabel("abundance")
        ax2.set_ylabel("density")

        # Tick-Labels
        tick_labels = [f"{int(start)}–{int(end)}" for start, end in intervals_weeks]
        ax.set_xticks(midpoints)
        ax.set_xticklabels(tick_labels, rotation=90)

        # Layout & Format
        plt.tight_layout()
        self.format_ax(ax)
        self.format_ax(ax2)
        ax2.legend(frameon=False, fontsize=8)

        if save:
            plt.savefig(save_path, dpi=300)
            print(f"Plot gespeichert: {save_path}")
        plt.close(fig)

    def figS3C(
            self,
            mu_values, fixed_std, folder_base_path, summary_csv_path,
            bins=50, ignore_outliers=True, hist_folder="histograms",
            ncols=5, xlim=None
    ):
        """
        For each mu in mu_values:
          - collect all t_ESRD values from replicate CSVs
          - (no shifting applied)
          - compute histogram, mean, std
          - save one combined figure with subplots for all histograms
          - save results into one summary CSV

        Parameters
        ----------
        mu_values : list or array
            List of mu values that were simulated.
        fixed_std : float
            Fixed standard deviation used to back-calculate n and k_off.
        folder_base_path : str
            Base path where the replicate folders (mu_*) are stored.
        summary_csv_path : str
            Path to save the summary CSV.
        bins : int
            Number of histogram bins.
        ignore_outliers : bool
            If True, drop top/bottom 1% before computing mean/std.
        hist_folder : str
            Folder to save histogram PDF.
        ncols : int
            Number of subplot columns in the big figure.
        xlim : tuple or None
            Common x-axis limits, e.g. (25, 35). If None → auto.
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        os.makedirs(hist_folder, exist_ok=True)

        summary_records = []
        all_hist_data = {}  # speichert t_ESRD pro mu

        for mu in mu_values:
            folder_name = f"mu_{mu:.2f}".replace(".", "_")
            folder_path = os.path.join(folder_base_path, folder_name)
            k_float = (mu / fixed_std) ** 2
            n = int(np.ceil(k_float))
            k_off = n / mu

            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found for mu={mu:.2f}: {folder_path}")
                continue

            all_t = []
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and fname.startswith("replicate_"):
                    df = pd.read_csv(os.path.join(folder_path, fname))
                    if "t_ESRD" in df.columns:
                        all_t.extend(df["t_ESRD"].dropna().values)

            all_t = np.array(all_t, dtype=float)
            if len(all_t) == 0:
                print(f"⚠️ No t_ESRD values for mu={mu:.2f}")
                summary_records.append({
                    "mu": mu,
                    "k_off": k_off,
                    "mean_t_ESRD": np.nan,
                    "std_t_ESRD": np.nan,
                    "n_values": 0
                })
                continue

            # optional: remove extreme outliers
            if ignore_outliers:
                low, high = np.percentile(all_t, [1, 99])
                all_t = all_t[(all_t >= low) & (all_t <= high)]

            mean_val = float(np.mean(all_t))
            std_val = float(np.std(all_t, ddof=1))

            summary_records.append({
                "mu": mu,
                "k_off": k_off,
                "mean_t_ESRD": mean_val,
                "std_t_ESRD": std_val,
                "n_values": len(all_t)
            })

            print(f"[mu={mu:.2f}] n={len(all_t)}, mean={mean_val:.2f}, std={std_val:.2f}")

            all_hist_data[mu] = (all_t, mean_val, std_val)

        # save summary
        pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
        print(f"Summary saved to {summary_csv_path}")

        # --- gemeinsame Figure mit Subplots ---
        n = len(all_hist_data)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(ncols * 2.2, nrows * 2),
                                 sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)

        for i, (mu, (all_t, mean_val, std_val)) in enumerate(sorted(all_hist_data.items())):
            ax = axes[i]
            bar_color = "black"

            # Histogramm
            ax.hist(all_t, bins=bins, color=bar_color, edgecolor=bar_color, alpha=0.8)

            # Mean- und Std-Linien
            ax.axvline(mean_val, color=self.colors['tILC'], linestyle="-", linewidth=2, label="mean")
            ax.axvline(mean_val - std_val, color=self.colors['IC'], linestyle='--', linewidth=2)
            ax.axvline(mean_val + std_val, color=self.colors['IC'], linestyle='--', linewidth=2)

            # Titel links oben im Subplot
            ax.text(0.02, 0.95, r"$\tau_{on}$ =" f" {mu:.2f}",
                    transform=ax.transAxes,
                    fontsize=11, va="top", ha="left")

            if xlim:
                ax.set_xlim(*xlim)

            # y-label nur links
            if i % ncols == 0:
                ax.set_ylabel("# runs")
            # x-label nur unten
            if i // ncols == nrows - 1:
                ax.set_xlabel(r"$t_{\mathrm{ESRD}}$ (weeks)")

            self.format_ax(ax)

        # überzählige Achsen unsichtbar machen
        for j in range(len(all_hist_data), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(hist_folder, "figS3C.pdf")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Combined histogram figure saved to {out_path}")

    def figS3D(
            self,
            mu_values, fixed_std, folder_base_path, summary_csv_path,
            bins=50, ignore_outliers=True, hist_folder="histograms",
            ncols=5, xlim=None
    ):
        """
        For each mu in mu_values:
          - collect all t_ESRD values from replicate CSVs
          - (no shifting applied)
          - compute histogram, mean, std
          - save one combined figure with subplots for all histograms
          - save results into one summary CSV

        Parameters
        ----------
        mu_values : list or array
            List of mu values that were simulated.
        fixed_std : float
            Fixed standard deviation used to back-calculate n and k_off.
        folder_base_path : str
            Base path where the replicate folders (mu_*) are stored.
        summary_csv_path : str
            Path to save the summary CSV.
        bins : int
            Number of histogram bins.
        ignore_outliers : bool
            If True, drop top/bottom 1% before computing mean/std.
        hist_folder : str
            Folder to save histogram PDF.
        ncols : int
            Number of subplot columns in the big figure.
        xlim : tuple or None
            Common x-axis limits, e.g. (25, 35). If None → auto.
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        os.makedirs(hist_folder, exist_ok=True)

        summary_records = []
        all_hist_data = {}  # speichert t_ESRD pro mu

        for mu in mu_values:
            folder_name = f"cv_{mu:.2f}".replace(".", "_")
            folder_path = os.path.join(folder_base_path, folder_name)
            k_float = (mu / fixed_std) ** 2
            n = int(np.ceil(k_float))
            k_off = n / mu

            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found for mu={mu:.2f}: {folder_path}")
                continue

            all_t = []
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and fname.startswith("replicate_"):
                    df = pd.read_csv(os.path.join(folder_path, fname))
                    if "t_ESRD" in df.columns:
                        all_t.extend(df["t_ESRD"].dropna().values)

            all_t = np.array(all_t, dtype=float)
            if len(all_t) == 0:
                print(f"⚠️ No t_ESRD values for mu={mu:.2f}")
                summary_records.append({
                    "mu": mu,
                    "k_off": k_off,
                    "mean_t_ESRD": np.nan,
                    "std_t_ESRD": np.nan,
                    "n_values": 0
                })
                continue

            # optional: remove extreme outliers
            if ignore_outliers:
                low, high = np.percentile(all_t, [1, 99])
                all_t = all_t[(all_t >= low) & (all_t <= high)]

            mean_val = float(np.mean(all_t))
            std_val = float(np.std(all_t, ddof=1))

            summary_records.append({
                "mu": mu,
                "k_off": k_off,
                "mean_t_ESRD": mean_val,
                "std_t_ESRD": std_val,
                "n_values": len(all_t)
            })

            print(f"[mu={mu:.2f}] n={len(all_t)}, mean={mean_val:.2f}, std={std_val:.2f}")

            all_hist_data[mu] = (all_t, mean_val, std_val)

        # save summary
        pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
        print(f"Summary saved to {summary_csv_path}")

        # --- gemeinsame Figure mit Subplots ---
        n = len(all_hist_data)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(ncols * 2.2, nrows * 2),
                                 sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)

        for i, (mu, (all_t, mean_val, std_val)) in enumerate(sorted(all_hist_data.items())):
            ax = axes[i]

            bar_color = "black"

            # Histogramm
            ax.hist(all_t, bins=bins, color=bar_color, edgecolor=bar_color, alpha=0.8)

            # Mean- und Std-Linien
            ax.axvline(mean_val, color=self.colors['tILC'], linestyle="-", linewidth=2, label="mean")
            ax.axvline(mean_val - std_val, color=self.colors['IC'], linestyle="--", linewidth=2)
            ax.axvline(mean_val + std_val, color=self.colors['IC'], linestyle="--", linewidth=2)

            # Titel links oben im Subplot
            ax.text(0.02, 0.95, r"$\sigma_{on}/\tau_{on}$ =" f" {mu:.2f}",
                    transform=ax.transAxes,
                    fontsize=11, va="top", ha="left")

            if xlim:
                ax.set_xlim(*xlim)

            ax.set_ylim(0,400)

            # y-label nur links
            if i % ncols == 0:
                ax.set_ylabel("# runs")
            # x-label nur unten
            if i // ncols == nrows - 1:
                ax.set_xlabel(r"$t_{\mathrm{ESRD}}$ (weeks)")

            self.format_ax(ax)

        # überzählige Achsen unsichtbar machen
        for j in range(len(all_hist_data), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(hist_folder, "figS3D.pdf")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Combined histogram figure saved to {out_path}")

    def figS3E(
            self,
            mu_values, fixed_std, folder_base_path, summary_csv_path,
            bins=50, ignore_outliers=True, hist_folder="histograms",
            ncols=5, xlim=None
    ):
        """
        For each mu in mu_values:
          - collect all t_ESRD values from replicate CSVs
          - (no shifting applied)
          - compute histogram, mean, std
          - save one combined figure with subplots for all histograms
          - save results into one summary CSV

        Parameters
        ----------
        mu_values : list or array
            List of mu values that were simulated.
        fixed_std : float
            Fixed standard deviation used to back-calculate n and k_off.
        folder_base_path : str
            Base path where the replicate folders (mu_*) are stored.
        summary_csv_path : str
            Path to save the summary CSV.
        bins : int
            Number of histogram bins.
        ignore_outliers : bool
            If True, drop top/bottom 1% before computing mean/std.
        hist_folder : str
            Folder to save histogram PDF.
        ncols : int
            Number of subplot columns in the big figure.
        xlim : tuple or None
            Common x-axis limits, e.g. (25, 35). If None → auto.
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        os.makedirs(hist_folder, exist_ok=True)

        summary_records = []
        all_hist_data = {}  # speichert t_ESRD pro mu

        for mu in mu_values:
            folder_name = f"mu_{mu:.2f}".replace(".", "_")
            folder_path = os.path.join(folder_base_path, folder_name)
            k_float = (mu / fixed_std) ** 2
            n = int(np.ceil(k_float))
            k_off = n / mu

            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found for mu={mu:.2f}: {folder_path}")
                continue

            all_t = []
            for fname in os.listdir(folder_path):
                if fname.endswith(".csv") and fname.startswith("replicate_"):
                    df = pd.read_csv(os.path.join(folder_path, fname))
                    if "t_ESRD" in df.columns:
                        all_t.extend(df["t_ESRD"].dropna().values)

            all_t = np.array(all_t, dtype=float)
            if len(all_t) == 0:
                print(f"⚠️ No t_ESRD values for mu={mu:.2f}")
                summary_records.append({
                    "mu": mu,
                    "k_off": k_off,
                    "mean_t_ESRD": np.nan,
                    "std_t_ESRD": np.nan,
                    "n_values": 0
                })
                continue

            # optional: remove extreme outliers
            if ignore_outliers:
                low, high = np.percentile(all_t, [1, 99])
                all_t = all_t[(all_t >= low) & (all_t <= high)]

            mean_val = float(np.mean(all_t))
            std_val = float(np.std(all_t, ddof=1))

            summary_records.append({
                "mu": mu,
                "k_off": k_off,
                "mean_t_ESRD": mean_val,
                "std_t_ESRD": std_val,
                "n_values": len(all_t)
            })

            print(f"[mu={mu:.2f}] n={len(all_t)}, mean={mean_val:.2f}, std={std_val:.2f}")

            all_hist_data[mu] = (all_t, mean_val, std_val)

        # save summary
        pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
        print(f"Summary saved to {summary_csv_path}")

        # --- gemeinsame Figure mit Subplots ---
        n = len(all_hist_data)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(ncols * 2.2, nrows * 2),
                                 sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)

        # ⬇️ Reihenfolge umgekehrt (vom größten zum kleinsten mu)
        for i, (mu, (all_t, mean_val, std_val)) in enumerate(reversed(sorted(all_hist_data.items()))):
            ax = axes[i]
            bar_color = "black"

            # Histogramm
            ax.hist(all_t, bins=bins, color=bar_color, edgecolor=bar_color, alpha=0.8)

            # Mean- und Std-Linien
            ax.axvline(mean_val, color=self.colors['tILC'], linestyle="-", linewidth=2, label="mean")
            ax.axvline(mean_val - std_val, color=self.colors['IC'], linestyle='--', linewidth=2)
            ax.axvline(mean_val + std_val, color=self.colors['IC'], linestyle='--', linewidth=2)

            # Titel links oben im Subplot
            ax.text(0.02, 0.95, r"$1/\tau_{on}$ =" f" {1/mu:.2f}",
                    transform=ax.transAxes,
                    fontsize=11, va="top", ha="left")

            if xlim:
                ax.set_xlim(*xlim)

            # y-label nur links
            if i % ncols == 0:
                ax.set_ylabel("# runs")
            # x-label nur unten
            if i // ncols == nrows - 1:
                ax.set_xlabel(r"$t_{\mathrm{ESRD}}$ (weeks)")

            self.format_ax(ax)

        # überzählige Achsen unsichtbar machen
        for j in range(len(all_hist_data), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(hist_folder, "figS3E.pdf")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Combined histogram figure saved to {out_path}")
    #############
    # Plots Figure 4

    def fig4B(self, filename):
        """
        Plot normalized kinetics in 2x2 subplots:
        Top row:    tILC (left), MO (right)
        Bottom row: vNK  (left), MOMA (right)

        Each simulation is truncated at the time when Damage reaches 1.
        ESRD points marked; pIC and depletion windows indicated (only in top row).
        """
        print('(Analysis) Generating - 2x2 kinetics (tILC, vNK, MO, MOMA).')

        # Parameter sets (same groups for all plots)
        psets = [
            ('NZB/W', self.params.nzbw_pIC_red),
            ('tILC depletion', self.params.nzbw_ILC_kd),
            ('vNK depletion', self.params.nzbw_NK_kd),
            ('tILC/vNK depl', self.params.nzbw_pIC_kd)
        ]
        colors = ['black', self.colors['tILC'], self.colors['vNK'], 'gray']

        linewidth = 3
        idx_tILC = self.annotation['tILC']
        idx_NK = self.annotation['vNK']
        idx_MO = self.annotation['IFN']
        idx_MOMA = self.annotation['Damage']
        idx_damage = self.annotation['Damage']

        _, _, norm = self.norm_fct_red()

        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5), sharex=True, sharey=True)
        ax_tILC, ax_MO = axes[0]
        ax_vNK, ax_MOMA = axes[1]

        def plot_kinetics(ax, idx_cell, ylabel):
            esrd_times = []
            for (label, pset), color in zip(psets, colors):
                print(f"Simulating {label} for {ylabel}...")
                sim_result = self.model.ode_simulator(pset)
                if sim_result is None:
                    print(f"  ERROR: simulation for {label} returned None!")
                    continue

                sim, _ = sim_result
                y_damage_norm = sim.y[idx_damage] / norm[idx_damage]

                try:
                    i_stop = np.where(y_damage_norm >= 1)[0][0]
                    t_stop = sim.t[i_stop]
                except IndexError:
                    i_stop = -1
                    t_stop = sim.t[-1]
                    print(f"  WARNING: {label} did not reach Damage=1 — using full duration.")

                esrd_times.append(t_stop)

                mask = sim.t <= t_stop
                t_trunc = sim.t[mask]
                y_trunc = sim.y[idx_cell][mask]
                y_norm = y_trunc / norm[idx_cell]

                ax.plot(t_trunc, y_norm, color=color, linewidth=linewidth, label=label)
                ax.plot(t_trunc[-1], y_norm[-1], 'o', color=color, markersize=5)

            ax.set_ylabel(f"{ylabel} (norm)")
            return max(esrd_times) if esrd_times else 0

        tmax_tILC = plot_kinetics(ax_tILC, idx_tILC, "tILC")
        tmax_vNK = plot_kinetics(ax_vNK, idx_NK, "vNK")
        tmax_MO = plot_kinetics(ax_MO, idx_MO, "MO")
        tmax_MOMA = plot_kinetics(ax_MOMA, idx_MOMA, "MOMA")
        t_max = max(tmax_tILC, tmax_vNK, tmax_MO, tmax_MOMA) + 1

        # Treatment markers only in the top row
        pic_start = 19
        pic_end = self.params.wt_pIC.t_end_trigger

        for ax in [ax_tILC, ax_MO]:
            ax.axvspan(pic_start, pic_end, ymin=1.18, ymax=1.2, color='gray', alpha=0.8, clip_on=False)
            ax.text((pic_start + pic_end) / 2, 1.22, 'pIC', ha='center',
                    transform=ax.get_xaxis_transform(), fontsize=10, color='gray')

            ax.axvspan(19, 25, ymin=1.02, ymax=1.035, color='black', alpha=0.8, clip_on=False)
            ax.text(22, 1.06, 'depletion', ha='center',
                    transform=ax.get_xaxis_transform(), fontsize=10, color='black')

        # Formatting all
        for ax in [ax_tILC, ax_vNK, ax_MO, ax_MOMA]:
            ax.set_xlim((17, max(t_max, 30)))
            ax.set_ylim(bottom=-0.1)
            self.format_ax(ax)

        ax_vNK.set_xlabel("time (weeks)")
        ax_MOMA.set_xlabel("time (weeks)")

        # Legends only once (top row)
        for ax in [ax_tILC, ax_MO]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='lower right', fontsize=10, frameon=False)

        plt.tight_layout(h_pad=1.2, w_pad=1.2)
        plt.subplots_adjust(hspace=0.25, wspace=0.25)  # tighter spacing
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def fig4C(self, var, filename, runs):
        """
        Robustness analysis: Barplot of normalized cell counts (mean ± std)
        at evaluation timepoint for NZB/W and three knockdown conditions.
        """
        print('(Analysis) Generating Fig3C - Robustness barplot for 4 parameter sets')

        colors = ['black', self.colors['tILC'], self.colors['vNK'], 'grey']
        t_eval = 25  # weeks

        all_data = []

        for i in range(runs):
            # --- 1. Sample baseline (NZB/W) ---
            sampled_base = copy.deepcopy(self.params.nzbw_pIC_red)
            pvars = [k for k in vars(sampled_base) if
                     not k.startswith('t_') and k not in
                     ['nk', 'k', 'k2', 'k3', 'name', 'n', 'y0', 'disruption', 'r', 'depl', 'r_nk']]
            for vname in pvars:
                val = getattr(sampled_base, vname)
                setattr(sampled_base, vname, self.sample_lognormal(val, rel_std=0.1))

            # --- 2. Normalization from sampled baseline ---
            dmg_95, dmg_t95, norm = self.norm_fct_red(paramset=sampled_base)

            # --- 3. Build KD parameter sets based on sampled baseline ---
            nzbw_ILC_kd = copy.deepcopy(sampled_base)
            nzbw_ILC_kd.r = 0.8
            nzbw_ILC_kd.t_start_kd = 19
            nzbw_ILC_kd.t_end_kd = 25
            nzbw_ILC_kd.depl = 0.2

            nzbw_NK_kd = copy.deepcopy(sampled_base)
            nzbw_NK_kd.r_nk = 0.8
            nzbw_NK_kd.t_start_kd = 19
            nzbw_NK_kd.t_end_kd = 25
            nzbw_NK_kd.depl = 0.2

            nzbw_pIC_kd = copy.deepcopy(sampled_base)
            nzbw_pIC_kd.r = 0.8
            nzbw_pIC_kd.r_nk = 0.8
            nzbw_pIC_kd.t_start_kd = 19
            nzbw_pIC_kd.t_end_kd = 25
            nzbw_pIC_kd.depl = 0.2

            # --- 4. Collect all conditions ---
            psets = [
                ('NZB/W', sampled_base),
                ('tILC depl', nzbw_ILC_kd),
                ('vNK depl', nzbw_NK_kd),
                ('tILC/vNK depl', nzbw_pIC_kd)
            ]

            # --- 5. Simulate all conditions ---
            for label, tmp_pset in psets:
                sim_result = self.model.ode_simulator(tmp_pset)
                if sim_result is None:
                    print(f"    [WARN] Run {i} skipped for {label} (None result)")
                    continue

                sim, _ = sim_result
                idx_eval = min(np.searchsorted(sim.t, t_eval), len(sim.t) - 1)

                for v in var:
                    vi = self.annotation[v]
                    count = sim.y[vi][idx_eval] / norm[vi]
                    all_data.append({
                        'Condition': label,
                        'Variable': v,
                        'Run': i,
                        'Value': count
                    })

        # --- 6. Summarize results ---
        df = pd.DataFrame(all_data)
        conditions_order = ['NZB/W', 'tILC depl', 'vNK depl', 'tILC/vNK depl']
        df["Condition"] = pd.Categorical(df["Condition"], categories=conditions_order, ordered=True)

        summary = df.groupby(["Condition", "Variable"]).agg(
            Mean=("Value", "mean"),
            STD=("Value", "std")
        ).reset_index()

        # --- 7. Plot ---
        fig, ax = plt.subplots(figsize=(5.5, 2.5))
        sns.barplot(
            data=summary,
            x="Variable",
            y="Mean",
            hue="Condition",
            palette=colors,
            errorbar=None,
            edgecolor='white',
            linewidth=2,
            ax=ax,
            dodge=True,
            width=0.8
        )

        # Add error bars
        for i, row in summary.iterrows():
            x = list(summary["Variable"].unique()).index(row["Variable"])
            cond_idx = conditions_order.index(row["Condition"])
            x_pos = x + (-0.3 + 0.2 * cond_idx)  # mimic seaborn dodge
            ax.errorbar(
                x=x_pos,
                y=row["Mean"],
                yerr=row["STD"],
                fmt='none',
                ecolor='black',
                elinewidth=1,
                capsize=3,
                zorder=5
            )

        self.format_ax(ax)
        ax.set_ylabel("cells (norm)")
        ax.set_ylim(bottom=0, top=1.5)
        ax.set_xlabel("")
        ax.legend(title=None, loc='upper right', frameon=False, fontsize=9, ncol=4)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def fig4D(self, filename):
        """
        Plot normalized kinetics for MO (left) and MOMA (right) in a single row (1x2).

        Each simulation is truncated at the time when Damage reaches 1.
        ESRD points marked; pIC and depletion windows indicated.
        """
        print('(Analysis) Generating - 1x2 kinetics (MO, MOMA).')

        # Parameter sets (same groups for all plots)
        psets = [
            ('NZB/W', self.params.nzbw_pIC_red),
            ('tILC depletion', self.params.nzbw_ILC_kd),
            ('vNK depletion', self.params.nzbw_NK_kd),
            ('tILC/vNK depl', self.params.nzbw_pIC_kd)
        ]
        colors = ['black', self.colors['tILC'], self.colors['vNK'], 'gray']

        linewidth = 3
        idx_MO = self.annotation['MO']
        idx_MOMA = self.annotation['MOMA']
        idx_damage = self.annotation['Damage']

        _, _, norm = self.norm_fct_red()

        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharex=True, sharey=True)
        ax_MO, ax_MOMA = axes

        def safe_norm_value(idx, sim):
            """Return safe normalization value for a variable."""
            try:
                val = norm[idx]
            except Exception:
                val = None
            if val is None or not np.isfinite(val) or val <= 0:
                sim_max = np.nanmax(sim.y[idx]) if sim is not None else 0.0
                if np.isfinite(sim_max) and sim_max > 0:
                    val = sim_max
                    print(f"  NOTE: norm for idx {idx} invalid — using sim max={sim_max:.3g} for normalization.")
                else:
                    val = 1.0
                    print(f"  WARNING: norm and sim max for idx {idx} are invalid — falling back to 1.0.")
            return val

        def plot_kinetics(ax, idx_cell, ylabel):
            esrd_times = []
            for (label, pset), color in zip(psets, colors):
                print(f"Simulating {label} for {ylabel}...")
                sim_result = self.model.ode_simulator(pset)
                if sim_result is None:
                    print(f"  ERROR: simulation for {label} returned None!")
                    continue

                sim, _ = sim_result
                # normalize Damage for truncation
                damage_norm_val = safe_norm_value(idx_damage, sim)
                y_damage_norm = sim.y[idx_damage] / damage_norm_val

                try:
                    i_stop = np.where(y_damage_norm >= 1)[0][0]
                    t_stop = sim.t[i_stop]
                except IndexError:
                    i_stop = -1
                    t_stop = sim.t[-1]
                    print(f"  WARNING: {label} did not reach Damage=1 — using full duration.")

                esrd_times.append(t_stop)

                mask = sim.t <= t_stop
                t_trunc = sim.t[mask]
                y_trunc = sim.y[idx_cell][mask]

                norm_val = safe_norm_value(idx_cell, sim)
                y_norm = y_trunc / norm_val

                ax.plot(t_trunc, y_norm, color=color, linewidth=linewidth, label=label)
                ax.plot(t_trunc[-1], y_norm[-1], 'o', color=color, markersize=5)

            ax.set_ylabel(f"{ylabel} (norm)")
            return max(esrd_times) if esrd_times else 0

        # --- plot both panels ---
        tmax_MO = plot_kinetics(ax_MO, idx_MO, "MO")
        tmax_MOMA = plot_kinetics(ax_MOMA, idx_MOMA, "MOMA")

        t_max = max(tmax_MO, tmax_MOMA) + 1

        # Treatment markers for both panels
        pic_start = 19
        pic_end = self.params.wt_pIC.t_end_trigger

        for ax in [ax_MO, ax_MOMA]:
            ax.axvspan(pic_start, pic_end, ymin=1.18, ymax=1.2, color='gray', alpha=0.8, clip_on=False)
            ax.text((pic_start + pic_end) / 2, 1.22, 'pIC', ha='center',
                    transform=ax.get_xaxis_transform(), fontsize=10, color='gray')

            ax.axvspan(19, 25, ymin=1.02, ymax=1.035, color='black', alpha=0.8, clip_on=False)
            ax.text(22, 1.06, 'depletion', ha='center',
                    transform=ax.get_xaxis_transform(), fontsize=10, color='black')

        # Formatting
        for ax in [ax_MO, ax_MOMA]:
            ax.set_xlim((17, max(t_max, 30)))
            ax.set_ylim(bottom=-0.1)
            self.format_ax(ax)
            ax.set_xlabel("time (weeks)")

        # Legend (only left panel)
        handles, labels = ax_MO.get_legend_handles_labels()
        if handles:
            ax_MO.legend(handles, labels, loc='lower right', fontsize=10, frameon=False)

        plt.tight_layout(w_pad=1.2)
        plt.subplots_adjust(wspace=0.25)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def fig4E(self, data, name):
        """
        Plot knockdown strength effect for all KD types, with arrow at 80% depletion.
        """
        print('(LupusAnalysis) Generate knockdown strength figure.')
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        sns.set_style("ticks")
        palette = {
            'NZB/W': 'black',
            'tILC depl': self.colors['tILC'],
            'vNK depl': self.colors['vNK'],
            'tILC/vNK depl': 'gray'
        }

        # Plot each line separately (to draw custom arrows after plotting)
        for condition, df_cond in data.groupby("condition"):
            sns.lineplot(data=df_cond, x='fac', y='delta_weeks',
                         label=condition, color=palette[condition],
                         linewidth=3, ax=ax)

        # Add red arrow pointing to fac = 0.8 and y = 0.8 on y-axis
        arrow_y_top = ax.get_ylim()[0] + 5  # oberer Rand
        arrow_y_bottom = ax.get_ylim()[0] -1.5 # etwas oberhalb der x-Achse
        ax.annotate('', xy=(0.8, arrow_y_bottom), xytext=(0.8, arrow_y_top - 3),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.6,head_length=1.0', color='black', linewidth=3),
                    annotation_clip=False)

        self.format_ax(ax)
        ax.set_xlabel(r'depletion strength (%)', fontsize=14)
        ax.set_ylabel(r'$t_{ESRD}$ (weeks)', fontsize=14)
        ax.set_ylim(30, 38)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([0, 50, 100])
        ax.legend(title=None, frameon=False, fontsize=9)

        ax.yaxis.set_major_locator(LinearLocator(3))

        plt.tight_layout()
        plt.savefig(name, dpi=800, bbox_inches='tight')
        plt.close()

    def fig4F(self, dfs, labels, out_path):
        """
        Plot timing screens as a 3-panel heatmap (one for each depletion condition),
        with equal subplot sizes and a shared horizontal colorbar below the plots.
        """
        sns.set_style("ticks")
        from matplotlib import gridspec

        # Setup figure with 2 rows: first row for heatmaps, second row for colorbar
        fig = plt.figure(figsize=(10, 4.5))  # adjust height for compact layout
        gs = gridspec.GridSpec(2, 3, height_ratios=[20, 1], hspace=0.3, wspace=0.3)

        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

        vmin = min(df['t50'].min() for df in dfs)
        vmax = max(df['t50'].max() for df in dfs)

        im = None
        for ax, df, title in zip(axes, dfs, labels):
            df = df[(df['start'] <= 28) & (df['t_treatment'] <= 15)]
            heat_data = df.pivot(index='start', columns='t_treatment', values='t50')

            im = ax.imshow(
                heat_data,
                cmap='Greys',
                origin='lower',
                aspect='equal',
                vmin=vmin,
                vmax=40
            )

            ax.set_title(title)
            self.format_ax(ax)
            ax.set_xlim(0, 10)
            ax.set_ylim(19, 27)
            ax.set_xlabel('duration (weeks)')
            if ax == axes[0]:
                ax.set_ylabel('onset time (weeks)')

        # Horizontal colorbar in the second row spanning all columns
        cax = fig.add_subplot(gs[1, :])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(r'$t_{ESRD}$ (weeks)')
        cbar.outline.set_color('gray')

        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    # Plots Figure S4

    def figS4(self, filename: str):
        """
        Zeigt normalisierte Zellkinetiken für NZB/W + Knockdown-Bedingungen.
        ESRD-Zeitpunkte werden als Marker eingezeichnet und die Kinetiken dort abgeschnitten.
        """

        print("(Analysis) Generating NZB/W Knockdown comparison plot.")

        # Modellzustände, die geplottet werden sollen
        plot_vars = [
            'C34', 'C43', 'C67', 'C76'
        ]

        # Parameter-Sets und Farben
        psets = [
            ('NZB/W', self.params.nzbw_pIC_red),
            ('tILC depl', self.params.nzbw_ILC_kd),
            ('vNK depl', self.params.nzbw_NK_kd),
            ('tILC/vNK depl', self.params.nzbw_pIC_kd)
        ]
        colors = ['black', self.colors['tILC'], self.colors['vNK'], 'gray']
        linewidth = 2.5

        # Normierung
        _, _, norm = self.norm_fct_red()

        # Subplot-Größe
        n = len(plot_vars)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 3.5, nrows * 2.3),
            sharex=True,
            gridspec_kw={'hspace': 0.3, 'wspace': 0.4}
        )
        axes = axes.flatten()

        for idx, var in enumerate(plot_vars):
            ax = axes[idx]
            var_idx = self.annotation[var]

            for (label, pset), color in zip(psets, colors):
                sim, cf = self.model.ode_simulator(pset)
                y = cf if var == 'IFN' else sim.y[var_idx]
                y_norm = y / norm[var_idx]

                # ESRD-Schnittpunkt (via Damage)
                y_damage = sim.y[self.annotation['Damage']] / norm[self.annotation['Damage']]
                try:
                    i_esrd = np.where(y_damage >= 1)[0][0]
                    t_esrd = sim.t[i_esrd]
                except IndexError:
                    i_esrd = -1
                    t_esrd = sim.t[-1]

                mask = sim.t <= t_esrd
                t_trunc = sim.t[mask]
                y_trunc = y_norm[mask]

                ax.plot(t_trunc, y_trunc, color=color, linewidth=linewidth, label=label)
                ax.plot(t_trunc[-1], y_trunc[-1], 'o', color=color, markersize=5)

            ax.set_ylabel(f"{var} (norm)")
            ax.set_xlim(18.5, 41.5)
            ax.set_ylim(bottom=-0.05)
            if var in ['C34', 'C67']:
                ax.set_ylim(-0.1,2.05)

            if var in ['C34']:
                ax.set_ylabel('C1 (norm)')
            if var in ['C43']:
                ax.set_ylabel('C2 (norm)')
            if var in ['C67']:
                ax.set_ylabel('C3 (norm)')
            if var in ['C76']:
                ax.set_ylabel('C4 (norm)')
            self.format_ax(ax)

            # Legende nur, wenn Platz (z. B. in letztem Plot)
            if idx == len(plot_vars) - 1:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, frameon=False, fontsize=11)

        # pIC-Markierungen (über erste vier Achsen)
        pic_start = self.params.nzbw_pIC_red.t_start_trigger
        pic_end = 25
        for ax in axes[:4]:
            ax.axvspan(pic_start, pic_end, ymin=1.04, ymax=1.06, color='gray', alpha=0.8, clip_on=False)
            ax.text((pic_start + pic_end) / 2, 1.11, 'pIC', ha='center', transform=ax.get_xaxis_transform(),
                    fontsize=11, color='gray')

        # x-Achsenbeschriftung für untere Reihe
        for ax in axes[-ncols:]:
            ax.set_xlabel("time (weeks)")

        # Leere Subplots ausblenden
        for ax in axes[n:]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

