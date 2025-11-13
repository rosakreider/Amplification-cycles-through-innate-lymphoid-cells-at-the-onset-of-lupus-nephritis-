from lupus_models import LupusIncIC
from lupus_params import Params
from lupus_analysis import Analysis

import numpy as np

params = Params()
model = LupusIncIC()

analysis = Analysis(model, params)

#############
# Figure 1
data_path = '/home/rosakreider/Desktop/data.csv'
#analysis.fig1B(data_path)

#############
# Figure 2
#analysis.fig2C(filename='results/subfigures/fig2C.pdf')
#data_fig2E = analysis.network_disruption_screen()
#analysis.fig2E(data_fig2E, 'results/subfigures/fig2E.pdf')
#data_fig2F = analysis.sens_analysis()
#analysis.fig2F(data_fig2F, "results/subfigures/fig2F.pdf")

# Figure S2
#analysis.figS2A(['tILC', 'vNK'], 'results/subfigures/figS2A.pdf')
#analysis.figS2BC(filename='results/subfigures/figS2')
#analysis.figS2D(data_fig2E, 'results/subfigures/figS2D.pdf')
#data_figS2E = analysis.sens_analysis_2()
#analysis.fig2F(data_figS2E, "results/subfigures/figS2E.pdf")

#############
# Figure 3 + S3
#analysis.fig3B(name='results/subfigures/fig3B')
#analysis.fig3C(runs=3, name='results/subfigures/fig3C')
#analysis.fig3D(runs=100, name='results/subfigures/fig3D.pdf')
#analysis.fig3G('results/subfigures/fig3G.pdf')
#analysis.figS3A(save_path='results/subfigures/figS3A.pdf')
mu_values = np.linspace(0.1, 1.0, 10)  # Beispielwerte
fixed_std = np.sqrt(3)/6
csv_path = "results/esrd_mu_scan_2.csv"
plot_path = "results/esrd_mu_plot.pdf"
#analysis.run_heterogeneity_analysis_mu(
#    mu_values=mu_values,
#    fixed_std=fixed_std,
#    runs_per_file=100,
#    n_repeats=100,
#    folder_base_path="./data/heterogeneity_mu",
#    summary_csv_path="./data/heterogeneity_mu_summary.csv"
#)

#analysis.summarize_mu_scan(
#    mu_values=mu_values,
#    fixed_std=fixed_std,
#    folder_base_path="./data/heterogeneity_mu",
#    summary_csv_path="./data/heterogeneity_mu_shifted_summary.csv",
#    bins=60
#)

#analysis.figS3C(
#    mu_values=mu_values,
#    fixed_std=fixed_std,
#    folder_base_path="./data/heterogeneity_mu",
#    summary_csv_path="./data/heterogeneity_mu_summary_histo.csv",
#    bins=60,
#    xlim=(25,35)
#)

#analysis.fig3F(
#    mu_values=[0.5],
#    fixed_std=fixed_std,
#    folder_base_path="./data/heterogeneity_mu",
#    summary_csv_path="./data/heterogeneity_mu_summary_histo.csv",
#    bins=60
#)


fixed_mu = 0.5                      # z. B. mittlere Verweildauer = 10 Wochen
runs = 1000                           # Anzahl Simulationsläufe pro Konfiguration
csv_path = "results/esrd_by_cv_new.csv"  # Speicherpfad
cv_vals = [0.5,  0.56, 0.61, 0.67, 0.73, 0.78, 0.83, 0.89,  0.95, 1.]

#analysis.run_heterogeneity_analysis_cv(
#    cv_values=cv_values,
#    fixed_mu=fixed_mu,
#    runs_per_file=100,
#    n_repeats=100,
#    folder_base_path="./data/heterogeneity_cv_new",
#    summary_csv_path="./data/heterogeneity_cv_summary_new.csv"
#)

#analysis.summarize_cv_scan(
#    cv_values=cv_vals,
#    folder_base_path="./data/heterogeneity_cv",
#    summary_csv_path="./data/heterogeneity_cv_shifted_summary.csv",
#    bins=60
#)

#analysis.figS3D(
#    mu_values=cv_vals,
#    fixed_std=fixed_std,
#    folder_base_path="./data/heterogeneity_cv",
#    summary_csv_path="./data/heterogeneity_cv_summary_histo.csv",
#    bins=60,
#    xlim=(25, 35)
#)

mu_values = [0.8, 0.5, 0.271, 0.19,  0.155, 0.135, 0.122, 0.113, 0.1]
  # Beispielwerte
csv_path = "results/esrd_by_mu_with_kon.csv"
plot_path = "results/esrd_switching_plot.pdf"
fixed_std = np.sqrt(3)/6

#analysis.run_heterogeneity_analysis_mu_fixed_ratio(
#    mu_values=mu_values,
#    fixed_std=fixed_std,
#    runs_per_file=100,
#    n_repeats=100,
#    folder_base_path="./data/heterogeneity_mu_ratio_new",
#    summary_csv_path="./data/heterogeneity_mu_ratio_summary_new.csv"
#)

#analysis.summarize_mu_fixed_ratio_scan(
#    mu_values=mu_values,
#    folder_base_path="./data/heterogeneity_mu_ratio_new",
#    summary_csv_path="./data/heterogeneity_mu_ratio_shifted_summary.csv",
#    bins=60
#)

#analysis.figS3E(
#    mu_values=mu_values,
#    fixed_std=fixed_std,
#    folder_base_path="./data/heterogeneity_mu_ratio_new",
#    summary_csv_path="./data/heterogeneity_mu_ratio_summary_histo.csv",
#    bins=60,
#    xlim=(25, 35)
#)

#analysis.fig3E()

#############
# Figure 4
#analysis.fig4B('results/subfigures/fig4B.pdf')
#analysis.fig4C(['tILC', 'vNK', 'MOMA', 'MO'], 'results/subfigures/fig4C.pdf', 1000)
#analysis.fig4D('results/subfigures/fig4D.pdf')
#data_fig4E = analysis.knockdown_analysis()
#analysis.fig4E(data_fig4E, 'results/subfigures/fig4E.pdf')
#data_fig4F = analysis.screen_nkp46_timing_multi(param_list=[
#        analysis.params.nzbw_ILC_kd,
#        analysis.params.nzbw_NK_kd,
#        analysis.params.nzbw_pIC_kd],
#        labels=['tILC depl', 'vNK depl', 'tILCvNK depl'])
#data_fig4F = analysis.load_nkp46_timing_from_csv(
#    labels=['tILC depl', 'vNK depl', 'tILCvNK depl'])
#analysis.fig4F(data_fig4F, ['tILC depl', 'vNK depl', 'tILCvNK depl'], 'results/subfigures/fig4F.pdf')

# Figure S4
#analysis.figS4('results/subfigures/figS4.pdf')
