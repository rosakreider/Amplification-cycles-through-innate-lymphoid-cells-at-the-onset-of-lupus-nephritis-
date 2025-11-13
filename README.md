# Amplification cycles through innate lymphoid cells at the onset of lupus nephritis

## Overview
This project investigates amplification cycles of innate lymphoid cells (ILCs) during the onset of lupus nephritis, combining **data analysis** and **mathematical modelling**.  
It provides reproducible simulation and plotting tools to study disease dynamics and immune interactions.

The repository consists of two main parts:
- **Data analysis** — scripts and notebooks for exploring experimental and simulation data  
- **Mathematical modelling** — core model and simulation framework (main entry: `lupus_plots.py`)

---

## Repository structure

    ├── data_analysis/ # Scripts and notebooks for data analysis

        ├── DA_Figure_1.yml # Conda environment file 
        ├── DA_ILC_Figure_1_pipeline_gitlab.Rmd # data analysis notebook 

    ├── mathematical_models/ # Scripts and notebooks for mathematical modelling

        ├── lupus_models.py # Model definitions (ODEs, stochastic processes, etc.)

        ├── lupus_params.py # Parameter sets used in simulations

        ├── lupus_analysis.py # Analysis and plotting functions

        ├── gillespie_simulator.py # Stochastic simulation implementation (Gillespie algorithm)

        ├── acute_inflammation_model.py # Supporting model for inflammation dynamics

        ├── lupus_plots.py # Main script to run simulations and generate figures

        ├── lupus.yml # Conda environment file 

    └── README.md # Project description


---

## Installation - Using Conda

```bash
# Clone the repository
git clone https://github.com/rosakreider/Amplification-cycles-through-innate-lymphoid-cells-at-the-onset-of-lupus-nephritis-.git
cd Amplification-cycles-through-innate-lymphoid-cells-at-the-onset-of-lupus-nephritis-

# Create and activate the environment
conda env create -f environment.yml
conda activate lupus_env
```
## Usage
### Run the main simulation
```bash
python lupus_plots.py
```

This will execute the main simulation workflow defined in lupus_plots.py, using the model structure and parameters from:

lupus_models.py

lupus_params.py

gillespie_simulator.py

The program outputs simulation results and generates corresponding plots and figures.

### Run data analysis

Scripts in the data_analysis/ folder perform statistical or exploratory analysis on data produced by the simulations or external datasets:
```bash
python data_analysis/your_script.py
```

## Links

[Add publication link]