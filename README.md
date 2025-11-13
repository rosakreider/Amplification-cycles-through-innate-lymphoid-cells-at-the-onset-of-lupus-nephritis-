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

### Step 1: Clone repository

```bash
# Clone the repository
git clone https://github.com/rosakreider/Amplification-cycles-through-innate-lymphoid-cells-at-the-onset-of-lupus-nephritis-.git
cd Amplification-cycles-through-innate-lymphoid-cells-at-the-onset-of-lupus-nephritis-
```

### Step 2: Install data analysis
```bash
...
```


### Step 3: Install modelling analysis

```bash
# Create and activate the modelling environment
conda env create -f lupus.yml
conda activate lupus
```
## Usage
### Step 1: Run data analysis

Scripts in the data_analysis/ folder perform statistical or exploratory analysis on data produced by the simulations or external datasets:
```bash
 ...
```
### Step 2: Run modelling analysis
```bash
python lupus_plots.py
```

The program outputs simulation results and generates corresponding plots and figures.

## Links

[Add publication link]