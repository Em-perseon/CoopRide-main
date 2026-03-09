# CoopRide
This is the code for paper "CoopRide: Cooperate All Grids in City-Scale Ride-Hailing Dispatching with Multi-Agent Reinforcement Learning"


## Installation and Setups 
``` python
conda create -n CoTa python=3.9
conda activate CoTa
pip install -r requirements.txt
```

## Run Experiments
* Command to run our method

``` python
cd run
python run_CoTa.py
```

* Command for baselines

```python
cd run
python run_MAPPO.py
python run_IPPO.py
python run_mdp.py
```

## Visualizations
* You can visualize the learning curves by tensorboard.
``` python
tensorboard --logdir logs
```

* You can visualize the cooperation intensity, city map and demand-supply heat map by following jupyters.
``` python
plot/plot_grid.ipynb
plot/grid_map.ipynb
plot/plot_meta.ipynb
```


cd coopride_llm
python main.py --config configs/coopride_config.yaml

set OPENAI_API_KEY=your_api_key_here
set OPENAI_BASE_URL=https://api.chatanywhere.tech/v1