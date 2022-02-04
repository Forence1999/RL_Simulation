#!/bin/bash

#for i in {1..5}
#do
#	python ./code/RL_Simulation.py
#done

#python ./RL_Simulation.py --agent_learn False --base_model_dir '../model/base_model' --based_on_base_model True --min_eps 0.05

python ./RL_Simulation.py --agent_learn False --base_model_dir '../model/all_model' --based_on_base_model True --min_eps 0.0
