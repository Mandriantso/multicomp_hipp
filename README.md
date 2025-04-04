# multicomp_hipp
Multicompartment model of hippocampal formation


### Setup
Simulations are run using NEURON 8.2.2 https://nrn.readthedocs.io/en/8.2.2/index.html with python version 3.10.9
The virtual environment can be created from the requirements.txt file by running
```
pip install -r requirements.txt
```

Then, all the mod files in the Mods folder must be compiled by running
```
mknrndll
```
in terminal after cding to the Mods folder. A nrnmech.dll file will be created. 

### Parameters files
Parameters files are stored in configs folder.
- **default_parameters_1.json** is the file used for connection weights optimization. The connection parameters were set from Nikos' model, adapted to my network size, and increased until we were able to see enough activity in all populations

- **default_parameters_2.json** is the file used with updated connection weights for Pyramidal and Basket cells

- **default_parameters_2_stim.json** is the file used with updated connection weights for Pyramidal and Basket cells, adding extracellular stimulation

### Simulation files
- **run_simulation.py** main code to run a single simulation
- **param_search.py** code for grid search on connection weights

### Run
To run the simulation in parallel, run:
```
mpiexec -n N python param_search.py
```
in terminal, with N being the number of processors to use