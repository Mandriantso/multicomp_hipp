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

### Run
To run the simulation in parallel, run:
```
mpiexec -n N python param_search.py
```
in terminal, with N being the number of processors to use