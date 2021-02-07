# OptiNet

#### Note:
The datasets and models are stored at: https://mega.nz/#F!1VkxDSgT. Unfortunetely, I do not have permission to share this data and can therefore not be publicly provided. The original dataset from Duke university can be found at: http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm

The refined datasets are also stored in the link given above, therefore the following steps in 'On a regular computer' can be skipped: 8-12. I leave the steps in documentation in case anyone would need to set up the project from scratch.

## How to run the project:
A version of **Python 3** has to be installed.
#### On a regular computer:
Note: Virtualenv is not necessary, it is only a contained environment you can run python code in. You can ignore step 3-5 and 21 if you do not want to use virtualenv.
1. Create a folder called _datasets_ in the project folder (_AMD_Detection_)
2. Download and extract _duke_ and/or _st_olavs_ dataset into _datasets_
3. Create a folder in the project folder called _.env_
4. Create a virtual environment: `virtualenv .env/neural_nets`
5. Start virtual environment: `source .env/neural_nets/bin/activate`
6. Install packages: `pip install -r requirements.txt`
7. Go to refiner folder: `cd dataset_refiners`
7. To refine _duke_ dataset: `python mat_dataset_refiner.py` | To refine _st_olavs_ dataset: `python patient_dataset_refiner.py`
8. For the _duke_ dataset, go to folder _duke_refined_: `cd ../datasets/duke_refined`
8. Create folders _val_ and _train_ in the newly generated _duke_refined_ folder.
9. Select files to be used for training (moved into _train_ folder) and files to be used for validation (moved into _val_ folder)
10. Exit the dataset folders: `cd ../..`
10. Adjust hyper-parameters by editing variables in the top of _main.py_
10. Run training: `python .\main.py`
11. To monitor performance, run the following command in a new terminal: `tensorboard --logdir=.../AMD_Detection/summaries` and put the url given into a browser. (If the code is on a server, use something like sshfs to get the files locally and run this command on your local computer. You will have to install ternsorboard for the computer running the command, which can be done by `pip install tensorboard`)
12. Once you are satisfied with the results or need to change something, just terminate the code. It is set to run for an extremely long time, if a single model is trained. However, if multiple models are trained, then it will use the timelimit in the hyperparameters for each model.
13. Edit parameters to fit your needs in _evaluation.py_.
13. Run evaluation of one or multiple models: `python evaluation.py`
14. Edit parameters to your needs in _visualizer.py_.
14. Run visualization on one model: `python visualizer.py`
14. Close environment: `deactivate`

Once these steps has been followed once, you only need to do steps 5, 7 and 13-21. Variables in _main.py_, _evaluation.py_, _visualizer.py_ must be changed for the code to run on a different dataset or with different hyperparameters. _evaluation.py_ and _visualizer.py_ used on a spesific model requires the same hyperparameters as that model was trained on. The hyperparameters for a model are stored in 'saved_models/[Model name]/hyper_params.json' (can be also printed when running _evaluation.py_). To make a named folder for multiple experiments run in one session, add a '.' at the end of the name. If one wish to evaluate multiple experiments also add the '.' at the end of the folder name.

#### On a cluster using SLURM:
Follow the steps exactly the same as in "On a regular computer" (remember to load a python3 module first), but switch out the following steps:

(8) Go to scripts: `cd scripts` \
(8.5) Refine a dataset: `sbatch data.slurm [duke OR st_olavs]`\
(10) Run training (one can adjust total timelimit inside _job.slurm_): `sbatch job.slurm` \
(12) To terminate code `scancel [Job ID]` or to terminate all your jobs `scancel -u [Your username]` \
(13) Evaluate a model: `sbatch eval.slurm`

Once the steps have been followed once you only need to follow steps 10-13 for training and evaluation.
Be aware that _data.slurm_, _job.slurm_, _eval.slurm_ have settings set for a specific cluster used by me, so these files might need to be adjusted when used on another cluster. Clusters differ in which modules they have, however they should be fairly similar to the ones seen in _data.slurm_, _job.slurm_ and _eval.slurm_. Furthermore, _visualizer.py_ does not work for the slurm setup. Extra note: _run_experiment.sh_ was a script spesifically made for a server on UTokyo, this file can be ignored.

## Troubleshoot notes:
- Problems running: `pip install -r requirements.txt`
  - PyTorch installs differently depending on a few factors such as OS (find CUDA version with `nvcc --version`)! Visit https://pytorch.org/ to check which command you need to use. Once you have manually installed PyTorch (after step 5 if you run virtualenv), try the command again.
  - Virtual environment does not allow you to pick Python versions you do not have installed. Your python version may be the issue. Python 3.6.6 has been tested and works. Try setting up the environment with this Python version.
