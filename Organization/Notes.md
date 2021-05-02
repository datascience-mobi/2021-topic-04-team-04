## create conda environment

1. create file environment.yml in project folder of project you are working on
1. define name and dependencies of environment in environment.yml file: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-an-environment-file-across-platforms
1. commit
1. create and activate new environment https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
   1. conda env create -f environment.yml (execute on terminal)
   1. conda activate environmentGroup4
   1. conda env list (to test)
1. update environment: change environment.yml file, commit, update and activate https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment
   1. conda env update --prefix ./env --file environment.yml  --prune
1. push, others can pull now environment.yml file and continue in the same way

## Create Python package
- https://www.tutorialsteacher.com/python/python-package 