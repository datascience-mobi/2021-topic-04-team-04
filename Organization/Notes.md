## Create conda environment
1. create file environment.yml in project folder of project you are working on
1. define name and dependencies of environment in environment.yml file: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-an-environment-file-across-platforms
1. create and activate new environment https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
   1. conda env create -f environment.yml (execute on terminal)
   1. conda activate environmentGroup4
   1. conda env list (to test)
1. update environment without history
    - conda env export --from-history > environment.yml
1. update environment from environment.yml file https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment
   1. conda env update --file environment.yml  --prune

## Create Python package
- https://www.tutorialsteacher.com/python/python-package

## Ideas
- Mehrere Bilder in Liste einlesen (skimage.io.imread_collection)

## if __name__ == '__main__': 
- https://www.data-science-architect.de/__name____main__/

## Dice score
- https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

## Use of image segmentation 
- https://imageannotation.home.blog/2020/06/18/what-is-the-application-of-image-segmentation/

