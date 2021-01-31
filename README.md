# IISC Course Assignments

What is this repository about?
There are several python notebooks and codes which are part of this folder. These are some assignments which I had done as part of the 3 credit coursework at IISC. 
As part of the coursework, clustering, recommendations, image classification, retrieval problems were solved.

For running the code, you may follow the below steps:
1. Run VSCode (or any other code editor) from Anaconda Command Prompt
2. Make sure you are using a dedicated virtual environment and activate it,
sample - 
 
'''code
conda create -n myenv python=3.7.3
'''

3. Install or upgrade all packages from requirements.txt

conda install --file requirements.txt
or pip install -r requirements.txt

You may need to add the full path of requirements.txt file depending on your pwd

4. You would have to change all data paths accordingly.

Important links:
1. Conda environment - https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
2. https://wiki.epfl.ch/help-git-en/how-can-i-import-an-existing-folder-into-my-repository

# Folders & Code Description -
## data - This folder contains all data files for the different workbooks & codes
## IRIS_Clustering_code - This folder contains 3 files : nn.py (Nearest Neighbour), kmeans_final.py , and normalizing.py

These files can be run using a python shell or any text editor (ex. ATOM).
For example -
Steps to execute python code using shell:
1. Open the nn.py file
2. Change the path of the training and test files as needed. It can be set in the code line no. 5 and 8 as 'path_to/iris_training.csv' and 'path_to/iris_test.csv'
3. Run using the command : python nn.py

Steps to execute code using any other text editor like ATOM:
1. Install package called "script"
2. Press Ctrl+Shift+P and then type "Script Options", set the current directory wherever you have saved the python code and save the profile under some name.
3. Change the path of the training and test files as needed. It can be set in the code line no. 5 and 8 as 'path_to/iris_training.csv' and 'path_to/iris_test.csv' (Preferably store these in the same location as the python code and no need to change this path)
4. Open the python code in the ATOM text editor
5. Press Ctrl+Shift+P and type "Script with profiles", select your saved profile from the list and then run the file.

## Notebooks
This folder contains several workbooks which solves problems like recommendations, analyzing product sentiment, image classification, image retrieval, document retrieval among others.
These workbook would use graph lab package for these purposes.
