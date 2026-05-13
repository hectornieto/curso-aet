#!/usr/bin/bash -l
source activate tseb-workshop 
jupyter lab --NotebookApp.token=jupyterlab-tseb-workshop --NotebookApp.allow_origin='http://localhost:3000' & 
myst start
