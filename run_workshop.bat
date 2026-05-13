@REM Activating environment
@CALL conda activate tseb-workshop

@REM Launching Juupyter Lab at port 3000
start /b jupyter lab --NotebookApp.token=jupyterlab-tseb-workshop --NotebookApp.allow_origin='http://localhost:3000'

@REM Stargint mystmc, please wait...
start /b myst start
