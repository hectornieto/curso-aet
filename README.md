# Workshop on pyTSEB modelling framework
Digital *Jupyter Book* collection developed for the **Workshop on pyTSEB modelling framework**.

## Installation
You can use [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgit.csic.es%2Ftech4agro%2Fcourses%2Feomaji-eoafrica/HEAD) for runing the notebooks in the cloud

Optionally you could run the notebooks locally. To do so, you should have these programs installed:

* Python and/or [Anaconda](https://www.anaconda.com/download/success). 
* [Git](https://git-scm.com/downloads)

We need to install all the library requisites for this tutorial included in [requisites.txt](./requirements.txt) or in [environment.yml](./environment.yml) 

Install the requirements either with conda/mamba:

`mamba env create -f environment.yml`

or

`conda env create -f environment.yml`

or with pip:

`pip install -r requirements.txt`


## Run the interactive book
Run the following command to initialize the book:

`bash run_workshop.sh`

or 

`run_workshop.bat` if you are working under Windows.

Open your web browser to [`http://localhost:3000`](http://localhost:3000)

## Contents
1. Physical principles.
    
    a. [Radiation transfer](./101-Net_radiation.ipynb)

    b. [Turbulent exchange of heat and vapour](./102-Turbulence_and_sensible_heat_flux.ipynb)
 
2. Introduction to TSEB
    
    a. [How TSEB works?: a quick overview of low-level code](./201-TSEB_introduction.ipynb)

3. [Optional] Retrieval of biophysical traits for TSEB
    
    a. [Use of PROSPECT-D and 4SAIL for estimating biophysical traits for TSEB](./301-spectral_properties.ipynb)


## License
Creative Commons Attribution-ShareAlike 4.0 International.

This work is licensed under Attribution-ShareAlike 4.0 International. To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/

This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, even for commercial purposes. If others remix, adapt, or build upon the material, they must license the modified material under identical terms.

  - BY: Credit must be given to you, the creator.
  - SA: Adaptations must be shared under the same terms. 
  



