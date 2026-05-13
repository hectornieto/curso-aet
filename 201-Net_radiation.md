---
title: Radiation transfer in canopies
subject: Tutorial
subtitle: Notebook to evaluate the different relationships and basic processes related to the interception and absorption of irradiance
short_title: Net Radiation
authors:
  - name: Héctor Nieto
    affiliations:
      - Instituto de Ciencias Agrarias, ICA
      - CSIC
    orcid: 0000-0003-4250-6424
    email: hector.nieto@ica.csic.es
  - name: Vicente Burchard-Levine
    affiliations:
      - Instituto de Ciencias Agrarias, ICA
      - CSIC
    orcid: 0000-0003-0222-8706
    email: vburchard@ica.csic.es
  - name: Benjamin Mary
    affiliations:
      - Insituto de Ciencias Agrarias
      - CSIC
    orcid: 0000-0001-7199-2885

license: CC-BY-SA-4.0
keywords: TSEB, radiation, Beer-Lambert law, albedo
myst:
  enable_extensions: ["deflist", "attrs_block", "attrs_inline"]

celltoolbar: Edit Metadata
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.14.3
---

+++

# Summary
This interactive Jupyter Notebook has the objective of evaluating the different relationships and basic processes related to the interception and absorption of irradiance.

We will make use of simulation models, particularly radiative transfer models at both leaf and canopy levels.

+++

# Instructions
Read carefully all the text and follow the instructions.

Once each section is read, run the jupyter code cell underneath (marked as `In []`) by clicking the icon `Run`, or pressing the keys SHIFT+ENTER of your keyboard. A graphical interface will then display, which allows you to interact with and perform the assigned tasks.

To start, please run the following cell to import all the packages required for this notebook. Once you run the cell below, an acknowledgement message, stating all libraries were correctly imported, should be printed on screen.

```{code-cell} ipython3
%matplotlib inline
from pathlib import Path
from ipywidgets import interact, interactive, fixed
from IPython.display import display
from functions import radiation_and_available_energy as fn
import numpy as np
```

# The energy balance
Evapotranspiration is basically an exchange of heat and water between the land surface and the atmosphere. Therefore, the estimation of the energy available at the land surface is one of the key factors affecting its retrieval. 

:::{figure} ./input/figures/radiation_balance.png
:alt:The Earth energy balance
:name:fig-energy-balance
The Global energy balance model
:::
Depending on the surface water status, this available energy is utilized for evaporating water (ET or latent heat flux - $\lambda E$ in terms of energy fluxes) and/or for increasing the surface temperature (sensible heat flux - $H$).

:::{math}
\lambda E + H = R_n - G
:::

where $G$ is the heat transmitted and stored into the ground, and $R_n$ is the net radiation.

The net radiation ($R_n$) can be computed as:

:::{math}
R_n = S^\downarrow \left(1 - \alpha\right) +  \epsilon \left(L^\downarrow - \sigma LST^4\right)
:::

where $S^\downarrow$ and $L^\downarrow$ are the shortwave and longwave incoming irradiances, respectively, which can be obtained from meteorological stations, numerical weather models or meteorological satellites. $\alpha$ and $\epsilon$ are surface albedo and emissivity, respectively, $LST$ is the Land Surface Temperature, and $\sigma\approx5.67E-8$ (W/m²K⁴) is the Stefan-Boltzmann constant. $\alpha$, $\epsilon$ and $LST$ can be estimated, more or less accurately, from Earth Observation (EO) data.

Considering the larger magnitude of shortwave radiance ($S$) compared to the longwave radiance ($L$), we will place more emphasis on the former in this tutorial/notebook. In particular, we will concentrate on the canopy and leaf properties that influence albedo and radiation partitioning between soil and canopy.

On the other hand, $G$ is usually of much lower magnitude, usually considered negligible at daily or even longer scales. When working at shorter timescale, e.g. when estimating $\lambda E$ at the satellite overpass time, $G$ is usually estimated as a fraction of surface net radiation (e.g. $G \approx 0.1 R_n$ for dense canopies), or as a fraction of net radiation of the soil below the canopy ($G \approx 0.3 R_{ns}$).

# Beer-Lambert law
The Beer-Lambert law is a physical law that allows to describe and interpret most of the radiative processes of our interest. When an electromagnetic beam passes through a medium containing an absorber of radiation, this light is attenuated proportionally to the concentration of such absorber ($\kappa$) and to the optical path length of the beam ($\ell$):

:::{math}
\tau = e^{-\kappa\ell}
:::

In the following interactive graph, you can see the effect of this law. Try modifying the values of the extinction coefficient, which increases with the concentration of the absorbing material in the medium. 

The plot will display the fraction of light that is transmitted through the medium as the beam path becomes longer:

```{code-cell} ipython3
w_lambert = interactive(fn.beer_lambert_law, kappa=fn.w_kappa, length=fixed(np.linspace(0, 10, 50)))
display(w_lambert)
```

This law permits, for instance, to explain the attenuation of solar radiation from the stratosphere to the Earth's surface. The air is composed of gases and particles that absorb radiation (most of them at certain wavelengths such as the ozone or the oxygen). The higher the sun is in the sky, the shorter the path length across the atmosphere, and, thus, a greater amount of solar irradiance reaches the land surface.

:::{figure} ./input/figures/airmass.png
:alt:Sun optical path length
:name:fig-path-length
The solar optical path length in a sphere (left) and its planar simplification (right)
:::

## Application to the fraction of intercepted radiation
Beer-Lambert law also allows the analysis or estimation of the radiation that is intercepted by the foliage in a canopy, as well as the radiation transmitted towards the ground.

Assuming that a canopy is primarily composed of leaves, and that these leaves intercept and absorb radiation (mostly photosynthetically active radiation, PAR), we can evaluate how much radiation is intercepted by the canopy according to the foliage density. The latter is usually defined as the Leaf Area Index (LAI): half of the total leaf surface per unit ground. The larger the LAI, the more likely that the sun beams will be intercepted by one or more leaves, and, thus less likely for solar beams to reach the ground. 

The solar elevation angle ($\beta$ in the previous graph), or its complementary the solar zenith angle (SZA), also plays a relevant role. The lower the sun is above the horizon, the longer the path of the sun beams will be when passing through the canopy, and thus the larger the light attenuation.

Since leaves are solid elements with a finite size (as opposed to the gases in the atmosphere), their orientation with respect to the nadir also plays a relevant role. Therefore, the Beer-Lambert law must be adapted to account for this effect. In the next graph, you will be able to experiment with this phenomenon. Try changing both the amount of leaves in terms of LAI and the dominant zenith orientation of leaves, i.e. from canopies with predominantly vertical (LIDF$\approx$90º) leaves to predominantly horizontal (LIDF$\approx$0º) leaves.

```{code-cell} ipython3
w_fipar = interactive(fn.plot_fipar, lai=fn.w_lai, leaf_angle=fn.w_leaf_angle)
display(w_fipar)
```

Watch how the largest fraction of intercepted radiation (or the largest light attenuation) always occurs at higher solar zenith angles (when the path length of the solar beams is larger), but that the attenuation rate depends mostly on the amount of leaves (higher LAI) and their dominant orientation.

LAI depends mainly on the plant development stage and plant growth, while the leaf orientation is more specific for the different species or plant functional types. 

As a rule of thumb, we could expect that plants with predominantly vertical (i.e. erectophylle) leaves are usually adapted to climates with high irradiance levels, with the aim of avoiding large exposure and interception around noon, when irradiance is maximum. On the other hand, plants with predominantly horizontal leaves tend to thrive in more shaded conditions or with low irradiance levels, in order to maximize the interception of solar radiation along the daytime. Between these two extremes, most of the plants show an angular distribution more or less random. For those cases, the calculation of intercepted radiation can be simplified as:

:::{math}
fIPAR = 1 - \exp \left(\frac{-0.5 LAI}{\cos\theta_s}\right)
:::
where $\theta_s$ is the sun zenith angle.

:::{seealso}
If you want to know more details on these calculations you can check the [pyTSEB GitHub source code](https://github.com/hectornieto/pyTSEB/blob/6dd5dffff08bc1f08edb3e89b4a79879e348146b/pyTSEB/TSEB.py#L1587 "https://github.com/hectornieto/pyTSEB/blob/6dd5dffff08bc1f08edb3e89b4a79879e348146b/pyTSEB/TSEB.py#L1587").
:::

# The albedo
As we have previously mentioned, the albedo ($\alpha$) is the key variable that determines the amount of intercepted light that is absorbed by the surface. $\alpha$ is defined as the proportion of incident shortwave radiation that is reflected by the surface. The shortwave net radiation ($S_n$) is therefore the balance between the incident shortwave irradiance ($S^\downarrow$) and the reflected shortwave radiance ($S^\uparrow = \alpha S^\downarrow$)

The spectral properties of the surface are key in determining the albedo. For instance, fresh snow shows large values ($\alpha\approx 0.9$) and, thus, reflects most of the solar irradiance, while oceans have much lower values ($\alpha\approx0.05$), absorbing a large amount of solar irradiance.

Leaves, due to their photosynthetic activity, absorb a large proportion of light due to the presence of chlorophylls, as well as other leaf pigments. For that reason, leaves yield relative low albedo values ($\approx0.15$). On the other hand, soils can have a large range of albedo values, depending on their mineral composition, texture and topsoil moisture. Therefore, the albedo of a vegetated surface will depend not only on leaf chlorophyll concentration but also on canopy density and, in a lesser degree, on the soil albedo in situations of sparse vegetation or initial growth stages.


## Vegetation anisotropy
Most of the Earth's surface show certain anisotropic behaviour when reflecting radiation i.e. it scatters different amounts of radiation depending on the scattering direction. The most extreme cases would be specular surfaces (such as mirrors, still water bodies or ice), which scatter most the irradiance in the opposite direction of the incidence angle. The opposite case would be lambertian surfaces, where apparent brightness/scattering is the same regardless the direction.

:::{figure} ./input/figures/lambertian_and_specular_reflection.png
:alt:Lambertian and specular scattering
:name:fig-anisotropy
The Lambertian (thin arrow lines) and specular (bold arrow line) scattering
:::

:::{note}
:class:dropdown
Unpolished wood would be closer to a lambertian surface, while polished and varnished wood would behave more as a specular/non-lambertian surface. 
:::

Vegetation, as it is mainly composed by an array of leaves, is also affected by this anisotropic behaviour. Therefore, plants will reflect radiation differently depending on the illumination geometry (i.e. the solar  position) and the scattering direction (i.e. the sensor position), possibly changing their albedo.

You can visualize this effect in the following plot. It depicts two polar graphs, one for the photosynthetically active radiation (PAR, 400-700$\mu$m) and a second one for the shortwave radiation (SW, 400-2500$\mu$m). These plots represent the reflectance factors simulated for a homogeneous canopy, using the radiative transfer models [PROSPECT-D](10.1016/j.rse.2017.03.004) and [4SAIL](10.1109/TGRS.2007.895844).

The concentric rings represent the range of zenith angles (zenith=0º at the plot origin, and 90º at the outermost ring). The radii of the plots show the azimuth angles (0º bearing north, 90º bearing east, 180º bearing south, and 270º bearing west). The solar position is depicted in the graph with a star. By default the sun is placed at solar noon in Northern latitudes (180º azimuth), at a solar zenith angle of 37º. The plots also print the integrated value of the reflectance factors , which is actually the albedo.

:::{seealso}
If you want to know more details on these calculations you can check [pypro4sail GithHub source code](https://github.com/hectornieto/pyPro4SAIL "https://github.com/hectornieto/pyPro4SAIL")
:::

```{code-cell} ipython3
w_bidirectional = interactive(fn.bidirectional_reflectance,
                              cab=fn.w_cab, cw=fn.w_cw, lai=fn.w_lai, leaf_angle=fn.w_leaf_angle, 
                              sza=fn.w_sza, saa=fn.w_saa, skyl=fn.w_skyl, soil_type=fn.w_soil)
display(w_bidirectional)                             
```

* Modify the chlorophyll content - `Cab`. How do reflectance factors and albedo change in the PAR region (400-700$\mu$m) and in the solar spectrum (400-2500$\mu$m)?

* Likewise, modify the leaf water content- `Cw`. How do reflectance factors and albedo change in the PAR region (400-700$\mu$m) and in the solar spectrum (400-2500$\mu$m)?

* Modify the solar position - `SZA` and `SAA`. On one side, you will see a peak of reflectance around the solar position (the star in the plot). This is the so-called hotspot, or the area most illuminated in the canopy. By contrast, the lowest values are on the opposite side, since leaves in this area are more occluded by the rest of the canopy. The albedo values also vary with the solar position, i.e. albedo changes along the day.

* This variation of albedo with solar  position is stronger with lower values of diffuse radiation. Increase the ratio of diffuse radiation `skyl`, which would simulate cloudy days. On totally overcast days (`skyl=1`), you will notice that the solar position is no longer relevant for the albedo or reflectance factors.

# Net shortwave radiation
Previously, we have seen that LAI, leaf angle distribution and chlorophyll concentrations are the probably the most relevant variables that determine the absorption of shortwave irradiance within a crop/vegetation layer.

In the next plot, we see the daily trend of net shortwave radiation for a horizontally homogeneous herbaceous crop (such a wheat field). These simulations assume that the cloudiness remains constant along the day and, thus, shortwave irradiance displays a sinusoidal curve: should cloudiness change along the day, the shortwave irradiance would display a series of valleys and peaks, according to whether the sun is occluded by clouds or not.

:::{important}
:class: dropdown
On the other hand, these simulations assume a constant ratio of diffuse irradiance along the day (fixed by the value set by `Skyl`). This is unrealistic in most cases, since around sunrise and sundown diffuse radiation tends to be larger, even under clear sky conditions. But for simplicity in these simulations, we made this assumption.
:::

The plot depicts the net radiation at the land surface ($S_n$, black line) as well as the net radiation partitioning between crop ($S_{n,C}$, green line) and ground ($S_{n,S}$, yellow line). The secondary axis displays the albedo in blue line.

:::{math}
S_{n,C}+S_{n,S}=S_n\\
\alpha = 1 - S_n/S\downarrow
:::

```{code-cell} ipython3
w_sn = interactive(fn.plot_net_solar_radiation,
                   lai=fn.w_lai, leaf_angle=fn.w_leaf_angle, h_c=fixed(1), f_c=fixed(1),
                   sdn_day=fn.w_sdn,
                   row_distance=fixed(1), row_direction=fixed(1), skyl=fn.w_skyl, 
                   fvis=fixed(0.55), lat=fn.w_lat, cab=fn.w_cab, cw=fn.w_cw, soil_type=fn.w_soil)
display(w_sn)
```

+++ {"jp-MarkdownHeadingCollapsed": true}

:::{important}
All parameters for all the plots used in this notebook are synchronized: any parameter change in one of the interactive plots is as well changed for the other plots, with all the graphs updated accordingly. This allows a quick intercomparison between graphs and simulations.
:::

* While keeping the chlorophyll and water content constant, observe how net radiation barely changes with variations of LAI. However, the partitioning of net radiation between the canopy and soil does drastically change with LAI. This has an important implications since LAI is key for the radiation partitioning and, thus, for the evaporative/transpirative capacity of soil and vegetation, as well as for photosynthesis.

* Now keep relatively low values of LAI (<1.5) and inspect the effect of leaf angular distribution. More vertical leaves (`LIDF`$\rightarrow 90^{\circ}$) would provoke a reduction of intercepted, and thus absorbed, radiation near solar noon. Watch now that if diffuse radiation increases (`Skyl`$\rightarrow$1), this effect is no longer evident. In this case, most of the radiation is coming equally from any direction of the hemisphere, making leaf inclination less relevant.

* Change the soil type and see how soil albedo affects net radiation, in particular at lower values of LAI when soil is more exposed to solar irradiance.

:::{seealso} 
If you want to know more details about these calculations you can check the [pyTSEB Github source code](https://github.com/hectornieto/pyTSEB/blob/9d1f02ec2968bb323c07528ba44144c3733f3737/pyTSEB/net_radiation.py#L543 "https://github.com/hectornieto/pyTSEB/blob/9d1f02ec2968bb323c07528ba44144c3733f3737/pyTSEB/net_radiation.py#L543").
:::

+++

# Shortwave radiation in clumpled canopies
So far, all simulations and processes have assumed a horizontally homogeneous canopy or crop, or at least dense enough to approximate a homogeneous crop. However, what happens when one is working with clumped canopies such as row crops or isolated canopies, e.g. in vineyards or orchards?

The hard fact is that everything is much more complex as on the one hand the canopy is usually clumped around a stem and branches, leaving some parts of the ground fully exposed. On the other hand, in row crops, the dimension of the canopy/trellis system and the orientation of the rows also affect how light is intercepted and transmitted between the crop and the ground.

In order to evaluate this situation, we need to consider additional parameters of crop structure:

* The canopy height ($h_c$), since higher canopies will cast longer shadows in the interrow.
* The fraction of canopy occupied by the ground ($f_c$), or the ratio between the canopy/trellis width ($w_c$) and the row spacing (L).
* The azimuth row orientation, with respect to the north. 0º will indicate that the rows are oriented S-N, negative values indicate that the rows go from NW to SE, and positive values from NE to SW, with -90º and/or 90º showing rows oriented E-W.

:::{figure} ./input/figures/row_crop_shadowing.webp
:alt: Row crop shadowing model
:name: row-crop-model
Canopy model for estimating the clumping index in row crops, F is the leaf area index, L is the distance between rows, hc is the canopy height, hb is the height of the canopy above ground, and wc is the canopy width
:::

The approach builds upon the concept of the clumping index, defined as a conversion factor that modifies the leaf area index of a real canopy (F) in a fictitious homogeneous canopy with $LAI_{eff} = \Omega\left(\theta, \psi\right) F$, such as its gap fraction ($G\left(\theta, \psi\right)$) is the same as the gap fraction of the real-world canopy:

:::{math}
:label:eq-gapfraction
\Omega\left(\theta,\psi\right) F \kappa_b\left(\theta\right)= -\log\left[G\left(\theta,\psi\right)\right]
:::
with $\kappa_b\left(\theta\right)$ is the beam extinction cofficient computed for a homogenous canopy (e.g. using [Campbell and Norman (1998)](https://doi.org/10.1007/978-1-4612-1626-1)).

The real canopy gap fraction is estimated on this simplied model as the sunlit part of the bare soil that is not shaded by the canopy ($1 - f_{sc}$) plus the gaps caused by the solar beam passing through the crop canopy (ignoring mutual shadowing between rows:

:::{math}
G(\theta ,\psi)={f_{sc}}(\theta ,\psi )\exp [ - {\kappa _{{\text{b}}}}(\theta )F]+[1 - {f_{{\text{sc}}}}(\theta ,\phi )].
:::

and $f_{sc}$ is computed from (#row-crop-model) using trigonometry:

:::{math}
{f_{{\text{sc}}}}(\theta ,\phi )=\frac{{{w_{\text{c}}}+({h_{\text{c}}} - {h_{\text{b}}})\tan \theta |\sin \phi |}}{L}
:::

In the following interactive plot, we have added these parameters to see their effect on net radiation. In addition, to ease the comparison with previous plot, we have included a second graph that shows the fraction of absorbed radiation for both a horizontally homogeneous crop and a row crop.

```{code-cell} ipython3
w_sn = interactive(fn.plot_net_solar_radiation,
                   lai=fn.w_lai, leaf_angle=fn.w_leaf_angle, h_c=fn.w_hc, f_c=fn.w_fc,
                   row_distance=fn.w_interrow, row_direction=fn.w_psi, sdn_day=fn.w_sdn, skyl=fn.w_skyl, 
                   fvis=fixed(0.55), lat=fn.w_lat, cab=fn.w_cab, cw=fn.w_cw, soil_type=fn.w_soil)
display(w_sn)
```

* The sun rises from the eastern side, at noon reaches its zenith towards the north (or south depending on whether we are north or south of the equator), and sunset is from the western side. Taking into account these facts, observe the effect of changing the row orientation from E-W to S-N. There should be a drop in canopy net radiation when the sun is parallel to the rows, as in this case there should be lower radiation interception by the canopy.

* Observe that with larger canopy fractions ($f_c$, or wider canopies relative to the row spacing), the crop behaves similarly to the homogeneous crops from the previous simulation.

* There is an interaction effect between canopy height and row separation (`L`). Narrower rows in lower crops display a similar effect as that of taller crops in wider rows.

* Finally, as we have seen in previous interactive plots, these effects are minimized at larger proportion of diffuse radiation.

:::{seealso}
Check the code in the [pyTSEB GitHub repository](https://github.com/hectornieto/pyTSEB/blob/9d1f02ec2968bb323c07528ba44144c3733f3737/pyTSEB/clumping_index.py#L124).
:::

+++

# Net longwave radiation
Net longwave radiation ($L_n$) corresponds to the thermal emission of radiation from the land surface and the absorption  of thermal irradiance from the atmosphere. $L_n$ depends therefore on the surface's temperature and its emissivity, as well as on the atmospheric thermal emission ($L\downarrow$), which at the same time depends on the atmospheric temperature and emissivity.

> $L_n = \left(1 - \epsilon_{surf}\right) L\downarrow - \epsilon_{surf} \sigma T_{surf}^4$

In the next interactive plot, we will see how net longwave radiation changes as a function of atmospheric conditions (air temperature and humidity) and the surface's temperature and emissivity. Since surface temperature also depends on atmospheric conditions, we simulated various ranges of surface temperatures in the x-axis, from fully watered crops in which the surface temperature is closer to the air temperature, to stressed crops and bare/sparse areas, in which surface temperature is significantly hotter than the air.

```{code-cell} ipython3
w_ln = interactive(fn.plot_longwave_radiation, t_air=fn.w_tair, hr=fn.w_hr, delta_t=fixed(np.linspace(-1, 20, 50)),
                   emiss=fn.w_emiss)
display(w_ln)
```

* Observe that the net longwave radiation yield values at a lower magnitude than the net shortwave radiation ( especially for clear skies around solar noon time). Only in cases with high temperatures and humidity net radiation exceeds 400 W/m².

* Finally, the surface emissivity, the values of which usually range from 0.95 in arid areas with scarce vegetation to 0.99 in dense vegetation surfaces, has a lesser effect in the calculation of net longwave radiation. Therefore, for computing of net radiation its retrieval does not require very high accuracy.

+++

# Conclusions
In this exercise we have seen that net shortwave radiation depends mainly on:
1. Shortwave irradiance, which can be obtained from meteorological stations or from numerical weather models (e.g. [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview")).
2. Surface albedo

Furthermore, the surface albedo mainly depends on:
1. Leaf and soil spectral properties, mainly leaf chlorophyll content as the main pigment absorber in the PAR region.
2. Canopy structural characteristics, mainly LAI but also leaf angle distribution.
3. In addition, for row crops, the row architecture may be crucial in characterizing the radiation partitioning between ground and canopy.

The net longwave radiation has a lower contribution to the global net radiation ($R_n$), and its estimation is usually simpler, requiring:
1. Incident longwave irradiance, which can be obtained from meteorological stations or from numerical weather models (e.g. [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview")).
2. Surface emissivity and temperature, both of which can be obtained from thermal infrared Earth Observation data.

These analyses will allow you to better understand the radiative part of the energy balance. In addition, these simulations will permit evaluating the cost/benefit of using simpler or more sophisticated models for estimating net radiation and its partitioning between soil and canopy in structurally complex canopies.

In the next exercise, you will work on the other main factor of the energy balance that can be estimated with thermal infrared Earth Observation, the [sensible heat flux](./102-Turbulence_and_sensible_heat_flux.ipynb).

:::{note}
Please feel free comment any thoughts.
:::
