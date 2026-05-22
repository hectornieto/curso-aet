---
title: Calculation of Shortwave and Longwave Net Radiation
subject: Tutorial
subtitle: Campbell & Norman Radiative Transfer
short_title: Net Radiation Calculation
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

license: CC-BY-SA-4.0
keywords: TSEB, 3SEB, radiation, Beer-Lambert law, albedo
myst:
  enable_extensions: ["deflist", "attrs_block", "attrs_inline"]

jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++

# Summary
This interactive Jupyter Notebook has the objective of showing the implemenation of TSEB-PT model in the [pyTSEB package](https://github.com/hectornieto/pyTSEB).


# Instructions
Read carefully all the text and follow the instructions.

Once each section is read, run the jupyter code cell underneath (marked as `In []`) by clicking the icon `Run`, or pressing the keys SHIFT+ENTER of your keyboard. A graphical interface will then display, which allows you to interact with and perform the assigned tasks.

To start, please run the following cell to import all the packages required for this notebook. Once you run the cell below, an acknowledgement message, stating all libraries were correctly imported, should be printed on screen.

```{code-cell} ipython3
%matplotlib inline
from pathlib import Path
from ipywidgets import interact, interactive, fixed, widgets
from IPython.display import display
import numpy as np
import pandas as pd
import tabulate
from pyTSEB import TSEB
from py3seb import py3seb
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
print("Libraries imported!")
```

# Net shortwave radiation
To compute the net shortwave radiation and its partitioning between the canopy and the soil we are using the model of {cite:t}`10.1007/978-1-4612-1626-1_15{Chapter 15}`. The key aspect of this model is the calculation of the trasmitted shortwave radiation throught the canopy ($\Tau_C$)

:::{seealso}
:class:dropdown
The full description of the radiation model can be found in {cite:t}`10.1007/978-1-4612-1626-1_15{Chapter 15}`
:::

The transmittance of shortwave radiation through the canopy depends on the wavelength due to vegetation absorbing a greater portion of the photosynthetically active radiation (PAR, 400–700 nm spectrum) than near-infrared radiation (NIR, 700–2500 nm spectrum) wavelengths. ($\Tau_C$ is partitioned into two components (direct/diffse) and in two spectral band (PAR/NIR)

:::{note}
:class:dropdown
We could include more spectral bands, such as (PAR/NIR/SWIR), with a slight increase of computational cost. 
:::

The direct-beam spectral (transmittance ($\tau_{C,DIR,\lambda}$) at a given solar zenith angle ($\theta_S$) is calculated following the equations of {cite:t}`10.1007/978-1-4612-1626-1_15` for a single layer crop:

:::{math}
:label:eq-tau-dir
\tau_{C,DIR,\lambda}\left(\theta_S\right)=\frac{\left(\rho_{C,\lambda}^*\left(\theta_S\right)^2 - 1\right)\exp \left( -\sqrt{\zeta_\lambda} \kappa_{b}\left(\theta_S\right)LAI\right)}{\left(\rho_{C,\lambda}^*\rho_{S,\lambda} - 1\right) + \rho_{C,\lambda}^*\left(\psi\right)\left(\rho_{C\lambda}^*\left(\theta_S\right) - \rho_{S,\lambda}\right)\exp \left( -2\sqrt {\zeta_\lambda \kappa_{b}\left(\theta_S\right)LAI} \right)}
:::

with ($\lambda$) being either the PAR or NIR. $\rho_{C,\lambda}^*\left(\psi\right)$ is the beam spectral reflection coefficient for a deep canopy with non-horizontal leaves (see Eq. [](#eq-rho-star)), $\zeta_\lambda$ is the leaf absortivity, $\kappa_{b}$ is the extinction coefficient for direct-beam radiation (per LAI unit), and $\rho_{S,\lambda}$ is the soil spectral reflectance. The multiple scattering between the soil and the canopy is accounted for in the $\rho _{C,\lambda}^*$ and $\rho_{S,\lambda}$ terms.

:::{math}
:label:eq-rho-star
\rho_{C,\lambda}^*\left(\theta_S\right)=\frac{2\kappa_b\left(\theta_S\right) \rho_\lambda^H}{\kappa_b\left(\theta_S\right)+1}
:::

$\rho_\lambda^H=\frac{1 - \sqrt{\zeta_\lambda}}{1+\sqrt{\zeta_\lambda}}$ is the reflectance factor for a canopy with horizontal leaves.

Finally, the canopy beam extinction $\kappa_b\left(\psi\right)$ is calculated based on the ellipsoidal LIDF of {cite:t}`10.1016/0168-1923(90)90030-A`

:::{math}
:label:eq-kbe
\kappa_b\left(\theta_S\right)=\frac{\sqrt{\chi^2+ \tan^2\theta_S}}{\chi+1.774\left(\chi+1.182\right)^{-0.733}}
:::
with $\chi$ the {cite:t}`10.1016/0168-1923(90)90030-A` LIDF parameter
:::{note}
$\chi = 1$ is the standard value for LIDF spherical distribution, with values lower than 1 reflecting planophylles canopies and $\chi > 1$ for erectophille canopies.
:::

Diffuse spectral transmittance ($\tau_{C,DIF,\lambda}$) is calculated by numerically integrating $\kappa_b$ over the hemisphere 

:::{math}
:label:eq-kd
\kappa_d =2 \int_0^\pi\kappa_b\left(\psi\right) \sin\psi \cos\psi d\psi
:::
and replacing $\kappa_b$ by $\kappa_d$ in Eq. [](#eq-tau-dir)

Similarly the canopy direct spectral albedo is computed as:
:::{math}
:label:eq-rho-dir
\rho_{C,DIR,\lambda}\left(\theta_S\right)=\frac{\rho_{C,\lambda}^*\left(\theta_S\right) + \left[\frac{\rho_{C,\lambda}^*\left(\theta_S\right) - \rho_{S,\lambda}}{\rho_{C,\lambda}^*\left(\theta_S\right)\rho_{s,\lambda} -1}\right] \exp\left(-2\sqrt{\zeta_\lambda}\kappa_b\left(\theta_S\right)LAI\right)}{1 + \rho_{C,\lambda}^*\left(\theta_S\right) + \left[\frac{\rho_{C,\lambda}^*\left(\theta_S\right) - \rho_{s,\lambda}}{\rho_{C,\lambda}^*\left(\theta_S\right)\rho_{S,\lambda} -1}\right] \exp\left(-2\sqrt{\zeta_\lambda}\kappa_b\left(\theta_S\right)LAI\right)}
:::
and the diffuse canopy albedo ($\rho_{C,DIF,\lambda}$) by replacing $\kappa_b\left(\theta_S\right)$ by $\kappa_d$

Therefore as inputs for computin net shortwave radiation we need to provide as input forcing the solar irradiance, direct and diffuse in the PAR and NIR, and then as main input the $LAI$, followed by soil and leaf spectra and {cite:t}`10.1016/0168-1923(90)90030-A` LIDF parameter ($\chi$). 

:::{seealso}
The code for estimating the diffuse and bean radiation is based on {cite:t}`10.1016/0168-1923(85)90020-6` and can be found at the [pyTSEB GitHub repository](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/net_radiation.py#L62)

The code for running the {cite:t}`10.1007/978-1-4612-1626-1_15` model can be found at the pyTSEB GitHub repository: [canopy spectral properties](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/net_radiation.py#L439) and [estimation of net shortwave radiation](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/net_radiation.py#L546C5-L546C21)
:::

## Retrieval of broadband spectral properties
For estimating leaf spectra we will use the leaf traits retrievals from the satellite biophysical traits at `input/canopy`. With the information of these traits we could run in forward mode the ProspectD leaf RTM to get a spectral reflectance and transmittance that could then be integrated to the required broadband regions. However, running ProspectD for a large array of pixels is too computationally expensive. For that reason we have developed a ProspectD emulator {cite:p}`10.1109/JSTARS.2021.3122573`: we generated broadband leaf reflectances and transmittances in the PAR and NIR from a large range of ProspectD simulations, covering all plausible range of leaf traits, which then will be used to train another random forest model that will relate the leaf traits to the broadband PAR and NIR leaf reflectance and transmittance.

Soil reflectance has a smaller role in most situations, since the radiation reaching the ground is smaller due to the interception of light by the canopy. However, in sparse and semi-arid conditions, where vegetation is scarce or even not present, soil albedo plays obviously a significant role. Furthermore, these semi-arid areas are usually characterized by brighter soils since they are composed by sands, salts and with a small fraction of organic matter. For that reason in {cite:t}`10.1109/JSTARS.2021.3122573` we developed a method to unmix the broadband soil reflectance ($\rho_{soil, \lambda}$) from the satellite observed reflectances (Eq. \ref{eq:rho-soil})

:::{math}
:label:eq:rho-soil
\rho_{soil, \lambda} =  \frac{\rho_{surface, \lambda} - \left[1 - P_0\left(\theta_v\right)\right] \rho_{leaf, \lambda}}{ P_0\left(\theta_v\right)} 
:::
where $P_0\left(\theta_v\right)$ is the canopy gap fraction at the satellite observation angle, computed by the Beer-Lambert law ($\exp\left[-\kappa\left(\theta_v\right) \mathrm{LAI}\right]$). $\rho_{surface, \lambda}$ is the broadband surface reflectance, and $\rho_{leaf, \lambda}$ is the leaf broadband reflectance.

+++

### Select a site

```{code-cell} ipython3
# Set the canopy and eddy covariance folders
input_dir = Path().absolute() / "input"
ec_dir = input_dir / "eddy_covariance"
sites = ec_dir.glob("*_SUBSET_HH.csv")
sites = [i.stem.split("_")[1] for i in sites]
# Site metadata file
site_metadata_file = input_dir / "sites.csv"
site_metadata = pd.read_csv(site_metadata_file, sep=";")
valid = np.isin(site_metadata["SITE_ID"], sites)
site_metadata = site_metadata.loc[valid]
                
w_site_list = widgets.Dropdown(
    options=sites,
    value=sites[0],
    description='Select site:',
    rows=9
)
display(w_site_list)
site_metadata
```

### Read the LAI and Micrometeorology data
We will read and merge the [input hourly micrometeorology](./input/eddy_covariance/) and [input biophysical traits](./input/canopy/) ASCII files. 

:::{seealso}
A detailed description of the Eddy Covariance measurements and the FLUXNET/OneFLUX post-
processing is described by {cite:t}`10.1038/s41597-020-0534-3`.

Estimates of daily LAI were obtained from an hybrid inversion of PROSPECTD and 4SAIL radiative transfer models using multiespectral satellite data from Sentinel-2 and Landsat. For more details about the approach you can read the notebook [301_spectral_properties](./301_spectral_properties.md)
:::

```{code-cell} ipython3
bio_dir = input_dir / "canopy"

# Get site properties
site = w_site_list.value
site_metadata = pd.read_csv(site_metadata_file, sep=";")
site_id = site_metadata["SITE_ID"] == site
lat = site_metadata.loc[site_id, "LOCATION_LAT"].item()
lon = site_metadata.loc[site_id, "LOCATION_LONG"].item()
elev = site_metadata.loc[site_id, "LOCATION_ELEV"].item()
utc_offset = site_metadata.loc[site_id, "UTC_OFFSET"].item()
igbp = site_metadata.loc[site_id, "IGBP"].item()

# Set the input files based on the chosen site
bio_filename = bio_dir / f"{site}_HLS-l2c.csv"
ec_filename = ec_dir / f"FLX_{site}_FLUXNET_SUBSET_HH.csv"
print(f"Biophysical traits file path is {bio_filename}")
print(f"EC file path is {ec_filename}")

# Read the fluxnet EC table and preprocess
ec =  pd.read_csv(ec_filename, sep=',', na_values=-9999)
ec["TIMESTAMP_START"] = pd.to_datetime(
    ec["TIMESTAMP_START"], format="%Y%m%d%H%M")

ec["TIMESTAMP_END"] = pd.to_datetime(
    ec["TIMESTAMP_END"], format="%Y%m%d%H%M")

new_cols = {}
new_cols["TIMESTAMP"] = pd.to_datetime(
        ec[["TIMESTAMP_START", "TIMESTAMP_END"]].mean(axis=1))

new_cols["DATE"] = new_cols["TIMESTAMP"].dt.date
ec = pd.concat([ec, pd.DataFrame(new_cols)], axis=1)

# Read the satellite biophysical traits table
bio = pd.read_csv(bio_filename, sep=";", na_values=-9999)
bio["DATE"] = pd.to_datetime(bio["DATE"], format="%Y-%m-%d").dt.date

# Merge both tables by date
ec = ec.merge(bio, on="DATE")

fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    horizontal_spacing=0.01,
                   subplot_titles=("LAI", "Leaf pigments"))

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["LAI"], 
                         name="LAI", mode="lines", line={"color": "black"}),
              row=1, col=1)

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["LAI"] * ec["FG"], 
                         name="gLAI", mode="lines", line={"color": "green", "dash": "dash"}),
              row=1, col=1)

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["ANT"], 
                         name="Leaf Antocyanins", mode="lines", line={"color": "red"}),
              row=2, col=1)

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["CAB"], 
                         name="Leaf Chorophyll", mode="lines", line={"color": "green"}),
              row=2, col=1)

fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_layout(title_text=f"Biophysical traits timeseries in {site}")
```

### Get the broadband spectral properties

+++

### Estimate net shortwave radiation based on biophyisical traits and incoming irradiance
Now we can execute the cell below to run the {cite:t}`10.1007/978-1-4612-1626-1_15` radiative transfer model implemente in pyTSEB using the site selected in [](#Select-a-site) section.

Besides of LAI, its leaf inclination distribution function and the spectral properties, in TSEB we also make use of the fractional cover ($f_c$) and canopy shape ($w_c/h_c$) for clumped canopies. These are used to computed a clumping index that converts the LAI to effective values based on the solar incidence angle (since in clumpled canopies shading can play a significant role)

:::{warning}
Both $f_c$ and $w_c/h_c$ are structural variables in TSEB and they cannot be operationally retrieved from satellite data. The typical `FCOVER` product **should not** be used as surrogate for $f_c$ since `FCOVER` not only represents the canopy clumpiness but also the canopy gap fraction.

In operational implementations, we derived a static $f_c$ value only for clumped canopies using prescribed values based on land cover/plant functional type
:::

```{code-cell} ipython3
RF_OBJECT = input_dir / "leaf_spectra.joblib"

def leaf_spectra(cab, car, cbrown, cw, cm, ant,
                 reg_object_file=RF_OBJECT):
    from joblib import load
    """Estimates broaleaf leaf spectra from PROSPECT-D biphysical traits"""
    
    reg_object = load(reg_object_file)
    x_array = pd.DataFrame({"Cab": cab,
                            "Car": car,
                            "Cbrown": cbrown,
                            "Cw": cw,
                            "Cm": cm,
                            "Ant": ant})

    abs_est = reg_object.predict(x_array).T
    return abs_est


def components_spectra(input_df, lai_sparse=1e6):
    [input_df["RHO_LEAF_VIS"], input_df["RHO_LEAF_NIR"],
     input_df["TAU_LEAF_VIS"], input_df["TAU_LEAF_NIR"]] = leaf_spectra(
        input_df["CAB"].values,
        input_df["CAR"].values,
        input_df["CBROWN"].values,
        input_df["CW"].values,
        input_df["CM"].values,
        input_df["ANT"].values)

    input_df["RHO_SOIL_VIS"], input_df["RHO_SOIL_NIR"] = soil_spectra(
        input_df, lai_sparse=lai_sparse)

    return input_df


def soil_spectra(input_df, lai_sparse):
    """Unmix the broadban surface reflectance beween 
    PROSPECT-D leaf reflectance and estimatade soil spectra
    """
    
    rho_soil_vis = np.full_like(input_df["LAI"], 0.15)
    rho_soil_nir = np.full_like(input_df["LAI"], 0.25)

    # Bare soil conditions, soil reflentance = satellite reflectance
    bare = np.logical_or(input_df["LAI"] == 0, input_df["F_C"] == 0)

    rho_soil_vis[bare] = input_df.loc[bare, "RHO_VIS"]
    rho_soil_nir[bare] = input_df.loc[bare, "RHO_NIR"]

    # Sparse vegetation conditions
    sparse = np.logical_and.reduce((input_df["LAI"] > 0,
                                    input_df["LAI"] <= lai_sparse,
                                    input_df["F_C"] > 0))

    local_lai = input_df.loc[sparse, "LAI"] / input_df.loc[sparse, "F_C"]
    fvc = TSEB.calc_F_theta_campbell(0,
                                     local_lai,
                                     x_LAD=input_df.loc[sparse, "X_LAD"])
    f_soil = (1 - input_df.loc[sparse, "F_C"]) + input_df.loc[sparse, "F_C"] * (1.0 - fvc)

    rho_soil_vis[sparse] = (input_df.loc[sparse, "RHO_VIS"] - (1 - f_soil)
                            * input_df.loc[sparse, "RHO_LEAF_VIS"]) / f_soil
    rho_soil_nir[sparse] = (input_df.loc[sparse, "RHO_NIR"] - (1 - f_soil)
                            * input_df.loc[sparse, "RHO_LEAF_NIR"]) / f_soil

    rho_soil_vis = np.clip(rho_soil_vis, 0.01, 0.99)
    rho_soil_nir = np.clip(rho_soil_nir, 0.01, 0.99)
    return rho_soil_vis, rho_soil_nir

print("Leaf and soil spectra functions correctly parsed, you can continue")
```

```{code-cell} ipython3
import yaml

# Set the input TSEB ancillary parameters
tseb_param_file = input_dir / "TSEB_params.yaml"

# Read metadata and get site info
site_metadata = pd.read_csv(site_metadata_file, sep=";")
site_id = site_metadata["SITE_ID"] == site

# Avoid pandas frabmentation warnings
ec = ec.copy()

# Compute the Cambpell chi LIDF parameter from Campbell mean leaf angle
ec.loc[:, "X_LAD"] = TSEB.rad.leafangle_2_chi(ec["LEAF_ANGLE"].values)

with open(tseb_param_file, "r") as fid:
    tseb_params = yaml.load(fid, Loader=yaml.FullLoader)

ec.loc[:, "F_C"] = tseb_params["F_C"][igbp]
w_c = tseb_params["W_C"][igbp]

# The ASCII table is missing the solar angles, so we will use the calc_sun_angles function of TSEB to compute the angles based on site location and timestamp
# The standard meridian time zone (15 deg. per hour)
stdlon = 15 * utc_offset
theta, saa = TSEB.met.calc_sun_angles(
    np.full_like(ec['LAI'].values, lat),
    np.full_like(ec["LAI"].values, lon),
    np.full_like(ec["LAI"].values, stdlon),
    ec['TIMESTAMP'].dt.dayofyear.values,
    ec['TIMESTAMP'].dt.hour.values + ec['TIMESTAMP'].dt.minute.values / 60.)

F = ec["LAI"].values / ec["F_C"].values
omega_0 = TSEB.CI.calc_omega0_Kustas(
    ec["LAI"].values, ec["F_C"].values, x_LAD=ec["X_LAD"].values)

omega = TSEB.CI.calc_omega_Kustas(
    omega_0, np.minimum(theta, 90), w_C=w_c)

lai_eff = F * omega


# Estimate the leaf and soil spectra
ec = components_spectra(ec)
# We append the VIS and PAR spectrum to be computationally more efficient in Numpy
rho_leaf = np.array((ec["RHO_LEAF_VIS"].values, ec["RHO_LEAF_NIR"].values))
tau_leaf = np.array((ec["TAU_LEAF_VIS"].values, ec["TAU_LEAF_NIR"].values))
rho_soil = np.array((ec["RHO_SOIL_VIS"].values, ec["RHO_SOIL_NIR"].values))


# In order to esimate Estimates the direct and diffuse solar radiation based on the Weiss and Norman model
difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(ec["SW_IN_F"].values,
                                                        theta,
                                                        press=np.full_like(theta, 1013.15))
par_dir = fvis * (1. - difvis) * ec["SW_IN_F"].values
nir_dir = fnir * (1. - difnir) * ec["SW_IN_F"].values
par_dif = fvis * difvis * ec["SW_IN_F"].values
nir_dif = fnir * difnir * ec["SW_IN_F"].values


# calculate absorptivity
amean = 1.0 - rho_leaf - tau_leaf
amean_sqrt = np.sqrt(amean)


# D I F F U S E   C O M P O N E N T S
# Integrate to get the diffuse transmitance

taud = 0
for angle in range(0, 90, 5):
    angle = np.radians(angle)
    akd = (np.sqrt(ec["X_LAD"].values**2 + np.tan(theta)**2) \
           / (ec["X_LAD"].values + 1.774 * (ec["X_LAD"].values + 1.182)**-0.733))  # Eq. 15.4
    taub = np.exp(-akd * ec["LAI"].values)
    taud += taub * np.cos(angle) * np.sin(angle) * np.radians(5)

taud = 2.0 * taud

# Diffuse light canopy reflection coefficients  for a deep canopy
akd = -np.log(taud) / ec["LAI"].values
rcpy= (1.0 - amean_sqrt) / (1.0 + amean_sqrt)  # Eq 15.7
rdcpy = 2.0 * akd * rcpy / (akd + 1.0)  # Eq 15.8

# Diffuse canopy transmission and albedo coeff for a generic canopy (visible)
expfac = amean_sqrt * akd * ec["LAI"].values
neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
xnum = (rdcpy * rdcpy - 1.0) * neg_exp
xden = (rdcpy * rho_soil - 1.0) + rdcpy * (rdcpy - rho_soil) * d_neg_exp
taudt = xnum / xden  # Eq 15.11
fact = ((rdcpy - rho_soil) / (rdcpy * rho_soil - 1.0)) * d_neg_exp
albd = (rdcpy + fact) / (1.0 + rdcpy * fact)  # Eq 15.9

# B E A M   C O M P O N E N T S
# Calculate canopy beam extinction coefficient
# Direct beam extinction coeff (spher. LAD)
akb = (np.sqrt(ec["X_LAD"].values**2 + np.tan(theta)**2) \
       / (ec["X_LAD"].values + 1.774 * (ec["X_LAD"].values + 1.182)**-0.733))  # Eq. 15.4

# Direct beam canopy reflection coefficients for a deep canopy
rbcpy = 2.0 * akb * rcpy / (akb + 1.0)  # Eq 15.8
# Beam canopy transmission and albedo coeff for a generic canopy (visible)
expfac = amean_sqrt * akb * lai_eff
neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
xnum = (rbcpy * rbcpy - 1.0) * neg_exp
xden = (rbcpy * rho_soil - 1.0) + rbcpy * (rbcpy - rho_soil) * d_neg_exp
taubt = xnum / xden  # Eq 15.11
fact = ((rbcpy - rho_soil) / (rbcpy * rho_soil - 1.0)) * d_neg_exp
albb = (rbcpy + fact) / (1.0 + rbcpy * fact)  # Eq 15.9

taubt, taudt, albb, albd, rho_soil = map(np.array,
                                         [taubt, taudt, albb, albd, rho_soil])

taubt[np.isnan(taubt)] = 1
taudt[np.isnan(taudt)] = 1
albb[np.isnan(albb)] = rho_soil[np.isnan(albb)]
albd[np.isnan(albd)] = rho_soil[np.isnan(albd)]

# Compute the canopy and soil net radiation using Cambpell RTM
sn_c = ((1.0 - taubt[0]) * (1.0- albb[0]) * par_dir
            + (1.0 - taubt[1]) * (1.0- albb[1]) * nir_dir
            + (1.0 - taudt[0]) * (1.0- albd[0]) * par_dif
            + (1.0 - taudt[1]) * (1.0- albd[1]) * nir_dif)

sn_s = (taubt[0] * (1.0 - ec["RHO_SOIL_VIS"].values) * par_dir
            + taubt[1] * (1.0 - ec["RHO_SOIL_NIR"].values) * nir_dir
            + taudt[0] * (1.0 - ec["RHO_SOIL_VIS"].values) * par_dif
            + taudt[1] * (1.0 - ec["RHO_SOIL_NIR"].values) * nir_dif)

print("Finished the computation of net shortwave radiation")
```

### Evaluate net shortwave radiation
And by running the cell below we validate those estimates with the hourly measurements in the eddy covariance flux tower

```{code-cell} ipython3
from model_evaluation import double_collocation as dc
import tabulate

# Avoid pandas frabmentation warnings
ec = ec.copy()

daytime = ec["SW_IN_F"] > 100

sn_model = sn_c + sn_s
sn_obs = ec["SW_IN_F"].values - ec["SW_OUT"].values

bias, mae, rmse = dc.error_metrics(
    sn_obs[daytime], sn_model[daytime])
cor, *_, d = dc.agreement_metrics(
    sn_obs[daytime], sn_model[daytime])
n, mean_obs, mean_pre, std_obs, std_pre = dc.descriptive_stats(
        sn_obs[daytime], sn_model[daytime])

error_df = {"N": [], "Obs.": [],
                "RMSE": [], "bias": [],
                "scale": [], "r": [], "d": [],
                }
error_df["N"].append(int(n))
error_df["Obs."].append(np.round(mean_obs, 2))
error_df["RMSE"].append(np.round(rmse, 2))
error_df["bias"].append(np.round(bias, 2))
error_df["scale"].append(np.round(std_obs / std_pre, 2))
error_df["r"].append(np.round(cor, 2))
error_df["d"].append(np.round(d, 2))
error_df = pd.DataFrame(error_df, index=["Sn"])


display(error_df)
fig = go.Figure()
fig.add_trace(go.Scattergl(x=sn_model[daytime], y=sn_obs[daytime], 
                         name="Shortwave net radiation", mode="markers"))
fig.add_trace(go.Scatter(x=[0, 1000], y=[0, 1000], mode="lines", name="1:1 line", line={"color": "black", "dash": "dash"}))
fig.update_layout(title_text=f"Observed vs. Estimated net radiation at {site}",
                  yaxis_range=[0, 1000], xaxis_range=[0, 1000],
                  xaxis_title="Estimated (W m-2)", yaxis_title="Observed (W m-2)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=sn_model[daytime], y=sn_obs[daytime], 
                         name="Sn", mode="markers"))
fig.add_trace(go.Scatter(x=[0, 1200], y=[0, 1200], mode="lines", name="1:1 line", line={"color": "black", "dash": "dash"}))
fig.update_layout(title_text=f"Observed vs. Estimated net shortwave radiation at {site}",
                  yaxis_range=[0, 1200], xaxis_range=[0, 1200],
                  xaxis_title="Estimated (W m-2)", yaxis_title="Observed (W m-2)")
```

# Net longwave radiation
The net longwave radiation a priori is much simpler as we assume that all emitted and indident longwave radiation is diffuse. In pyTSEB we follow a similar approach as the one described in {cite:t}`https://doi.org/10.1016/S0168-1923(99)00005-2` for partially vegetated surfaces, which also partitions net longwave radiation into the soil ($L_{n,S}$) and the canopy components ($L_{n,C}$). The only difference between pyTSEB and the formulation of {cite:p}`https://doi.org/10.1016/S0168-1923(99)00005-2` is that we incorporate longwave scattering between the soil and the canopy:


:::{math}
L_{n,S} &= \epsilon_S \tau_{LAI} L^down + \epsilon_S \left(1.0 - \tau_{LAI} L^down \right) L_C - L_S\\
L_{n,C} &= \epsilon_C \left(1.0 - \tau_{LAI} L^down\right) \left(L_{sky} + L_S\right) - 2.0 \left(1.0 - \tau_{LAI}\right) L_C
:::

where $L^down$ is the downwelling atmospheric longwave irradiance, $L_S$ and $L_C$ are the longwave emission of respectively soil and canopy layers, $tau_{LAI}$ is analogous to the shortwave diffuse transmittance and thus it can be calculated from Eq. [](#eq-kd), $\epsilon_S$ is soil emisivity and $\epsilon_C$ can be derived from Eq. [](#eq-rho-dir) ($\epsilon_C = 1 - \rho_{C,DIF,l}$) and leaf emissivity.

:::{note}
Even we use a modified version of {cite:t}`https://doi.org/10.1016/S0168-1923(99)00005-2`, the pyTSEB package also includes the original formulation [here](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/net_radiation.py#L244)
:::

:::{attention}

However, $L_S$ and $L_C$ depend respectively on soil and canopy temperature, which at this stage is an unknown. For that reason **the application of this equation is done internaly when deriving the component temperature and sensible heat fluxes**
:::

```{code-cell} ipython3
from pyTSEB.net_radiation import calc_spectra_Cambpell

def calc_L_n_Campbell(T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD=1):
    ''' Net longwave radiation for soil and canopy layers

    Estimates the net longwave radiation for soil and canopy layers unisg based on equation 2a
    from [Kustas1999]_ and incorporated the effect of the Leaf Angle Distribution based on [Campbell1998]_

    Parameters
    ----------
    T_C : float
        Canopy temperature (K).
    T_S : float
        Soil temperature (K).
    L_dn : float
        Downwelling atmospheric longwave radiation (w m-2).
    lai : float
        Effective Leaf (Plant) Area Index.
    emisVeg : float
        Broadband emissivity of vegetation cover.
    emisGrd : float
        Broadband emissivity of soil.
    x_LAD: float, optional
        x parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_LAD=1 for a spherical LAD.

    Returns
    -------
    L_nC : float
        Net longwave radiation of canopy (W m-2).
    L_nS : float
        Net longwave radiation of soil (W m-2).

    References
    ----------
    .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''
    def calc_stephan_boltzmann(t_k):
        '''Calculates the total energy radiated by a blackbody.
    
        Parameters
        ----------
        t_k : float
            body temperature (Kelvin)
    
        Returns
        -------
        M : float
            Emitted radiance (W m-2)'''
        # Stephan Boltzmann constant (W m-2 K-4)
        sb = 5.670373e-8 
        m = sb * t_k**4
        return m
        
    # calculate long wave emissions from canopy, soil and sky
    L_C = emisVeg * calc_stephan_boltzmann(T_C)
    L_S = emisGrd * calc_stephan_boltzmann(T_S)
    # Calculate the canopy spectral properties
    _, albl, _, taudl = calc_spectra_Cambpell(lai,
                                              0,
                                              1.0 - emisVeg,
                                              0,
                                              1.0 - emisGrd,
                                              x_lad=x_LAD,
                                              lai_eff=None)

    # calculate net longwave radiation divergence of the soil
    L_nS = emisGrd * taudl * L_dn + emisGrd * (1.0 - taudl) * L_C - L_S
    L_nC = (1 - albl) * (1.0 - taudl) * (L_dn + L_S) - 2.0 * (1.0 - taudl) * L_C
    return L_nC, L_nS

print("Parsed net longwave radiatin function correctly, you can continue to next cell")
```

# Net shortwave radiation in 3SEB
In this case we will work with the "dehesa" of Majadas de Tiétar

```{code-cell} ipython3
site = "ES-LMa"

# fractional cover
f_c_ov	= 0.2  # overstory fractional cover (e.g., clumped trees)
f_c_un	= 1.  # understory fractional cover (e.g., homogeneous grass)
# canopy height
h_c_ov	= 8.0  # Canopy height (m) of overstory (e.g. trees)
h_c_un	= 0.25 

# Thermal spectra
e_c_ov = 0.99  # Overstory Leaf emissivity
e_c_un = 0.97  # Understory Leaf emissivity
e_s = 0.94  # Soil emissivity
```

## Decompose LAI between the overstory and understory

```{code-cell} ipython3
# Set the input files based on the chosen site
bio_filename = bio_dir / f"{site}_HLS-l2c.csv"
ec_filename = ec_dir / f"FLX_{site}_FLUXNET_SUBSET_HH.csv"
print(f"Biophysical traits file path is {bio_filename}")
print(f"EC file path is {ec_filename}")

# Read the fluxnet EC table and preprocess
ec =  pd.read_csv(ec_filename, sep=',', na_values=-9999)
ec["TIMESTAMP_START"] = pd.to_datetime(
    ec["TIMESTAMP_START"], format="%Y%m%d%H%M")

ec["TIMESTAMP_END"] = pd.to_datetime(
    ec["TIMESTAMP_END"], format="%Y%m%d%H%M")

new_cols = {}
new_cols["TIMESTAMP"] = pd.to_datetime(
        ec[["TIMESTAMP_START", "TIMESTAMP_END"]].mean(axis=1))

new_cols["DATE"] = new_cols["TIMESTAMP"].dt.date
ec = pd.concat([ec, pd.DataFrame(new_cols)], axis=1)

# Read the satellite biophysical traits table
bio = pd.read_csv(bio_filename, sep=";", na_values=-9999)
bio["DATE"] = pd.to_datetime(bio["DATE"], format="%Y-%m-%d").dt.date

# Merge both tables by date
ec = ec.merge(bio, on="DATE")

# Ecosystem LAI 
lai_eco = ec['LAI'].values

# Compute the Cambpell chi LIDF parameter from Campbell mean leaf angle
ec.loc[:, "X_LAD"] = TSEB.rad.leafangle_2_chi(ec["LEAF_ANGLE"].values)

# Set overstory fraction
ec.loc[:, "F_C"] = f_c_ov

# initialize overstory LAI_ov
lai_ov = np.zeros_like(lai_eco)

years = pd.to_datetime(ec['DATE'].values).year

# get lai_ov for each year 
for year in np.unique(years):
    lai_year = ec['LAI'].values[years == year]
    # lowest 10% percentile as estime of LAI_ov (during summer period)
    lai_min = np.percentile(lai_year,10)
    lai_ov[years == year] = lai_min
    
# overstory LAI
ec['LAI_ov'] = lai_ov
# understory LAI 
lai_un = (ec['LAI'].values - ec['LAI_ov'].values)/(1-f_c_ov) # (account for LAI_un below tree canopy)
lai_un[lai_un<0.3] = 0.3
ec['LAI_un'] = lai_un


print('Plotting LAI decomposition time series [...]')
fig = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    horizontal_spacing=0.01,
                   subplot_titles=("LAI", "Leaf pigments"))

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["LAI"], 
                         name="LAI_eco", mode="lines", line={"color": "black"}),
              row=1, col=1)

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["LAI_ov"], 
                         name="LAI_ov", mode="lines", line={"color": "green", "dash": "dash"}),
              row=1, col=1)

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["LAI_un"], 
                         name="LAI_un", mode="lines", line={"color": "indianred", "dash": "dash"}),
              row=1, col=1)

fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_layout(title_text=f"Biophysical traits timeseries in {site}")
```

## Calculate shortwave net radiation partitioning

```{code-cell} ipython3

# The ASCII table is missing the solar angles, so we will use the calc_sun_angles function of TSEB to compute the angles based on site location and timestamp
# The standard meridian time zone (15 deg. per hour)
stdlon = 15 * utc_offset
theta, saa = TSEB.met.calc_sun_angles(
    np.full_like(ec['LAI'].values, lat),
    np.full_like(ec["LAI"].values, lon),
    np.full_like(ec["LAI"].values, stdlon),
    ec['TIMESTAMP'].dt.dayofyear.values,
    ec['TIMESTAMP'].dt.hour.values + ec['TIMESTAMP'].dt.minute.values / 60.)

# Calculate clumping index for overstory
omega0_ov = TSEB.CI.calc_omega0_Kustas(lai_ov, f_c_ov, x_LAD=1, isLAIeff=True)
Omega_ov = TSEB.CI.calc_omega_Kustas(omega0_ov, np.minimum(theta, 90), w_C=w_c)
F_ov = lai_ov/f_c_ov # Real LAI_ov
# effective LAI (tree layer)
lai_ov_eff =  F_ov * Omega_ov

# Estimates the direct and diffuse solar radiation
difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(ec["SW_IN_F"].values,
                                                        theta,
                                                        press=np.full_like(theta, 1013.15))
par_dir = fvis * (1. - difvis) * ec["SW_IN_F"].values
nir_dir = fnir * (1. - difnir) * ec["SW_IN_F"].values
par_dif = fvis * difvis * ec["SW_IN_F"].values
nir_dif = fnir * difnir * ec["SW_IN_F"].values

# Estimate the leaf and soil spectra
ec = components_spectra(ec)

# We append the VIS and PAR spectrum to be computationally more efficient in Numpy
rho_leaf = np.array((ec["RHO_LEAF_VIS"].values, ec["RHO_LEAF_NIR"].values))
tau_leaf = np.array((ec["TAU_LEAF_VIS"].values, ec["TAU_LEAF_NIR"].values))
rho_soil = np.array((ec["RHO_SOIL_VIS"].values, ec["RHO_SOIL_NIR"].values))

# net radiation transmission through the three sources
sn_c_ov, sn_s, sn_c_un = py3seb.calc_Sn_Campbell(ec['LAI_ov'].values, # LAI of overstory
                                                              ec['LAI_un'].values, # LAI of understory
                                                              theta, # sun zenigth angle
                                                              par_dir + nir_dir, # direct incoming irradiance
                                                              par_dif + nir_dif,  # diffuse incoming irradiance
                                                              fvis, # fration of total visible radiation
                                                              fnir, # fration of total NIR radiation
                                                              ec["RHO_LEAF_VIS"].values, # Overstory Broadband leaf reflectance in the visible region (400-700nm)
                                                              ec["RHO_LEAF_VIS"].values, # Understory Broadband leaf reflectance in the visible region (400-700nm)
                                                              ec["TAU_LEAF_VIS"].values, # Overstory Broadband leaf transmittance in the visible region (400-700nm)
                                                              ec["TAU_LEAF_VIS"].values, # Understory Broadband leaf transmittance in the visible region (400-700nm)
                                                              ec["RHO_LEAF_NIR"].values, # Overstory Broadband leaf reflectance in the NIR region (700-2500nm)
                                                              ec["RHO_LEAF_NIR"].values, # Understory Broadband leaf reflectance in the NIR region (700-2500nm)
                                                              ec["TAU_LEAF_NIR"].values, # Overstory Broadband leaf transmittance in the NIR region (700-2500nm)
                                                              ec["TAU_LEAF_NIR"].values, # Understory Broadband leaf transmittance in the NIR region (700-2500nm)
                                                              ec["RHO_SOIL_VIS"].values, # Soil Broadband reflectance in the visible region (400-700nm)
                                                              ec["RHO_SOIL_NIR"].values, # Soil Broadband reflectance in the NIR region (700-2500nm)
                                                              h_c_ov, # overstory canopy height
                                                              h_c_ov * 0.5, # ratio of hc where leaf canopy begins in overstory layer
                                                              w_c, # width to height ratio of overstory
                                                              f_c_ov, # Fractional cover of overstory
                                                              x_LAD=1, # Overstory x parameter for the ellipsoildal Leaf Angle Distribution function of Campbell 1988
                                                              x_LAD_sub=1, # Understory x parameter for the ellipsoildal Leaf Angle Distribution function of Campbell 1988
                                                              LAI_eff=lai_ov_eff, # Effective LAI of overstory
                                                              LAI_eff_sub=ec['LAI_un'].values) # Effective LAI of understory


sn_c_ov[~np.isfinite(sn_c_ov)] = 0
sn_s[~np.isfinite(sn_s)] = 0
sn_c_un[~np.isfinite(sn_c_un)] = 0

# evaluate against tower measurements
daytime = ec["SW_IN_F"] > 100

sn_model = sn_c_ov + sn_c_un + sn_s
sn_obs = ec["SW_IN_F"].values - ec["SW_OUT"].values

bias, mae, rmse = dc.error_metrics(
    sn_obs[daytime], sn_model[daytime])
cor, *_, d = dc.agreement_metrics(
    sn_obs[daytime], sn_model[daytime])
n, mean_obs, mean_pre, std_obs, std_pre = dc.descriptive_stats(
        sn_obs[daytime], sn_model[daytime])

error_df = {"N": [], "Obs.": [],
                "RMSE": [], "bias": [],
                "scale": [], "r": [], "d": [],
                }
error_df["N"].append(int(n))
error_df["Obs."].append(np.round(mean_obs, 2))
error_df["RMSE"].append(np.round(rmse, 2))
error_df["bias"].append(np.round(bias, 2))
error_df["scale"].append(np.round(std_obs / std_pre, 2))
error_df["r"].append(np.round(cor, 2))
error_df["d"].append(np.round(d, 2))
error_df = pd.DataFrame(error_df, index=["Sn"])


display(error_df)
print('Plotting modelled shortwave irradiance evaluation [...]')

fig = go.Figure()
fig.add_trace(go.Scattergl(x=sn_model[daytime], y=sn_obs[daytime], 
                         name="Shortwave net radiation", mode="markers"))
fig.add_trace(go.Scatter(x=[0, 1000], y=[0, 1000], mode="lines", name="1:1 line", 
                         line={"color": "black", "dash": "dash"}))
fig.update_layout(title_text=f"Observed vs. Estimated net radiation at {site}",
                  yaxis_range=[0, 1000], xaxis_range=[0, 1000],
                  xaxis_title="Estimated (W m-2)", yaxis_title="Observed (W m-2)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=sn_model[daytime], y=sn_obs[daytime], 
                         name="Sn", mode="markers"))
fig.add_trace(go.Scatter(x=[0, 1200], y=[0, 1200], mode="lines", name="1:1 line", 
                         line={"color": "black", "dash": "dash"}))
fig.update_layout(title_text=f"Observed vs. Estimated net shortwave radiation from 3-source model at {site}",
                  yaxis_range=[0, 1200], xaxis_range=[0, 1200],
                  xaxis_title="Estimated (W m-2)", yaxis_title="Observed (W m-2)")
```

# Conclusions

* In this notebook we learn the basics of the low level code of pyTSEB to run TSEB-PT using as input one single value
* A very similar approach would be done for running TSEB-PT with an image or a tabulated timeseries.
* The pyTSEB package includes Jupyter Notebooks that allow using a Graphical User Interface to parse all TSEB inputs
* In an upcoming session we will show you how to derive all inputs required for TSEB using Copernicus data from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
