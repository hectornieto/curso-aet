---
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

# Summary
This interactive Jupyter Notebook has the objective of showing the implemenation of the TSEB-PT and 3SEB-PT models using the [pyTSEB package](https://github.com/hectornieto/pyTSEB) and [py3SEB package](https://github.com/VicenteBurchard/py3SEB).

+++

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
from pyTSEB import TSEB
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
print("Libraries imported!")
```

# The TSEB model

The Two-Source Energy Balance (TSEB) model [Norman et al. (1995)](https://doi.org/10.1016/0168-1923(95)02265-Y) was specifically designed to account for directionality of radiometric temperature measurements and to avoid the use of any empirical "excess" resistance terms in the model. To achieve this, both the heat and water fluxes are separated into a soil and canopy layer, with a set of resistances set in series.

:::{note}
In [Norman et al. (1995)](https://doi.org/10.1016/0168-1923(95)02265-Y) the original resistance scheme was in parallel
:::
    
:::{figure} ./input/figures/tseb.jpg
:alt:Two Source Energy Balance model
:name:tseb-model
Schematic diagram of the TSEB model resistance network for sensible heat flux and the basic set of equations used to obtain an iterative solution. Source: [Kustas et al. (2018)](https://doi.org/10.1175/BAMS-D-16-0244).
:::

Energy fluxes are therefore split into soil and canopy, considering the conservation of energy:
    
:::{math}
:label:eq-eb
  R_{n} & \approx H + \lambda E + G\\
  R_{n,S} & \approx H_{S} + \lambda E_{S} + G\\
  R_{n,C} & \approx H_{C} + \lambda E_{C}\label{eq:Energy_Balance_TSEB}
:::

with $R_n$ being the net radiation, $H$ the sensible heat flux, $\lambda E$ the latent heat flux or evapotranspiration, and $G$ the soil heat flux (all fluxes are expressed in W m$^{-2}$). The approximation in Eq. [](#eq-eb) reflects additional components of the energy balance that are usually neglected, such as heat advection, storage of energy in the canopy layer or energy for the fixation of CO$_2$, which are not computed by the model.
    
Remote sensing Energy Balance Models, REBMs, rely on the ability of the radiometric information to estimate, net radiation, soil heat flux and sensible heat flux [(Norman et al., 1995)](https://doi.org/10.1016/0168-1923(95)02265-Y). To see how pyTSEB works we will run several pieces of code to estimate each of the components of the energy balance.

::::{seealso}
:class:dropdown
This is a full description of the EC column fields:

[](./fluxnet2015_variables.md)
::::

# Step-by-step run of TSEB

## Select a site

```{code-cell} ipython3
# Set the canopy and eddy covariance folders
input_dir = Path().absolute() / "input"
ec_dir = input_dir / "eddy_covariance"
sites = ec_dir.glob("*_SUBSET_HH*.csv")
sites = [i.stem.split("_")[1] for i in sites]

# Site metadata file
site_metadata_file = input_dir / "sites.csv"
site_metadata = pd.read_csv(site_metadata_file, sep=";")

#valid = np.in1d(site_metadata["SITE_ID"], sites)
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

print('Plotting biophysical traits and pigment time series [...]')
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

## Net shortwave radiation
To compute the net shortwave radiation and its partitioning between the canopy and the soil we are using the model of [Campbell and Norman, 1998, Chapter 15](10.1007/978-1-4612-1626-1). The key aspect of this model is the calculation of the trasmitted shortwave radiation throught the canopy ($\Tau_C$)

:::{seealso}
:class:dropdown
The full description of the radiation model can be found in [Campbell and Norman, 1998, Chapter 15](10.1007/978-1-4612-1626-1_15)
:::

The transmittance of shortwave radiation through the canopy depends on the wavelength due to vegetation absorbing a greater portion of the photosynthetically active radiation (PAR, 400–700 nm spectrum) than near-infrared radiation (NIR, 700–2500 nm spectrum) wavelengths. ($\Tau_C$ is partitioned into two components (direct/diffse) and in two spectral band (PAR/NIR)

:::{note}
:class:dropdown
We could include more spectral bands, such as (PAR/NIR/SWIR), with a slight increase of computational cost. 
:::

The direct-beam spectral (transmittance ($\tau_{C,DIR,\lambda}$) at a given solar zenith angle ($\theta_S$) is calculated following the equations of Campbell and Norman (1998) for a single layer crop:

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

Finally, the canopy beam extinction $\kappa_b\left(\psi\right)$ is calculated based on the ellipsoidal LIDF of [Campbell (1990)](10.1016/0168-1923(90)90030-A)

:::{math}
:label:eq-kbe
\kappa_b\left(\theta_S\right)=\frac{\sqrt{\chi^2+ \tan^2\theta_S}}{\chi+1.774\left(\chi+1.182\right)^{-0.733}}
:::
with $\chi$ the [Campbell (1990)](10.1016/0168-1923(90)90030-A) LIDF parameter
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

Therefore as inputs for computin net shortwave radiation we need to provide as input forcing the solar irradiance, direct and diffuse in the PAR and NIR, and then as main input the $LAI$, followed by soil and leaf spectra and [Campbell (1990)](10.1016/0168-1923(90)90030-A) LIDF parameter ($\chi$). 

:::{seealso}
The code for estimating the diffuse and bean radiation is based on [Weiss and Norman (1985)](10.1016/0168-1923(85)90020-6) and can be found at the [pyTSEB GitHub repository](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/net_radiation.py#L62)

The code for running the [Cambpell and Norman (1998)](10.1007/978-1-4612-1626-1_15) model can be found at the pyTSEB GitHub repository: [canopy spectral properties](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/net_radiation.py#L439) and [estimation of net shortwave radiation](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/net_radiation.py#L546C5-L546C21)
:::

### Retrieval of broadband spectral properties
For estimating leaf spectra we will use the leaf traits retrievals from the satellite biophysical traits at `input/canopy`. With the information of these traits we could run in forward mode the ProspectD leaf RTM to get a spectral reflectance and transmittance that could then be integrated to the required broadband regions. However, running ProspectD for a large array of pixels is too computationally expensive. For that reason we have developed a ProspectD emulator [(Guzinski et al. 2021)](10.1109/JSTARS.2021.3122573): we generated broadband leaf reflectances and transmittances in the PAR and NIR from a large range of ProspectD simulations, covering all plausible range of leaf traits, which then will be used to train another random forest model that will relate the leaf traits to the broadband PAR and NIR leaf reflectance and transmittance.

Soil reflectance has a smaller role in most situations, since the radiation reaching the ground is smaller due to the interception of light by the canopy. However, in sparse and semi-arid conditions, where vegetation is scarce or even not present, soil albedo plays obviously a significant role. Furthermore, these semi-arid areas are usually characterized by brighter soils since they are composed by sands, salts and with a small fraction of organic matter. For that reason in [Guzinski et al. (2021)](10.1109/JSTARS.2021.3122573) we developed a method to unmix the broadband soil reflectance ($\rho_{soil, \lambda}$) from the satellite observed reflectances (Eq. \ref{eq:rho-soil})

:::{math}
:label:eq:rho-soil
\rho_{soil, \lambda} =  \frac{\rho_{surface, \lambda} - \left[1 - P_0\left(\theta_v\right)\right] \rho_{leaf, \lambda}}{ P_0\left(\theta_v\right)} 
:::
where $P_0\left(\theta_v\right)$ is the canopy gap fraction at the satellite observation angle, computed by the Beer-Lambert law ($\exp\left[-\kappa\left(\theta_v\right) \mathrm{LAI}\right]$). $\rho_{surface, \lambda}$ is the broadband surface reflectance, and $\rho_{leaf, \lambda}$ is the leaf broadband reflectance.

#### Run the cell below to parse the functions described above

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

### Estimate net shortwave radiation based on biophyisical traits and incoming irradiance
Now we can execute the cell below to run the [Campbell and Norman (1998)](10.1007/978-1-4612-1626-1_15) radiative transfer model implemente in pyTSEB using the site selected in [](#Select-a-site) section.

Besides of LAI, its leaf inclination distribution function and the spectral properties, in TSEB we also make use of the fractional cover ($f_c$) and canopy shape ($w_c/h_c$) for clumped canopies. These are used to computed a clumping index that converts the LAI to effective values based on the solar incidence angle (since in clumpled canopies shading can play a significant role)

:::{warning}
Both $f_c$ and $w_c/h_c$ are structural variables in TSEB and they cannot be operationally retrieved from satellite data. The typical `FCOVER` product **should not** be used as surrogate for $f_c$ since `FCOVER` not only represents the canopy clumpiness but also the canopy gap fraction.

In operational implementations, we derived a static $f_c$ value only for clumped canopies using prescribed values based on land cover/plant functional type (See {ref}`tab:lc-params` below)
:::

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
print('Plotting modelled shortwave irradiance evaluation [...]')

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

## Net longwave radiation
The net longwave radiation a priori is much simpler as we assume that all emitted and indident longwave radiation is diffuse. In pyTSEB we follow a similar approach as the one described in [Kustas and Norman (1999)](https://doi.org/10.1016/S0168-1923(99)00005-2) for partially vegetated surfaces, which also partitions net longwave radiation into the soil ($L_{n,S}$) and the canopy components ($L_{n,C}$). The only difference between pyTSEB and the formulation of [Kustas and Norman (1999)](https://doi.org/10.1016/S0168-1923(99)00005-2) is that we incorporate longwave scattering between the soil and the canopy:


:::{math}
L_{n,S} &= \epsilon_S \kappa_l LAI L_{sky} + \epsilon_S \left(1.0 - \kappa_l LAI * L_{sky} \right) L_C - L_S\\
L_{n,C} &= \epsilon_C \left(1.0 - \kappa_l LAI * L_{sky}\right) \left(L_{sky} + L_S\right) - 2.0 \left(1.0 - \kappa_l LAI\right) L_C
:::

where $L_{sky}$ is the downwelling atmospheric longwave irradiance, $L_S$ and $L_C$ are the longwave emission of respectively soil and canopy layers, $\kappa_l$ is analogous to the shortwave diffuse transmittance ($\kappa_d$) and thus it can be calculated from Eq. [](#eq-kd), $\epsilon_S$ is soil emisivity and $\epsilon_C$ can be derived from Eq. [](#eq-rho-dir) ($\epsilon_C = 1 - \rho_{C,DIF,l}$) and leaf emissivity.

:::{note}
Even we use a modified version of [Kustas and Norman (1999)](https://doi.org/10.1016/S0168-1923(99)00005-2), the pyTSEB package also includes the original formulation [here](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/net_radiation.py#L244)
:::

However, $L_S$ and $L_C$ depend respectively on soil and canopy temperature, which at this stage is an unknown. For that reason the application of this equation is done internaly when deriving the component temperature and sensible heat fluxes (see the upcoming Section [](#sec-h))

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

## Soil heat flux
Soil heat flux is usually estimated as a fraction of the soil net radiation $R_{n,S}$. Such fraction can be assumed constant along the day (e.g. $G=0.35 R_{n,S}$) or can change along the day, such as the sinusoidal function proposed by [Santanello and Friedl, 2003](10.1175/1520-0450(2003)042<0851:DCISHF>2.0.CO;2). 

Since the computation of net longwave radiation is dependant of getting estimates of $T_C$ and $T_S$ and thus is computed internally in the TSEB modules during the iteration that will be discussed in the next section, $G$ is also computed within the TSEB function, being iteratevely recomputed every time a new $T_C$ and $T_S$ is obtained. The different TSEB versions in pyTSEB accept computing $G$ in four different ways:

1. As a constant ratio of $R_{n,S}$ [function `calc_G_ratio`](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L2809)
2. As time dependent ratio of $R_{n,S}$ according to a sinusoidal function [function `calc_G_time_diff`](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L2729)
3. As time dependent ratio of $R_{n,S}$ according to a double assymetric sigmoid function [](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L2769)
4. As precomputed/prescribed values
 
:::{note}
This last option can be chosen to force the use of measured $G$ when running TSEB, which sometimes can be useful when testing and evaluating new (sub)modules into TSEB. 
:::



## Surface roughness
In addition, REBMs need additional ancillary inputs, such as air temperature, wind speed and canopy height or roughness. The latter accounts for the efficiency in the turbulent transport of heat and water between the land surface and the overlying air [Raupach (1994)](10.1007/BF00709229)[Alfieri et al. 2019](10.1007/s00271-018-0610-z) (you can revisity the  [102-Turbulence_and_sensible_heat_flux](./102-Turbulence_and_sensible_heat_flux.ipynb) notebook). Specifically for pyTSEB, vegetation structure and density are also important for estimating wind and radiation extinction through the canopy layer affecting the radiation partitioning [Parry et al. (2019)](10.1007/s00271-019-00621-x) and turbulent transport of  momentum, heat  and water vapour in the canopy air space [Nieto et al. (2019)](10.1007/s00271-018-0611-y).
   
In pyTSEB, depending on the plant functional trait, roughness can be estimated using different methods. For short and herbaceous crops, the typical ratio of canopy height ($h_c$) is applied:

:::{math}
:label:eq-roughness-short
d_0 &= 2/3 h_c
z_{0m} & = 1/8 h_c 
:::
where $d_0$ is the zero-plane displacement height and $z_{0m}$ is the roughness length for momentum.

For tall and woody canopies, in which leaf density and coverage can play a more importan role, pyTSEB can adopt the formulation of [Schaudt and Dickinson (2000)](10.1016/S0168-1923(00)00153-2), which at the same time it relies on the model by [Raupach (1994)](https://doi.org/10.1007/BF00709229)

### Estimate the structural variables based on Earth Observation LAI timeseries
A Land cover map can be used to assign ET model parameters, which are difficult to estimate directly from other Earth Observation data [Guzinski et al. (2021)](10.1109/JSTARS.2021.3122573). Those parameters, and values assigned to different land cover classes, are listed in Table \ref{tab:lc-params}. Nevertheless, the TSEB-PT model requires all of the parameters, apart from stomatal resistance, while ETLook requires only vegetation height and stomatal resistance . Values of other parameters were adapted from \citet{guzinski_modelling_2020}. In such framework, and for annual plant functional types, canopy height is computed considering its growth as Eq.  {eq}`eq:hc-lai`:

:::{math}
:label:eq:hc-lai
    h_c = h_{min} + \left(h_{max} - h_{min}\right)\mathrm{min}\left(\frac{\mathrm{LAI}}{\mathrm{LAI}_{max}},\,1\right)
:::

However, in this case, in order to better adjust the obstacle (canopy) height in forests between the different climatic conditions, we propose to use the University of Maryland's Global Forest Canopy Height, 2019. A global 30~m spatial resolution global forest canopy height map developed by combining the Global Ecosystem Dynamics Investigation (GEDI) LiDAR and Landsat time-series [(Potapov et al. 2021)](10.1016/j.rse.2020.112165).

:::{table}
:name:tab:lc-params
Land Cover Look-Up-Table for ancillary and structural parameters required by TSEB model, adapted from [Guzinski et al (2025)](10.5194/egusphere-2025-4342), $h_{min}$ (m) is the minimum canopy height; $h_{max}$ (m) is the maximum canopy height occurring when leaf area index (LAI) reaches its optimal maximum $\mathrm{LAI}_{max}$; $f_c$ is the at-Nadir fraction of the ground occupied by a clumped canopy ($f_c=1$ for a homogeneous canopy); $w_c/h_c$ is the canopy shape parameter, representing the canopy width to canopy height ratio; $l_w$ (m) is the average leaf size.

IGBP  | $h_{min}$ | $h_{max}$ | $\mathrm{LAI}_{max}$ | $f_c$	| $w_c/h_c$	| $l_w$
:---  | ---:      |  ---:     | ---:                 | ---:     | ---:      | ---:     
ENF | GEDI | GEDI | N/A | 1 | 0.5 | 0.01
DNF | GEDI | GEDI | N/A | 1 | 0.5 | 0.01
EBF | GEDI | GEDI | N/A | 1 | 1 | 0.05
DBF | GEDI | GEDI | N/A | 1 | 1 | 0.1
MF | GEDI | GEDI | N/A | 1 | 1 | 0.1
OSH | 1 | 1 | N/A | 0.2 | 1 | 0.05
CSH | 2 | 2 | N/A | 1 | 1 | 0.05
SAV | 8 | 8 | N/A | 0.2 | 1 | 0.05
WSA | 8 | 8 | N/A | 0.2 | 1 | 0.05
GRA | 0 | 0.25 | 4 | 1 | 1 | 0.01
CRO | 0 | 2 | 4 | 1 | 1 | 0.05
WET | 1 | 1 | N/A | 0.5 | 1 | 0.05
CVM | 0 | 8 | 4 | 1 | 1 | 0.05
BSV | 0 | 0.5 | 4 | 0.2 | 1 | 0.01
WAT | 0 | 0 | N/A | 0 | 0 | 0
:::

```{code-cell} ipython3
def lai_2_hc(lai, hc_max=0, lai_max=4):
    """Empirical equations to generate herbaceous canopy height from LAI"""
    if hc_max > 0:
        hc = np.clip(0, hc_max * lai / lai_max, hc_max)
    else:
        hc = lai / 24.
    return hc

print("Parsed LAI to canopy height function correctly, you can continue to next cell")
```

Now, with the structural parameters define we can now compute the zero-plane displacement height and roughness length for momentum

```{code-cell} ipython3
import yaml
# Set the input TSEB ancillary parameters
tseb_param_file = input_dir / "TSEB_params.yaml"
gedi_hc_file = input_dir / "ec_sites_gedi_ES.csv"

# Minimum roughness caused by soil
z0_soil = 0.001

def lai_2_hc(lai, hc_max=0, lai_max=4):
    """Estimates canopy height based on LAI for herbaceous canopies"""
    if hc_max > 0:
        hc = np.clip(0, hc_max * lai / lai_max, hc_max)
    else:
        hc = lai / 24.
    return hc

def alpha_pt_komatsu(h_c):
    alpha_pt = np.maximum(-0.269 * np.log(h_c) + 1.31, 0)
    return alpha_pt

# Avoid pandas frabmentation warnings
ec = ec.copy()

# Read ancillary TSEB parameters and LiDAR canopy height data
hc_data = pd.read_csv(gedi_hc_file, sep=";")

with open(tseb_param_file, "r") as fid:
    tseb_params = yaml.load(fid, Loader=yaml.FullLoader)


ec.loc[:, "F_C"] = tseb_params["F_C"][igbp]
w_c = tseb_params["W_C"][igbp]
leaf_size = tseb_params["LEAF_SIZE"][igbp]
landcover = tseb_params["LC"][igbp]

if igbp in ["CRO", "CVM"]:
    ec.loc[:, "H_C"] = np.maximum(lai_2_hc(ec["LAI"].values,
                                         hc_max=tseb_params["HC"][igbp]),
                                0)
elif igbp in ["GRA", "BSV"]:
    ec.loc[:, "H_C"] = np.maximum(lai_2_hc(ec["LAI"].values,
                                         hc_max=0),
                                0)
else:
    site_id = hc_data["SITE_ID"] == site
    if site_id.any() and float(hc_data.loc[site_id, "H_C"].item()) > 0:
        ec.loc[:, "H_C"] = float(hc_data.loc[site_id, "H_C"].item())
    else:
        print("Canopy height information for the site not available, "
              "getting tabulated data accoring to land cover type")
        ec.loc[:, "H_C"] = tseb_params["HC"][igbp]

alpha_pt_0 = np.full_like(ec["LAI"].values, tseb_params["alpha_PT"][igbp])
if igbp in ["ENF", "DNF", "MF"]:
    alpha_pt_0 = alpha_pt_komatsu(metdata["H_C"])
elif igbp in ["EBF", "DBF"]:
    alpha_pt_0 = 0.83

    
print(f"{site} has the following site characteristics:\n"
      f"\t Latitude: {lat} deg. \n"
      f"\t Longitude: {lon} deg. \n"
      f"\t Elevation: {elev} m \n"
      f"\t IGBP class: {igbp}\n"
      f"\t Canopy width to height ratio: {w_c}\n"
      f"\t Effective leaf size: {leaf_size} m \n"
      f"\t {1 - tseb_params['F_C'][igbp]} % clumped \n"
)

[z_0m, d_0] = TSEB.res.calc_roughness(ec["LAI"].values,
                                      ec["H_C"].values,
                                      np.full_like(ec["LAI"].values, w_c),
                                      np.full_like(ec["LAI"].values, landcover),
                                      f_c=ec["F_C"].values)

d_0[d_0 < 0] = 0
z_0m[np.isnan(z_0m)] = z0_soil
z_0m[z_0m < z0_soil] = z0_soil

print('Plotting input time series [...]')

fig = make_subplots(rows=4, cols=1,
                    shared_xaxes=True,
                    horizontal_spacing=0.01,
                   subplot_titles=("LAI", "Green fraction", "Canopy height"))

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["LAI"], 
                         name="LAI", mode="lines", marker={"color": "black"}),
              row=1, col=1)

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["FG"], 
                         name="Green fractoin", mode="lines", marker={"color": "green"}),
              row=2, col=1)

fig.add_trace(go.Scattergl(x=ec["DATE"], y=ec["H_C"], 
                         name="Canopy height", mode="lines", marker={"color": "blue"}),
              row=3, col=1)

fig.add_trace(go.Scattergl(x=ec["DATE"], y=alpha_pt_0, 
                         name="A priori PT coefficient", mode="lines", marker={"color": "orange"}),
              row=4, col=1)

fig.update_xaxes(title_text="Date", row=4, col=1)
fig.update_layout(title_text=f"Biophysical traits timeseries in {site}")
```

:::{seealso}
You can check the code implementation in pyTSEB [here](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/resistances.py#L124)
:::

+++ {"jp-MarkdownHeadingCollapsed": true}

(sec-h)=
## Sensible heat flux
The key in TSEB models is the partition of sensible heat flux into the soil and canopy layers, which depends on the soil and canopy temperatures ($T_S$ and $T_C$ respectively, left side of Fig. [](#tseb)). Given the difficulty of obtaining the pure component temperatures, even with very high resolution data due to canopy gaps,[Norman et al (1995)](https://doi.org/10.1016/0168-1923(95)02265-Y) found a solution to retrieve $T_S$ and $T_C$ using a single observation of the directional radiometric temperature $T_{rad}\left(\theta\right)$ as this is the case for most of the remote sensing systems. Eq. [](#eq-TSEB_Trad) decomposes the composite $T_{rad}\left(\theta\right)$ temperature between its components $T_S$ and $T_C$, assuming blackbody emission of thermal radiance:

:::{math}
:label:eq-TSEB_Trad
\sigma T_{rad}^4\left(\theta\right)=f_c\left(\theta\right)\sigma\,T_{C}^4+\left[1-f_{c}\left(\theta\right)\right]\sigma\,T_{S}^4
:::
where $f_c\left(\phi\right)$ is the fraction of vegetation observed by the sensor. Since [](#eq-TSEB_Trad) consists of two unknowns and only one equation, an iterative process to find $H_S$, $T_S$, $H_C$ and $T_C$ is defined based upon an initial guess of potential canopy transpiration, and under the assumption that during daytime hours condensation for the soil/substrate should not occur. The initial canopy latent and sensible heat fluxes are estimated based on the [Priestley and Taylor (1972)](https://doi.org/10.1175/1520-0493(1972)100<0081:OTAOSH>2.3.CO;2) formulation for potential transpiration (Eq. [](#eq-TSEB_PT)) and Eq. [](#eq-eb).
    
:::{math}
:label:eq-TSEB_PT
\lambda E_C &= \alpha_{PT} \, f_g \frac{\Delta}{\Delta + \gamma}R_{n,C}\\
H_C &= R_{n,C} - \lambda E_C \label{eq:H_0}
:::
where $\alpha_{PT}$ is the Priestley-Taylor coefficient, initially set to 1.26, $f_g$ is the fraction of vegetation that is green and hence capable of transpiring, $\Delta$ is the slope of the saturation vapour pressure versus temperature, $\gamma$ is the psychrometric constant. $T_{C}$ is then computed by inverting the equation for turbulent transport of heat between the surface and the reference height above the surface)
    
With a first estimate of $T_{C}$, soil temperature is computed from Eq. [](#eq-TSEB_Trad) and then soil sensible and latent heat fluxes. At this stage, if soil latent heat flux results to be non-negative, a solution is found, otherwise the Priestley-Taylor coefficient is reduced incrementally , reducing hence the canopy transpiration, in order to avoid negative soil latent heat flux, until a realistic solution is found (no condensation occurring neither in the soil nor in the canopy in daytime). 

:::{seealso}
For more details the reader is addressed to the works of [Norman et al (1995)](10.1016/0168-1923(95)02265-Y) or [Kustas and Norman (1999)](10.1016/S0168-1923(99)00005-2)
:::

The full code of TSEB-PT is implemented in the pyTSEB GitHub repository ([](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L499)). The general workflow is described in {numref}`workflow %s <wf:tseb>`

:::{mermaid}
:label:wf:tseb
:enumerated:true

flowchart TD   
    MO_0[/Initial MO length $$L_0 = \infty$$/]
    PT_0[/Initial PT value: $$\alpha=1.26$$/]
    canopy_flux[Compute canopy fluxes]
    tc[/$$T_C$$/]
    unmix[Compute temperature]
    ts[/$$T_S$$/]
    soil_flux[Compute soil fluxes]
    le_s[/$$\lambda E_S$$/]
    neg_le{Is $$\lambda E_S$$ negative?}
    pt_reduce[Reduce PT value]
    PT_1[/Reduced PT value/]
    MO[Compute MO lenght]
    MO_1[/ $$L_1$$/]
    converge{Is $$L_1 \simeq L_0$$}
    update_mo[$$L_0 = L_1$$]
    solution[/TSEB fluxes/]

    subgraph MO iteration
        direction TB
        MO_0 --> canopy_flux
        PT_0 --> canopy_flux
        subgraph PT iteration
            direction LR
            canopy_flux --> tc
            tc --> unmix --> ts
            ts --> soil_flux --> le_s
            le_s --> neg_le
            neg_le -- yes --> pt_reduce
            pt_reduce --> PT_1
            PT_1 --> canopy_flux
            
        end

        neg_le -- no ----> MO --> MO_1
        MO_1 --> converge
        converge -- yes ---> solution
        converge -- no --> update_mo
        update_mo --> canopy_flux
        
    end
:::

:::{note}
There are two main steps in TSEB-PT:
* A first step that prepares all ancillary variables, such as air density, heat capacity of air..., and initializes a priori values of $T_C$, $T_S$, $Ln$, and $G$: From line [728](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L728) to line [756](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L756).
* A second iterative step that consists on two nested loops:
  + an outer loop that searches for convergence in the friction velocity and Monin-Obukhov lenght and serves to recalculate the aerodynamic resistances: From line [756](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L756) to line [903](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L903)
  + an inner loop that starts with the potential Priestley-Taylor coefficient and interatively starts being reduced for each case individually until non-negative turbulent fluxes are obtained for both canopy and soil: From line [780](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L780) to line [882](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L882) 
:::

:::{note}
The outer loop regarding the recalculation of atmospheric stability factors tends to converge pretty fast under daytime/unstable conditions, but on the other hand it has difficulties in converging under stable/nighttime conditions. Therefore in pyTSEB it is set a number of maximum iterations for this loop reflected in the [`ITERATIONS`](https://github.com/hectornieto/pyTSEB/blob/382e4fc01e965143ebafdaefe5be9b45c737455a/pyTSEB/TSEB.py#L103C1-L103C11) constant
:::

+++

# run TSEB-PT
Now we are almost ready to run TSEB-PT. As a summary the {numref}`workflow %s <wf:tseb-inputs>` shows all the inputs and outputs generated by TSEB and its submodules:

:::{mermaid}
:label:wf:tseb-inputs
:enumerated:true
---
title: pyTSEB workflow
---
flowchart LR
    weather[(Weather Forcing)]
    lst[/LST/]
    lai[/Total LAI/]
    glai[/Green LAI/]
    lidf[/Leaf angle/]
    albedo[/albedo & transmittance/]
    hc[/Canopy Height/]
    wc[/Canopy Shape/]
    fc[/Cover of clumped canopy/]
    lw[/Effective Leaf Width/]

    rtm[Radiative Transfer]
    tseb[TSEB-PT]

    roughness[/Aerodynamic roughness/]
    sn[/Net Soil/Canopy SW Radiation/]
    ln[/Net Soil/Canopy LW Radiation/]
    g[/Soil Heat Flux/]
    h[/Soil/Canopy Sensible Heat Flux/]
    le[/Soil/Canopy Latent Heat Flux/]

    lst & lai & glai & lw ----> tseb
    tseb --> ln & g & h & le
    weather & wc & fc & lai & albedo & lidf --> rtm
    weather --> tseb
    rtm --> sn
    sn --> tseb
    lai & hc & fc --> roughness
    roughness --> tseb
:::

+++

In addition, we also need to especify the height of measurements of both the wind speed ($z_u$) and air temperature ($z_t$), since these to parameters are key to correctly estimate the aerodynamic roughness. These values can be obtained from the site characteristics in [](#site-description)

:::{warning}
If you recall from the Eq. 3 of the [102-Turbulence_and_sensible_heat_flux](./102-Turbulence_and_sensible_heat_flux.ipynb) notebook,$z_u$ and  $z_t$ are within a logarithmic function and thus these values should **never be lower than the canopy height**, with the risk otherwise to generate arithmetic errors and produce no valid data. In case of tall canopies combined with data from agrometeorological stations (usually with sensors at 2m) it is recommended to drive these values to be well above the canopy, ideally above the roughness sublayer (e.g. $2 \times h_c$).
:::

As an example on how TSEB finds iterativelly one plausible solutition (i.e. non-negative latent heat fluxes during daytime) we will run one single case and plot the different values during the iterative process:

### Satellite products
These values are just a sample extracted from ECOSTRESS LST product and biophysical traits derived from Sentinel-2. 

:::{seealso}
If you want to know more about how biophysical traits (LAI, pigments, ...) are derived you can optionally look at the notebook  {doc}`301_spectral_properties.ipynb`
:::

```{code-cell} ipython3
lst = 314.58 # ECOSTRESS Land Surface Temperature  (K)
vza = 9.4321671  # ECOSTRESS View Zenith Angle (deg.)
sza = 28.6739051703528  # ECOSTRESS Sun Zenith Angle (deg.)
lai = 0.89952921293024  # Sentinel-2 Leaf Area Index
leaf_angle = 57.5344928614439  # Sentinel-2 Mean Leaf Inclination Angle (deg.)
f_g = 0.826559189567835  # Sentinel-2 Fraction of LAI that is green
print("Parsed Earth Observation inputs correctly, you can continue to next cell")
```

### Weather forcing
Weather forcing is extracted from the [ECMWF ERA5 hourly reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview), which are interpolated and topographically corrected to get the weather forcings at the time of the satellite overpass.

:::{seealso}
The weather forcing preprocessing is done using the [meteo_utils GitHub package](https://github.com/hectornieto/meteo_utils)
:::

```{code-cell} ipython3
t_air = 302.309448242188  # air temperature (K)
ea = 14.79697608947758  # Vapour pressure (hPa)
ws = 4.32157111167908  # Wind speed (ms-1)
press = 980.325317382812  # Atmospheric pressure (hPa)
lw_in = 332.861083984375  # Longwave irradiance (Wm-2)
sw_in = 1014.41126346588  # Shortwave irradiance (Wm-2)

sn_c = 500.114215358855  # Estimated canopy net shortwave radiation (Wm-2)
sn_s = 259.319271056837  # Estimated soil net shortwave radiation (Wm-2)

Z_T = 100  # Temperature measurement at blending height
Z_U = 100  # Wind measurement at blending height

print("Parsed weather forcing correctly, you can continue to next cell")
```

### Structural canopy parameters

```{code-cell} ipython3
f_c	= 1  # Homogeneous canopy
w_c = 1  # Homogeneous canopy
leaf_width = 0.01  # Effective leaf size (m)
h_c	= 0.0791470505387601  # Canopy height (m)

print("Parsed ancillary canopy structural parameters correctly, you can continue to next cell")
```

## Initial state variables and constants

```{code-cell} ipython3
resistance_method = [0]
resistance_params = {}
# Standard PT value
alpha_pt_0 = 1.26

# Thermal spectra
e_v = 0.99  # Leaf emissivity
e_s = 0.94  # Soil emissivity

# Calculate constants
rho = TSEB.met.calc_rho(press, ea, t_air)  # Air density
c_p = TSEB.met.calc_c_p(press, ea)  # Heat capacity of air
[z_0m, d_0] = TSEB.res.calc_roughness(lai,
                                      h_c,
                                      w_c,
                                      TSEB.res.CROP,
                                      f_c=np.array(f_c))

x_lad =  TSEB.rad.leafangle_2_chi(leaf_angle)
z_0h = TSEB.res.calc_z_0H(z_0m, kB=0)  # Roughness length for heat transport

# Calculate LAI dependent parameters for dataset where LAI > 0
omega0 = TSEB.CI.calc_omega0_Kustas(lai, f_c, x_LAD=x_lad, isLAIeff=True)
F = np.asarray(lai / f_c)  # Real LAI
# Fraction of vegetation observed by the sensor
f_theta = TSEB.calc_F_theta_campbell(vza, F, w_C=w_c, Omega0=omega0, x_LAD=x_lad)

print("Computed initial TSEB state variables, you can continue to next cell")
```

## Monin-Obukhov iteration and Priestley Taylor iteration

```{code-cell} ipython3
# Create the arrays to store the different values during the iterations
t_c_array = []
t_s_array = []
le_c_array = []
le_s_array = []
h_c_array = []
h_s_array = []
t_aero_array = []
alpha_pt_array = []
l_mo_array = []
l_diff_array = []

L_MO_THRES = 0.001

def calc_t_s(lst, t_c, f_theta):
    '''Estimates soil temperature from the directional LST.

    Parameters
    ----------
    lst : float
        Directional Radiometric Temperature (K).
    t_c : float
        Canopy Temperature (K).
    f_theta : float
        Fraction of vegetation observed.

    Returns
    -------
    t_s: float
        Soil temperature (K).

    References
    ----------
    Eq. 1 in [Norman1995]_'''

    t_temp = lst**4 - f_theta * t_c**4
    t_s = (t_temp / (1.0 - f_theta))**0.25
    return t_s

# Assume neutral conditions
l_mo = np.array(np.inf)
l_old = float(l_mo)
u_friction = TSEB.MO.calc_u_star(ws, Z_U, l_mo, d_0, z_0m)
# First assume that canopy temperature equals the minumum of Air or
# radiometric T
t_c = np.minimum(lst, t_air)
# Retrieve soil temperature from LST and Tc based on gap fraction
t_s = calc_t_s(lst, t_c, f_theta)
# First guess of aerodynamic temperature is t air
t_ac = float(lst)

l_diff = np.inf
for n_iterations in range(50):
    alpha_pt_rec = alpha_pt_0 + 0.1
    le_s = -1
    while le_s < 0:    
        # Reduce the Priestley Taylor parameter by 0.1
        alpha_pt_rec -= 0.1
    
        # Calculate aerodynamic resistances
        r_a, r_x, r_s = TSEB.calc_resistances(
                  resistance_method,
                  {"R_A": {"z_T": Z_T, "u_friction": u_friction, "L": l_mo,
                           "d_0": d_0, "z_0H": z_0h},
                   "R_x": {"u_friction": u_friction, "h_C": h_c,
                           "d_0": d_0,
                           "z_0M": z_0m, "L": l_mo, "F": F, "LAI": lai,
                           "leaf_width": leaf_width,
                           "z0_soil": z0_soil,
                           "massman_profile": [0, []],
                           "res_params": {k: resistance_params[k] for k in resistance_params.keys()}},
                   "R_S": {"u_friction": u_friction, "h_C": h_c,
                           "d_0": d_0,
                           "z_0M": z_0m, "L": l_mo, "F": F, "omega0": omega0,
                           "LAI": lai, "leaf_width": leaf_width,
                           "z0_soil": z0_soil, "z_u": Z_U,
                           "deltaT": t_s - t_ac, 'u': ws, 'rho': rho,
                           "c_p": c_p, "f_cover": f_c, "w_C": w_c,
                           "massman_profile": [0, []],
                           "res_params": {k: resistance_params[k] for k in resistance_params.keys()}}
                   }
        )
    
        # Calculate net longwave radiation with current values of T_C and T_S
        ln_c, ln_s = calc_L_n_Campbell(
            t_c, t_s, lw_in, lai, e_v, e_s, x_LAD=x_lad)
        rn_c = sn_c + ln_c
        rn_s = sn_s + ln_s
        # Compute Soil Heat Flux Ratio
        g_flux = 0.35 * rn_s
        
        # Calculate the canopy and soil temperatures using the Priestley
        # Taylor appoach
        h_c_flux = TSEB.calc_H_C_PT(
            rn_c, f_g, t_air, press, c_p, alpha_pt_rec)
        
        t_c = TSEB.calc_T_C_series(
            lst, t_air, r_a, r_x, r_s, f_theta, h_c_flux, rho, c_p)
        
        # Calculate soil temperature
        t_s = calc_t_s(lst, t_c, f_theta)
        
        # Recalculate soil aerodynamic resistance
        _, _, r_s = TSEB.calc_resistances(
                  resistance_method,
                  {
                   "R_S": {"u_friction": u_friction, "h_C": h_c,
                           "d_0": d_0,
                           "z_0M": z_0m, "L": l_mo, "F": F, "omega0": omega0,
                           "LAI": lai, "leaf_width": leaf_width,
                           "z0_soil": z0_soil, "z_u": Z_U,
                           "deltaT": t_s - t_ac, 'u': ws, 'rho': rho,
                           "c_p": c_p, "f_cover": f_c, "w_C": w_c,
                           "massman_profile": [0, []],
                           "res_params": {k: resistance_params[k] for k in resistance_params.keys()}}
                   }
        )
    
        # Get air temperature at canopy interface
        t_ac = ((t_air / r_a + t_s / r_s + t_c / r_x)
                   / (1.0 / r_a + 1.0 / r_s + 1.0 / r_x))
    
        # Calculate soil fluxes
        h_s_flux = rho * c_p * (t_s - t_ac) / r_s
    
        # Estimate latent heat fluxes as residual of energy balance at the
        # soil and the canopy
        le_s = rn_s - g_flux - h_s_flux
        le_c = rn_c - h_c_flux
        
        # No tranpiration 
        if alpha_pt_rec <= 0.0: 
            h_s_flux = np.minimum(h_s_flux, rn_s - g_flux)
            g = np.maximum(g_flux, rn_s - h_s_flux)
            le_s = 0
            le_c = 0
            
        # Calculate total fluxes
        h = h_c_flux + h_s_flux
        le = le_c + le_s
    
        t_c_array.append(t_c)
        t_s_array.append(t_s)
        le_c_array.append(le_c)
        le_s_array.append(le_s)
        h_c_array.append(h_c_flux)
        h_s_array.append(h_s_flux)
        t_aero_array.append(t_ac)
        alpha_pt_array.append(alpha_pt_rec) 
        l_mo_array.append(z_0m / l_mo)
        
        # Recompute the MO length        
        l_mo = TSEB.MO.calc_L(
            u_friction, t_air, rho, c_p, h, le)
        u_friction = TSEB.MO.calc_u_star(
            ws, Z_U, l_mo, d_0, z_0m)
        
        l_diff = np.fabs(l_mo - l_old) / np.fabs(l_old)
        l_diff_array.append(l_diff)
        

    l_diff = np.fabs(l_mo - l_old) / np.fabs(l_old)
    l_old = float(l_mo)
    if l_diff <= L_MO_THRES:
        break


iterations = np.arange(len(le_c_array))

[t_c_array, t_s_array, le_c_array, le_s_array, h_c_array, 
 h_s_array, t_aero_array, alpha_pt_array, l_mo_array, l_diff_array] = map(
     np.array, [t_c_array, t_s_array, le_c_array, le_s_array, h_c_array, 
 h_s_array, t_aero_array, alpha_pt_array, l_mo_array, l_diff_array])


fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True)

axs[0].plot(iterations, alpha_pt_array, "k-", label=r"$\alpha_PT$")
axs[0].set_ylim([0, 1.36])

axs[1].plot(iterations, l_mo_array, "k-", label=r"Stability")
ax = axs[1].twinx()
ax.plot(iterations, l_diff_array, "b:", label="Rel. diff.")
ax.axhline(y=L_MO_THRES, color="b", ls="--")
axs[1].legend(loc="upper right")
ax.legend(loc="upper left")
axs[2].plot(iterations, t_c_array, "g-", label=r"T$_c$")
axs[2].plot(iterations, t_s_array, "r-", label=r"T$_s$")
axs[2].plot(iterations, t_aero_array, "k:", label=r"T$_0$")
axs[2].legend()
axs[3].plot(iterations, le_c_array, "g-", label=r"LE$_c$")
axs[3].plot(iterations, le_s_array, "r-", label=r"LE$_s$")
axs[3].axhline(y=0, ls="--", color="k")
axs[3].legend()
```

## Extrapolation to daily values
TSEB-PT directly estimates instantaneous turbulent fluxes at the LST acquisition time (in W m⁻²). However most applications require daily estimates of evapotranspiration (e.g. in mm day⁻¹). 

One simple approach to extrapolate the instantaneous latent heat fluxes ($\lambda E_{inst.}$) to daily ET rates is by assuming that the ratio of instantaneous to daily shortwave irriadiance ($\frac{S^{\downarrow}_{day}}{S^{\downarrow}_{inst.}}$) is preserved during daytime hours. Based on this assumption daily latent heat flux can be estimated using {eq}`eq:et`:

:::{math}
:label:eq:et
\lambda E_{day} = \lambda E_{inst.} \frac{S^{\downarrow}_{day}}{S^{\downarrow}_{inst.}}
:::

And then the daily latent heat flux (W m⁻²) is converted water units (mm day⁻¹) by the latent heat of vaporization.

```{code-cell} ipython3
ta_day = 298.563903808594  # Average daily temperature (K)
sw_in_day = 335.073944091797  # Average shortwave irradiance (Wm-2)

le_daily = le * sw_in_day / sw_in
et = TSEB.met.flux_2_evaporation(le_daily, ta_day, 24)

print(f"Daily ET is estimatet at {et:.2f} mm day⁻¹")
```

## Example of Satellite-based implementation

We will now show an example on how to run pyTSEB with satellite imagery. We will run the model with an example imagery from the Sentinel constellation around (2km x 2km) the [Majadas](https://meta.icos-cp.eu/resources/stations/ES_ES-LMa) EC towers (ES-LMa/ES-LM1/ES-LM2). 

For this example, we will only use the land surface temperature (LST) and leaf area index (LAI) products from the Sentinel constellation. The LST was obtained from Sentinel-3 SLSTR sensor and was sharpened to 20m using the Sentinel-2 MSI sensor and a Data Mining Sharpening algorithm ([pyDMS](https://github.com/radosuav/pyDMS)). While the LAI products were estimated by inverting a radiative transfer model (i.e. [pyProSail](https://github.com/hectornieto/pypro4sail) ) using a hybrid appraoch. These processing algorithms are described in [Guzinski et al. 2023](https://www.sciencedirect.com/science/article/pii/S1569843223004119) and [Guzinski et al. 2025](https://doi.org/10.5194/egusphere-2025-4342).

Normally, other products such land cover, green fraction, canopy height (i.e. GEDI) maps and other spatially explicit variables can also be intgrated into the workflow but for this example we keep it simple. 

:::{seealso}
For more in depth tutorials on how pre-process cloud-based Sentinel inputs and run TSEB, please see [SenET toolbox](https://github.com/DHI/Sen-ET-OpenEO-toolbox)
:::

### Open satellite inputs
We have prepared inputs of LST and LAI images from 2022-04-16 over ES-LMa (2x2km around tower)

```{code-cell} ipython3
from osgeo import gdal

sat_dir = input_dir / 'satellite'
# open lst
lst_file = sat_dir / '20220416T103841_S3-DMS_LST.tif'
lst_fid = gdal.Open(str(lst_file))
lst_ar = lst_fid.GetRasterBand(1).ReadAsArray()
# open lai 
lai_file = sat_dir / '20220416_S2_LAI.tif'
lai_fid = gdal.Open(str(lai_file))
lai_ar = lai_fid.GetRasterBand(1).ReadAsArray()

s3_overpass_datetime = pd.to_datetime('2022-04-16 10:30') # time in UTC

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.03,
    subplot_titles=("LST", "LAI"))

# LST
fig.add_trace(
    go.Heatmap(
        z=lst_ar,
        colorscale="RdYlBu_r",
        zmin=295,
        zmax=305,
        colorbar=dict(
            title="LST (K)",
            orientation="h",
            x=0.2,      
            y=-0.25,     
            len=0.5,    
            thickness=15)),
    row=1,col=1)

# LAI
fig.add_trace(
    go.Heatmap(
        z=lai_ar,
        zmin=0,
        zmax=4,
        colorscale="Viridis",
        colorbar=dict(
            title="LAI (m²/m²)",
            orientation="h",
            x=0.72,     
            y=-0.25,     
            len=0.5,   
            thickness=15)),  
    row=1, col=2)

fig.update_layout(
    title=f"Sentinel inputs from {s3_overpass_datetime}",
    height=500,
    width=1000)

# Make pixels square like imshow
fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)
fig.update_yaxes(autorange="reversed", scaleanchor="x2", row=1, col=2)
```

## Meteo and ancillary data from EC tower
Since we are use the example from Majadas, we will acquire meteorological and ancillary data directly form tower data (i.e. ES-LMa)

```{code-cell} ipython3
site = 'ES-LMa'
# Set the input files to Majadas tower (ES-LMa)
bio_filename = bio_dir / f"{site}_HLS-l2c.csv"
ec_filename = ec_dir / f"FLX_{site}_FLUXNET_SUBSET_HH.csv"

site_id = site_metadata["SITE_ID"] == site
lat = site_metadata.loc[site_id, "LOCATION_LAT"].item()
lon = site_metadata.loc[site_id, "LOCATION_LONG"].item()
elev = site_metadata.loc[site_id, "LOCATION_ELEV"].item()
utc_offset = site_metadata.loc[site_id, "UTC_OFFSET"].item()
stdlon = 15 * utc_offset
# Measurement height of wind speed and air temperature (in the case of ES-LMa, it is 15m)
z_u = 15. 
z_t = 15.
    
# Read the fluxnet EC table and preprocess
ec_maj =  pd.read_csv(ec_filename, sep=',', na_values=-9999)
ec_maj["TIMESTAMP_START"] = pd.to_datetime(
    ec_maj["TIMESTAMP_START"], format="%Y%m%d%H%M")


# satellite overpass mask
datetime_mask = ec_maj['TIMESTAMP_START'] == s3_overpass_datetime + pd.Timedelta(hours=utc_offset)
# extract meteo data at overpass time 
Ta = float(ec_maj['TA_F'].values[datetime_mask][0]) + 273.15 # air temperature (convert to K)
sw_in = float(ec_maj['SW_IN_F'].values[datetime_mask][0])  # incoming shortwave irradiance
lw_in = float(ec_maj['LW_IN_F'].values[datetime_mask][0])  # incoming longwave irradiance
vpd = float(ec_maj['VPD_F'].values[datetime_mask][0])  # vapour pressure deficit 
es = TSEB.met.calc_vapor_pressure(Ta) # saturated vapour pressure
ea = es - vpd # actual vapour pressure
u =  float(ec_maj['WS_F'].values[datetime_mask][0])  # wind speed
P =  float(ec_maj['PA_F'].values[datetime_mask][0]) * 10  # air pressure convert to hPa

print(f'Meteo conditions at {site} during S3 overpass ({s3_overpass_datetime}):\n Ta = {np.round(Ta,2)} K\n SWin = {np.round(sw_in,2)} Wm-2\n ea = {np.round(ea,2)} hPa\n u = {np.round(u,2)} ms-1')
```

### Biophysical and structural parameters
For simplicity, we wiill assume a constant $f_c$ and $h_c$ across the image window. In other applications, other spatial or satellite-based estimates could also be used to inform these variables such as tree fractional cover products (e.g., canopy height models from LiDAR such as [GEDI](https://gedi.umd.edu/dataproducts/products/)).

```{code-cell} ipython3
# from site information
fc_ar = np.full_like(lst_ar, 1) # fractional cover
hc_ar = np.full_like(lst_ar, 8) # canopy height
    
# sun angles during S3 overpass
sza, saa = TSEB.met.calc_sun_angles(lat,
                                    lon,
                                    stdlon,
                                    s3_overpass_datetime.dayofyear,
                                    s3_overpass_datetime.hour + s3_overpass_datetime.minute / 60.)
# local/real LAI of overstory
F = lai_ar/ fc_ar
# clulmping of overstory
omega_0 = TSEB.CI.calc_omega0_Kustas(lai_ar, fc_ar, x_LAD=1)
omega = TSEB.CI.calc_omega_Kustas(omega_0, np.minimum(sza, 90), w_C=w_c)
# effective LAI of overstory
lai_eff = F * omega

## roughness parameters
[z_0m, d_0] = TSEB.res.calc_roughness(lai_ar,
                                      hc_ar, 
                                      np.full_like(lai_ar, w_c),
                                      np.full_like(lai_ar, TSEB.res.BROADLEAVED_E),
                                      f_c=fc_ar)
# Ensure realistic values
d_0[d_0 < 0] = 0
z_0m[np.isnan(z_0m)] = z0_soil
d_0[d_0 < z0_soil] = z0_soil

print("Parsed biophysical and structural parameters correctly, you can continue to next cell")
```

## Running pyTSEB with satellite inputs

```{code-cell} ipython3
vza = np.full_like(lst_ar, 21.88) # from S3-SLSTR product

print('Estimating net shortwave radiation [...]\n')

# Estimates the direct and diffuse solar radiation in NIR and VIS
difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(np.full_like(lst_ar, sw_in),
                                                        np.full_like(lst_ar, sza),
                                                        press=np.full_like(lst_ar, P))

par_dir = fvis * (1. - difvis) * np.full_like(lst_ar, sw_in)
nir_dir = fnir * (1. - difnir) * np.full_like(lst_ar, sw_in)
par_dif = fvis * difvis * np.full_like(lst_ar, sw_in)
nir_dif = fnir * difnir * np.full_like(lst_ar, sw_in)

# shortwave radiation model
sn_c, sn_s = TSEB.rad.calc_Sn_Campbell(lai_ar, 
                                       np.full_like(lst_ar, sza), 
                                       par_dir + nir_dir, 
                                       par_dif + nir_dif, 
                                       fvis, 
                                       fnir,
                                       np.full_like(lst_ar, 0.05),
                                       np.full_like(lst_ar, 0.08),
                                       np.full_like(lst_ar, 0.26),
                                       np.full_like(lst_ar, 0.33),
                                       np.full_like(lst_ar, 0.07),
                                       np.full_like(lst_ar, 0.07),
                                       x_LAD=1, 
                                       LAI_eff=lai_eff)

# use Norman and Kustas 1995 resistance framework
Resistance_flag=[0,{}]
# using constant ratio appraoch to estimate G
G_constant = 0.35
calcG = [[1], G_constant]

print('Running TSEB [...]\n')

[flag_PT_all, T_soil, T_veg, T_AC, Ln_S, Ln_C, LE_C, H_C,
     LE_S, H_S, G_mod, R_S, R_X, R_A, u_friction, L, n_iterations] = TSEB.TSEB_PT(lst_ar,
                                                                                    vza,
                                                                                    np.full_like(lst_ar, Ta),
                                                                                    np.full_like(lst_ar, u),
                                                                                    np.full_like(lst_ar, ea),
                                                                                    np.full_like(lst_ar, P),
                                                                                    sn_c,
                                                                                    sn_s,
                                                                                    np.full_like(lst_ar, lw_in),
                                                                                    lai_ar,
                                                                                    hc_ar,
                                                                                    e_v,
                                                                                    e_s,
                                                                                    z_0m,
                                                                                    d_0,
                                                                                    z_t,
                                                                                    z_u,
                                                                                    leaf_width=leaf_width,
                                                                                    alpha_PT=alpha_pt_0,
                                                                                    f_c=fc_ar,
                                                                                    f_g=np.full_like(lst_ar,0.8),
                                                                                    calcG_params=calcG,
                                                                                    resistance_form=Resistance_flag)


# save ouputs in outdict 
LE = LE_C + LE_S
H = H_C + H_S

Rn_C = Ln_C + sn_c
Rn_S = Ln_S + sn_s
Rn = Rn_C + Rn_S



fig = make_subplots(
    rows=1,
    cols=3,
    horizontal_spacing=0.03,
    subplot_titles=("LE", "H", "Rn"))

# LE
fig.add_trace(
    go.Heatmap(
        z=LE,
        colorscale="Blues",
        zmin=100,
        zmax=500,
        colorbar=dict(
            title="LE (W/m²)",
            orientation="h",
            x=0.13,     
            y=-0.2,     
            len=0.3,    
            thickness=15)),
    row=1,col=1)

# H
fig.add_trace(
    go.Heatmap(
        z=H,
        zmin=100,
        zmax=500,
        colorscale="Reds",
        colorbar=dict(
            title="H (W/m²)",
            orientation="h",
            x=0.48,      
            y=-0.2,    
            len=0.3,    
            thickness=15)),  
    row=1, col=2)

# Rn
fig.add_trace(
    go.Heatmap(
        z=Rn,
        zmin=200,
        zmax=800,
        colorscale="Oranges",
        colorbar=dict(
            title="Rn (W/m²)",
            orientation="h",
            x=0.82,      
            y=-0.2,     
            len=0.3,    # colorbar length
            thickness=15)),  
    row=1, col=3)


fig.update_layout(
    title="Modelled instananeous fluxes from TSEB",
    height=500,
    width=1500)

# Make pixels square like imshow
fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)
fig.update_yaxes(autorange="reversed", scaleanchor="x2", row=1, col=2)
fig.update_yaxes(autorange="reversed", scaleanchor="x3", row=1, col=3)
```

:::{hint}
**Questions:**
- How do the modelled fluxes compare to the tower measurements during overpass time?
- How could we get a map of daily ET (mm/day) from these instananeous fluxes?
:::

+++

# Three-Source Energy Balance (3SEB) model adaptation
The Three-Source Energy Balance (3SEB) model ([**Burchard-Levine et al. 2022**](https://doi.org/10.1111/gcb.16002)) is an adapted version of the TSEB model [**Norman et al. (1995)**](https://doi.org/10.1016/0168-1923(95)02265-Y), where an additional vegetation source is incorporated into TSEB to better characterize agro-ecosystems with multiple vegetation layers that have distinct structural and physiological features. This adaptation was initially envisioned for savanna-like ecosystems, which are characterized by the co-existence of tree and herbaceous species with very different structural and phenological dynamics. 

However, the 3SEB model structure is also well adapted for perennial tree crops that have cover crops in the interrow, such as vineyards, olive orchards, almonds etc. In principle, the 3SEB model is more suited to represent the complex interactions between a clumped vegetation source planted in rows with an additional herbaceous vegetation source located in the interrows. 

:::{figure} ./input/figures/ThreeSource_scheme_v3_highres.png
:alt: 3SEB resistances scheme
:name: 3seb-scheme
Three-Source energy balance scheme for transport sensible heat (H). From [**Burchard-Levine et al. 2022**](https://doi.org/10.1111/gcb.16002)
:::

In 3SEB, the energy balance is therefore separated between the vegetation overstory (ov), vegetation understory (un) and soil (s) sources as: 

:::{math}
:label:eq-eb-3seb
  R_{n} & \approx H + \lambda E + G\\
  R_{n,ov} & \approx H_{ov} + \lambda E_{ov}\label{eq:Energy_Balance_TSEB}\\
  R_{n,un} & \approx H_{un} + \lambda E_{un}\label{eq:Energy_Balance_TSEB}\\
  R_{n,s} & \approx H_{s} + \lambda E_{s} + G
:::

The energy balance equations are simplified as they neglect additional components of the energy balance that are usually neglected, such as heat advection, storage of energy in the canopy layer or energy for the fixation of CO2, which are not computed by the model. 

## 3SEB inputs
3SEB has very similar data requirements to TSEB (as discussed above) but a key distinction is the need to separate biophsyical properties (e.g., LAI and $H_c$) for the two vegetation sources instead of only one vegetation source as in TSEB.

+++

## Structural properties for each vegetation layer
Here we will characterze each vegetation (e.g. trees and grasses) with their respective values

```{code-cell} ipython3
# fractional cover
f_c_ov	= 0.2  # overstory fractional cover (e.g., clumped trees)
f_c_un	= 1.  # understory fractional cover (e.g., homogeneous grass)

# leaf width
leaf_width_ov = 0.05  # Effective leaf size (m) of overstory (e.g. trees)
leaf_width_un = 0.01  # Effective leaf size (m) of overstory (e.g. grass)

# canopy height
h_c_ov	= 8.0  # Canopy height (m) of overstory (e.g. trees)
h_c_un	= 0.25  # Canopy height (m) of understory (e.g. grass)
print("Parsed ancillary canopy structural parameters correctly, you can continue to next cell")
```

### LAI and $f_g$ decomposition between vegetation layers 

In a savanna ecosystem, we would need to characterize the LAI for both the trees (i.e. overstory) and grasses (i.e. understory) or, in vineyards for example, the grapevine (i.e. overstory) and the cover crop (i.e. understory). There is no general formula to separate or unmix LAI from the different vegetation sources, there are potentially many different methods that can be used to disentangle the biophysical properties of both vegetation layers (e.g. unmixing algorithms, radiative transfer modelinge etc)

In this example, we will apply 3SEB over an Iberian savanna-like ecosystem (known as Dehesas in Spain or Montados in Portugal). These ecosystems are characterized by the co-existence of scattered trees (mostly Holm Oaks or *Quercux Ilex L*) over annual herbaceous species. The decomposition of features between both vegetation elements is challenging in savannas, as they overlap over each other and cannot be discriminated at the conventional satellite spatial resolution (>10m) or EC tower footprint. However, the two vegetation layers have distinct phenological patterns and these temporal differences can be used to infer the LAI of each source. During the dry period, the lack of water availability causes the understory species to senesce, while the overstory, largely composed of evergreen trees with deeper rooting systems, are adapted to survive these arid conditions. As such, during the seasonal drought, the landscape spectral signal and green LAI (i.e., $LAI_{eco}$) is mostly composed of the overstory ($LAI_{ov}$), capturing little to none of the $LAI_{un}$. Therefore, for each year, the $LAI_{eco}$ from the peak dry period can be considered as $LAI_{ov}$ and for sites with evergreen tree species, a year-round constant $LAI_{ov}$ can be implemented as the LAI from the tree species do not fluctuate greatly and have minimal phenology (i.e. [Luo et al., 2018](https://doi.org/10.3390/rs10081293); [Moore et al., 2016](https://doi.org/10.5194/bg-13-2387-2016).

```{code-cell} ipython3
# Ecosystem LAI 
lai_eco = ec['LAI'].values

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

## Roughness and clumping index
Similar to TSEB we need to calculate the areodynamic roughness parameters, along with the clumping index for each vegetation layer

```{code-cell} ipython3
# Overstory (i.e. Evergreen trees)
# =========
[z_0m_ov, d_0_ov] = TSEB.res.calc_roughness(lai_ov,
                                            h_c_ov, 
                                            np.ones(lai_ov.shape) * w_c,
                                            np.ones(lai_ov.shape) * TSEB.res.BROADLEAVED_E,
                                            f_c=np.ones(lai_ov.shape) * f_c_ov)
# Ensure realistic values
d_0_ov[d_0_ov < 0] = 0
z_0m_ov[np.isnan(z_0m_ov)] = z0_soil
d_0_ov[d_0_ov < z0_soil] = z0_soil

# Understory (i.e. grasses)
# ==========
z_0m_un, d_0_un = TSEB.res.calc_roughness(lai_un, 
                                          h_c_un,
                                          np.ones(lai_un.shape) * w_c,
                                          np.ones(lai_un.shape) * TSEB.res.GRASS,
                                          f_c=np.ones(lai_ov.shape) * f_c_un)

# Ensure realistic values
d_0_un[d_0_un < 0] = 0
z_0m_un[np.isnan(z_0m_un)] = z0_soil
z_0m_un[z_0m_un < z0_soil] = z0_soil

# Calculate clumping index for overstory
omega0_ov = TSEB.CI.calc_omega0_Kustas(lai_ov, f_c_ov, x_LAD=x_lad, isLAIeff=True)
Omega_ov = TSEB.CI.calc_omega_Kustas(omega0_ov, np.minimum(sza, 90), w_C=w_c)
F_ov = lai_ov/f_c_ov # Real LAI_ov
# effective LAI (tree layer)
lai_ov_eff =  F_ov * Omega_ov
print("Parsed clumping and roughness paramters correctly, you can continue to next cell")
```

On the other hand, for the grass understory we can assume that it is horizontally homogeneous and non-clumped so we don't need to calculate the clumping index

+++

## Three-Source Net Shortwave Radiation Estimates

For 3SEB, we use an adapted radiation transmission model similar to the model described in [**Chapter 15 of Campbell and Norman (1998)**](https://doi.org/10.1007/978-1-4612-1626-1_15) and above for TSEB. The only difference is that we simulate radiation transmission for an extra vegetation source. In this case, we also estimate the projected shadow of the overstory onto the substrate system, taking into account its influence on the proportion of direct/diffuse radiation in the substrate.

:::{note}
The details regarding the adapted three-source shortwave radiation transmission model is found in the supplementary information in [**Burchard-Levine et al. 2022**](https://doi.org/10.1111/gcb.16002) and in the [py3SEB code](https://github.com/VicenteBurchard/py3SEB/blob/main/py3seb/py3seb.py#L1851)
:::

```{code-cell} ipython3
from model_evaluation import double_collocation as dc
import tabulate
from py3seb import py3seb

# Thermal spectra
e_c_ov = 0.99  # Overstory Leaf emissivity
e_c_un = 0.97  # Understory Leaf emissivity
e_s = 0.94  # Soil emissivity

# Estimates the direct and diffuse solar radiation
difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(ec["SW_IN_F"].values,
                                                        theta,
                                                        press=np.full_like(theta, 1013.15))
par_dir = fvis * (1. - difvis) * ec["SW_IN_F"].values
nir_dir = fnir * (1. - difnir) * ec["SW_IN_F"].values
par_dif = fvis * difvis * ec["SW_IN_F"].values
nir_dif = fnir * difnir * ec["SW_IN_F"].values

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
fig.add_trace(go.Scatter(x=[0, 1000], y=[0, 1000], mode="lines", name="1:1 line", line={"color": "black", "dash": "dash"}))
fig.update_layout(title_text=f"Observed vs. Estimated net radiation at {site}",
                  yaxis_range=[0, 1000], xaxis_range=[0, 1000],
                  xaxis_title="Estimated (W m-2)", yaxis_title="Observed (W m-2)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=sn_model[daytime], y=sn_obs[daytime], 
                         name="Sn", mode="markers"))
fig.add_trace(go.Scatter(x=[0, 1200], y=[0, 1200], mode="lines", name="1:1 line", line={"color": "black", "dash": "dash"}))
fig.update_layout(title_text=f"Observed vs. Estimated net shortwave radiation from 3-source model at {site}",
                  yaxis_range=[0, 1200], xaxis_range=[0, 1200],
                  xaxis_title="Estimated (W m-2)", yaxis_title="Observed (W m-2)")
```

## Computing Sensible heat flux with 3SEB

The key distinction of 3SEB is that we need to separate sensible heat fluxes between vegetation overstory, understory and soil layers. All this, using the the radiometric land surface temperature (LST) derived from remote sensing as the key boundary condition. It is assumed that the singular LST observation is the combination of signals stemming from all three sources such as :

:::{math}
:label:eq-3SEB_Trad
LST =(f_{ov}\left(\theta\right)T_{ov}^4+\left[1-f_{ov}\left(\theta\right)\right]T_{sub}^4)^{1/4}\\

\\T_{sub} =(f_{un}\left(\theta\right)T_{un}^4+\left[1-f_{un}\left(\theta\right)\right]T_{s}^4)^{1/4}\\

:::
where $f_c\left(\theta\right)$ is the fraction of vegetation observed by the sensor (either for understory (un) and overstory (ov)). Since the above equations consists of three unknowns and only two equations, an iterative process is conducted using two initial guesses of potential canopy transpiration for both vegetation sources, and under the assumption that during daytime hours condensation for the soil/substrate should not occur. The initial canopy latent and sensible heat fluxes are estimated based on the [Priestley and Taylor (1972)](https://doi.org/10.1175/1520-0493(1972)100<0081:OTAOSH>2.3.CO;2) formulation for potential transpiration (Eq. [](#eq-TSEB_PT)) and Eq. [](#eq-eb) as discussed above for TSEB.


In 3SEB, resistance framework is first treated as a parallel (i.e., uncoupled) tree-substrate system to obtain tree canopy sensible heat flux ($H_C$) and substrate (understory vegetation+soil) ($H_{sub}$) using the heat transport equations:

:::{math}
:label:ec-hc-hsub
H_C &= \rho C_p \frac{T_C - T_{a}} {R_A}\\
H_{sub} &= \rho C_p \frac{T_{sub} - T_{a}} {R_A+R_{sub}}
:::


Subsequently, the substrate fluxes and temperatures are further separated incorporating a series (i.e. coupled) approach:

:::{math}
:label:ec-hcsub-hs
H_{C,sub} &= \rho C_p \frac{T_{C,sub} - T_{ac}} {R_X}\\
H_s &= \rho C_p \frac{T_S - T_{ac}} {R_S}
:::


Since $T_C$, $T_{C,sub}$ and $T_S$ are unknown apriori, the original 3SEB formulation (3SEB-PT) implements a Priestley-Taylor (PT) formulation, as in [Norman et al. (1995)](https://doi.org/10.1016/0168-1923(95)02265-Y), to compute a first estimate of the canopy LE and H for both overstory and understory. 

:::{note}
#### This is essentially the same approach as described above for TSEB-PT but applied twice, one for overstory-substrate system and another for understory-soil substrate system,

:::

+++

## py3SEB implementation example

We will now show an example on how to run py3SEB with satellite imagery. We will run the model with an example imagery from the Sentinel constellation over the Majadas experimental site (ES-LMa/ES-LM1/ES-LM2). 

For this example, we will only use the land surface temperature (LST) and leaf area index (LAI) products from the Sentinel constellation. The LST was obtained from Sentinel-3 SLSTR sensor and was sharpened to 20m using the Sentinel-2 MSI sensor and a Data mininng Sharpenign algorithm ([pyDMS](https://github.com/radosuav/pyDMS)). While the LAI products were estimated by inverting a radiative transfer model (i.e. [pyProSail](https://github.com/hectornieto/pypro4sail) ) using a hybrid appraoch. These processing algorithms are described in [Guzinski et al. 2023](https://www.sciencedirect.com/science/article/pii/S1569843223004119) and [Guzinski et al. 2025](https://doi.org/10.5194/egusphere-2025-4342)

:::{seealso}
For more in depth tutorials on how pre-process cloud-based Sentinel inputs and run 3SEB, please see [pyseb-copernicus](https://github.com/VicenteBurchard/py3seb-copernicus) which is based on the [Sen-ET toolbox](https://github.com/DHI/Sen-ET-OpenEO-toolbox)
:::

:::{note}
We will use the same Sentinel images and meteo data as shown above for TSEB
:::

+++

### Biophysical and structural parameters
For simplicity, we wiill assume a constant $LAI_{ov}$, $f_{c,ov}$ and $h_c$ across the image window. In different applications, other spatial or satellite-based estimates could also be used to inform these variables such as tree fractional cover products (e.g., [Copernicus Tree Cover](https://land.copernicus.eu/en/products/high-resolution-layer-forests-and-tree-cover) or canopy height models from LiDAR such as [GEDI](https://gedi.umd.edu/dataproducts/products/)).

```{code-cell} ipython3
# setting constant values across image array
# overstory
lai_ov = np.full_like(lst_ar, 0.4) # acquried from LAI decomposition (see above)
fc_ov_ar = np.full_like(lst_ar, f_c_ov) # average tree fractional cover
hc_ov_ar = np.full_like(lst_ar, h_c_ov) # average tree height
    
# understory
lai_un = (lai_ar - lai_ov) /(1-f_c_ov)
lai_un[lai_un<0.3] = 0.3
fc_un_ar = np.full_like(lst_ar, f_c_un)# homogeneous grass cover
hc_un_ar = np.full_like(lst_ar, h_c_un) # grass cover height

# get sun angles during S3 overpass datetime
sza, saa = TSEB.met.calc_sun_angles(lat,
                                    lon,
                                    stdlon,
                                    s3_overpass_datetime.dayofyear,
                                    s3_overpass_datetime.hour + s3_overpass_datetime.minute / 60.)
# local/real LAI of overstory
F_ov = lai_ov/ fc_ov_ar
# clulmping of overstory
omega_0 = TSEB.CI.calc_omega0_Kustas(lai_ov, fc_ov_ar, x_LAD=1)
omega = TSEB.CI.calc_omega_Kustas(omega_0, np.minimum(sza, 90), w_C=w_c)
# effective LAI of overstory
lai_ov_eff = F_ov * omega

## roughness parameters
# Overstory (i.e. Evergreen trees)
[z_0m_ov, d_0_ov] = TSEB.res.calc_roughness(lai_ov,
                                            hc_ov_ar, 
                                            np.full_like(lai_ov, w_c),
                                            np.full_like(lai_ov, TSEB.res.BROADLEAVED_E),
                                            f_c=fc_ov_ar)
# Ensure realistic values
d_0_ov[d_0_ov < 0] = 0
z_0m_ov[np.isnan(z_0m_ov)] = z0_soil
d_0_ov[d_0_ov < z0_soil] = z0_soil

# Understory (i.e. grasses)
# ==========
z_0m_un, d_0_un = TSEB.res.calc_roughness(lai_un, 
                                          hc_un_ar,
                                          np.full_like(lai_un, w_c),
                                          np.full_like(lai_un, TSEB.res.GRASS),
                                          f_c=fc_un_ar)

# Ensure realistic values
d_0_un[d_0_un < 0] = 0
z_0m_un[np.isnan(z_0m_un)] = z0_soil
z_0m_un[z_0m_un < z0_soil] = z0_soil
print("Parsed biophysical and structural parameters correctly, you can continue to next cell")
```

## Running py3SEB with Satellite inputs

```{code-cell} ipython3
print('Estimating net shortwave radiation [...]\n')

# Estimates the direct and diffuse solar radiation
difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(np.full_like(lst_ar, sw_in),
                                                        np.full_like(lst_ar, sza),
                                                        press=np.full_like(lst_ar, P))
par_dir = fvis * (1. - difvis) * np.full_like(lst_ar, sw_in)
nir_dir = fnir * (1. - difnir) * np.full_like(lst_ar, sw_in)
par_dif = fvis * difvis * np.full_like(lst_ar, sw_in)
nir_dif = fnir * difnir * np.full_like(lst_ar, sw_in)


# net radiation transmission through the three sources
sn_c_ov, sn_s, sn_c_un = py3seb.calc_Sn_Campbell(lai_ov, # LAI of overstory
                                                 lai_un, # LAI of understory
                                                  np.full_like(lst_ar, sza), # sun zenith angle
                                                  par_dir + nir_dir, # direct incoming irradiance
                                                  par_dif + nir_dif,  # diffuse incoming irradiance
                                                  fvis, # fration of total visible radiation
                                                  fnir, # fration of total NIR radiation
                                                  np.full_like(lst_ar, 0.05), # Overstory Broadband leaf reflectance in the visible region (400-700nm)
                                                  np.full_like(lst_ar, 0.05), # Understory Broadband leaf reflectance in the visible region (400-700nm)
                                                  np.full_like(lst_ar, 0.08), # Overstory Broadband leaf transmittance in the visible region (400-700nm)
                                                  np.full_like(lst_ar, 0.08), # Understory Broadband leaf transmittance in the visible region (400-700nm)
                                                  np.full_like(lst_ar, 0.26), # Overstory Broadband leaf reflectance in the NIR region (700-2500nm)
                                                  np.full_like(lst_ar, 0.26), # Understory Broadband leaf reflectance in the NIR region (700-2500nm)
                                                  np.full_like(lst_ar, 0.33), # Overstory Broadband leaf transmittance in the NIR region (700-2500nm)
                                                  np.full_like(lst_ar, 0.33), # Understory Broadband leaf transmittance in the NIR region (700-2500nm)
                                                  np.full_like(lst_ar, 0.07), # Soil Broadband reflectance in the visible region (400-700nm)
                                                  np.full_like(lst_ar, 0.28), # Soil Broadband reflectance in the NIR region (700-2500nm)
                                                  hc_ov_ar, # overstory canopy height
                                                  hc_ov_ar * 0.5, # ratio of hc where leaf canopy begins in overstory layer
                                                  w_c, # width to height ratio of overstory
                                                  fc_ov_ar, # Fractional cover of overstory
                                                  x_LAD=1, # Overstory x parameter for the ellipsoildal Leaf Angle Distribution function of Campbell 1988
                                                  x_LAD_sub=1, # Understory x parameter for the ellipsoildal Leaf Angle Distribution function of Campbell 1988
                                                  LAI_eff=lai_ov,# Effective LAI of overstory
                                                  LAI_eff_sub=lai_un) # Effective LAI of understory

vza = np.full_like(lst_ar, 21.88)  # from S3-SLSTR product
      
print('Running 3SEB [...]\n')
# using constant ratio appraoch to estimate G
G_constant = 0.35
calcG = [[1], G_constant] 

# use Norman and Kustas 1995 resistance framework
Resistance_flag=[0,{}]

[flag_PT_all, T_S, T_C_ov, T_C_un, T_AC, Ln_sub, Ln_C_ov, Ln_C_un, Ln_S, LE_C_ov, H_C_ov, LE_C_un, H_C_un,
 LE_S, H_S, G_mod, R_S, R_sub, R_X, R_A, u_friction, L, n_iterations] = py3seb.ThreeSEB_PT_PT_SEQ(lst_ar,
                                                                                         vza,
                                                                                         np.full_like(lst_ar, Ta),
                                                                                         np.full_like(lst_ar, u),
                                                                                         np.full_like(lst_ar, ea),
                                                                                         np.full_like(lst_ar, P),
                                                                                         sn_c_ov,
                                                                                         sn_s,
                                                                                         sn_c_un,
                                                                                         np.full_like(lst_ar, lw_in),
                                                                                         lai_ov,
                                                                                         lai_un,
                                                                                         hc_ov_ar,
                                                                                         hc_un_ar,
                                                                                         e_v,
                                                                                         e_v,
                                                                                         e_s,
                                                                                         z_0m_ov,
                                                                                         z_0m_un,
                                                                                         d_0_ov,
                                                                                         d_0_un,
                                                                                         z_u,
                                                                                         z_t,
                                                                                         leaf_width=leaf_width_ov,
                                                                                         leaf_width_sub=leaf_width_un,
                                                                                         f_c=fc_ov_ar,
                                                                                         f_c_sub=fc_un_ar,
                                                                                         f_g=np.full_like(lst_ar,0.9),
                                                                                         f_g_sub=np.full_like(lst_ar,0.75),
                                                                                         calcG_params=calcG,
                                                                                         resistance_form=Resistance_flag)
# save ouputs in outdict 
LE = LE_C_ov + LE_C_un + LE_S
H = H_C_ov + H_C_un + H_S

Rn_C = Ln_C_ov + sn_c_ov
Rn_C_sub = Ln_C_un + sn_c_un
Rn_S = Ln_S + sn_s
Rn = Rn_C + Rn_C_sub + Rn_S



fig = make_subplots(
    rows=1,
    cols=3,
    horizontal_spacing=0.03,
    subplot_titles=("LE", "H", "Rn"))

# LE
fig.add_trace(
    go.Heatmap(
        z=LE,
        colorscale="Blues",
        zmin=0,
        zmax=400,
        colorbar=dict(
            title="LE (W/m²)",
            orientation="h",
            x=0.13,     
            y=-0.2,     
            len=0.3,    
            thickness=15)),
    row=1,col=1)

# H
fig.add_trace(
    go.Heatmap(
        z=H,
        zmin=0,
        zmax=400,
        colorscale="Reds",
        colorbar=dict(
            title="H (W/m²)",
            orientation="h",
            x=0.48,      
            y=-0.2,    
            len=0.3,    
            thickness=15)),  
    row=1, col=2)

# Rn
fig.add_trace(
    go.Heatmap(
        z=Rn,
        zmin=0,
        zmax=600,
        colorscale="Oranges",
        colorbar=dict(
            title="Rn (W/m²)",
            orientation="h",
            x=0.82,      
            y=-0.2,     
            len=0.3,    # colorbar length
            thickness=15)),  
    row=1, col=3)


fig.update_layout(
    title="Modelled instananeous fluxes from 3SEB",
    height=500,
    width=1500)

# Make pixels square like imshow
fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)
fig.update_yaxes(autorange="reversed", scaleanchor="x2", row=1, col=2)
fig.update_yaxes(autorange="reversed", scaleanchor="x3", row=1, col=3)
```

## Flux partitioning between sources
With 3SEB we have the added value of being able to estimate the partionning between the two main vegetation components (i.e. trees and grasses)

```{code-cell} ipython3

fig = make_subplots(
    rows=1,
    cols=3,
    horizontal_spacing=0.03,
    subplot_titles=("LE_ov", "LE_un", "LE_s"))

# LE
fig.add_trace(
    go.Heatmap(
        z=LE_C_ov,
        colorscale="Blues",
        zmin=0,
        zmax=300,
        colorbar=dict(
            title="LE_ov (W/m²)",
            orientation="h",
            x=0.13,     
            y=-0.2,     
            len=0.3,    
            thickness=15)),
    row=1,col=1)

# H
fig.add_trace(
    go.Heatmap(
        z=LE_C_un,
        zmin=0,
        zmax=300,
        colorscale="Blues",
        colorbar=dict(
            title="LE_un (W/m²)",
            orientation="h",
            x=0.48,      
            y=-0.2,    
            len=0.3,    
            thickness=15)),  
    row=1, col=2)

# Rn
fig.add_trace(
    go.Heatmap(
        z=LE_S,
        zmin=0,
        zmax=300,
        colorscale="Blues",
        colorbar=dict(
            title="LE_s (W/m²)",
            orientation="h",
            x=0.82,      
            y=-0.2,     
            len=0.3,    # colorbar length
            thickness=15)),  
    row=1, col=3)


fig.update_layout(
    title="LE Partitioning from 3SEB",
    height=500,
    width=1500)

# Make pixels square like imshow
fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)
fig.update_yaxes(autorange="reversed", scaleanchor="x2", row=1, col=2)
fig.update_yaxes(autorange="reversed", scaleanchor="x3", row=1, col=3)
```

:::{hint}
**Questions:**
- How do the modelled fluxes compare to the tower measurements?
- How could we get a map of daily ET (mm/day) from these instananeous fluxes?
- Why do we not see so much spatial variability in $LE_{ov}$?
:::

+++

# Conclusions

* In this notebook we learn the basics of the low level code of pyTSEB to run TSEB-PT using as input one single value
* We showed an example of running TSEB-PT with spatialized inputs (e.g. satellite-based inputs) 
* We also learned the basics of the low level code of py3SEB and its main differences compared to TSEB
* For additional information regarding to derive all inputs required for TSEB/3SEB using Copernicus data from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) see [Sen-ET OpenEO Toolbox][https://github.com/DHI/Sen-ET-OpenEO-toolbox] and [py3seb-copernicus](https://github.com/VicenteBurchard/py3seb-copernicus)
