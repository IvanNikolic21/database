import os
from os import path
import time
import glob
#initialize environment variables
time_start = time.time()
#os.environ['OPENBLAS_NUM_THREADS'] = str(1)
#os.environ['OMP_NUM_THREADS'] = str(1)
import socket
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import numpy as np
from astropy import cosmology
import sys
sys.path.append('/mnt/lustre/users/inikoli/run_directory/')
import save as save
import logging
from ps import ps1D_coeval, ps2D_coeval, ps_coeval
import py21cmmc
import argparse
from py21cmmc import mcmc
from py21cmmc import LikelihoodNeutralFraction
from py21cmmc import CoreLightConeModule
from py21cmmc import LikelihoodLuminosityFunction, CoreLuminosityFunction
from py21cmmc import LikelihoodForest, CoreForest
from py21cmmc import LikelihoodPlanck
from py21cmmc import LikelihoodBase
from py21cmmc import CoreCoevalModule
from py21cmfast import AstroParams, CosmoParams
from py21cmfast._utils import InfinityorNaNError
#set logger from 21cmFAST

logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default='/home/inikoli/lustre/run_directory/production_database/')
parser.add_argument("--cache_dir", type=str, default='/home/inikoli/lustre/run_directory/_cache/')
parser.add_argument("--threads", type=int, default=12)
parser.add_argument("--params_dir", type=str, default='/home/inikoli/')
inputs = parser.parse_args()

#main parameter combinations:
user_params = {
    'HII_DIM':200,
    'BOX_LEN':400.0,  #note the change
    'USE_INTERPOLATION_TABLES': True,
    'USE_FFTW_WISDOM': True,
    'PERTURB_ON_HIGH_RES': True,
    'N_THREADS': inputs.threads,
    'OUTPUT_ALL_VEL': False,  #for kSZ need to save all velocity components.
    'USE_RELATIVE_VELOCITIES' : True,
    'POWER_SPECTRUM': 5,
}

cosmo_params = { 
    'hlittle': 0.6688,
    'OMm': 0.321,
    'OMb':0.04952,
    'POWER_INDEX':0.9626,
}

flag_options = {
    'USE_MASS_DEPENDENT_ZETA': True,
    'INHOMO_RECO': True,
    'PHOTON_CONS': False,#for now
    'EVOLVING_R_BUBBLE_MAX': False, #This parameter is not present in master!
    'USE_TS_FLUCT': True,
    'USE_MINI_HALOS': True,
}

global_params = {
#    'Z_HEAT_MAX': 15.0, 
    'T_RE': 2e4,
    'ALPHA_UVB': 2.0,
    'PhotonConsEndCalibz':3.5
}

### tau_GP function is the exact replica of 21cmmc.
def tau_GP(gamma_bg, delta, temp, redshifts, cosmo_params):
    r"""Calculating the lyman-alpha optical depth in each pixel using the fluctuating GP approximation.
    Parameters
    ----------
    gamma_bg : float or array_like
        The background photonionization rate in units of 1e-12 s**-1
    delta : float or array_like
        The underlying overdensity
    temp : float or array_like
        The kinectic temperature of the gas in 1e4 K
    redshifts : float or array_like
        Correspoding redshifts along the los
    """
    gamma_local = np.zeros_like(gamma_bg)
    residual_xHI = np.zeros_like(gamma_bg, dtype=np.float64)

    flag_neutral = gamma_bg == 0
    flag_zerodelta = delta == 0

    if gamma_bg.shape != redshifts.shape:
        redshifts = np.tile(redshifts, (*gamma_bg.shape[:-1], 1))

    delta_ss = (
            2.67e4 * temp ** 0.17 * (1.0 + redshifts) ** -3 * gamma_bg ** (
                2.0 / 3.0)
    )
    gamma_local[~flag_neutral] = gamma_bg[~flag_neutral] * (
            0.98
            * (
                    (1.0 + (delta[~flag_neutral] / delta_ss[
                        ~flag_neutral]) ** 1.64)
                    ** -2.28
            )
            + 0.02 * (1.0 + (
                delta[~flag_neutral] / delta_ss[~flag_neutral])) ** -0.84
    )

    Y_He = 0.245
    # TODO: use global_params
    residual_xHI[~flag_zerodelta] = 1 + gamma_local[
        ~flag_zerodelta] * 1.0155e7 / (
                                            1.0 + 1.0 / (4.0 / Y_He - 3)
                                    ) * temp[~flag_zerodelta] ** 0.75 / (
                                            delta[~flag_zerodelta] * (
                                                1.0 + redshifts[
                                            ~flag_zerodelta]) ** 3
                                    )
    residual_xHI[~flag_zerodelta] = residual_xHI[~flag_zerodelta] - np.sqrt(
        residual_xHI[~flag_zerodelta] ** 2 - 1.0
    )

    return (
            7875.053145028655
            / (
                    cosmo_params.hlittle
                    * np.sqrt(
                cosmo_params.OMm * (1.0 + redshifts) ** 3
                + cosmo_params.OMl
            )
            )
            * delta
            * (1.0 + redshifts) ** 3
            * residual_xHI
    )

import py21cmfast as p21c
from py21cmfast import wrapper as lib

my_cache=inputs.cache_dir    #update this to the desired _cache directory.
if not os.path.exists(my_cache):
    os.mkdir(my_cache)

#p21c.config['direc'] = my_cache

#######################
model_name = "database"
#######################


cosmo = cosmology.FlatLambdaCDM(
    H0=cosmo_params['hlittle']*100, 
    Om0=cosmo_params['OMm'],
    Tcmb0=2.725,
    Ob0=cosmo_params['OMb']
)

#prior of rescale parameters.
class Prior(LikelihoodBase):
    def reduce_data(self, ctx):
        params = ctx.getParams()
        return  [params.log10_f_rescale, params.f_rescale_slope]

    def computeLikelihood(self, model):
        return -0.5 * ((model[0] / 5)**2 + (model[1] / 2.5)**2)

mag_brightest = -20.0
lf_zs_saved = [6,7,8,9,10,12,15]
lf_zs = [6, 7, 8, 10] 
forest_zs = [5.4, 5.6, 5.8, 6.0] # note the change in redshifts
coeval_zs = [5,6,7,8,9,10]
container = None

n_muv_bins = 100 # for uvlf
n_realization = 150 #for forest
observation="bosman_optimistic"

data_forest = np.load(
        path.join(path.dirname(py21cmmc.__file__), "data/Forests/Bosman18/data.npz"),
        allow_pickle=True,
    )

tau_mean_CMB = 0.0569   #CMB data
tau_sigma_u_CMB = 0.0073
tau_sigma_l_CMB = 0.0066
n_z_interp = 25
z_extrap_min = 5.0
z_extrap_max = 30.0

while True:
    seed_now = np.random.randint(low=0, high=2**22-1)
    np.random.seed(seed = seed_now)
    model_name = "database" + str(seed_now)
    p21c.global_params.Z_HEAT_MAX = 35.0 #this is creating problems with max_redshift

    params_full = np.loadtxt(inputs.params_dir + 'params.txt')
    index_this = np.random.randint(0, np.shape(params_full)[0])
    params_this = params_full[index_this]

    cosmo = cosmology.FlatLambdaCDM(H0=cosmo_params['hlittle']*100, Om0=cosmo_params['OMm'],Tcmb0=2.725,Ob0=cosmo_params['OMb'])
    lightcone_quantities = (
       "xH_box",
       "density",
       "velocity",
       "brightness_temp",
       "temp_kinetic_all_gas",
       "Gamma12_box",
       "Ts_box",
    )
    output_dir = inputs.output_dir
    global_quantities = list(lightcone_quantities) + ["dNrec_box", "MFP_box"]
    astro_params = {
        'F_STAR10' : params_this[0],
        'ALPHA_STAR' : params_this[1],
        't_STAR' : params_this[2],
       # 'M_TURN' : params_this[3],
        'F_ESC10' : params_this[3],
        'ALPHA_ESC' : params_this[4],
        'F_STAR7_MINI' : params_this[6],
        'F_ESC7_MINI' : params_this[7],
        'L_X' : params_this[8],
        'NU_X_THRESH' : params_this[9],
        'L_X_MINI' : params_this[10],
    }
#    astro_params = {
#        'F_STAR10' : -1.30,
#        'ALPHA_STAR' : 0.5,
#        't_STAR' : 0.44,
#        'F_ESC10': -1.3,
#        'ALPHA_ESC' : -0.1 + (np.random.uniform()-0.5),
    #    'F_STAR7_MINI' : -2.20,
    #    'F_ESC7_MINI' : -2.1,
#        'L_X' : 41.0,
#        'NU_X_THRESH' : 700,
    #}
    #cosmo_params['SIGMA_8'] = params_this[5]
    #log10_f_rescale_now = params_this[10]
    #f_rescale_slope_now = params_this[11]
    log10_f_rescale_now = None
    f_rescale_slope_now = None
    cosmo_params['SIGMA_8'] = params_this[5]
    parameter_names = list(astro_params.keys()) + ['SIGMA_8', 'log10_f_rescale', 'f_rescale_slope']
    astro_params_now = AstroParams(astro_params)
    cosmo_params_now = CosmoParams(cosmo_params)
    try:
        
        if not container.check_params(astro_params,
                                        cosmo_params):
            container = None
            container = save.HDF5saver(
                astro_params_now,
                cosmo_params_now,
                parameter_names,
                output_dir,
                log10_f_rescale_now,
                f_rescale_slope_now
            )
    except (NameError,AttributeError) as e:

        container = save.HDF5saver(
            astro_params_now,
            cosmo_params_now,
            parameter_names,
            output_dir,
            log10_f_rescale_now,
            f_rescale_slope_now
        )
    if not container.exists():
        container.create()
    container.add_hostname(socket.gethostname())

    init_seed_now = np.random.randint(low=0, high=2**32-1)
    container.add_rstate(init_seed_now)

    my_cache_now = my_cache + '/' + str(init_seed_now)
    if not os.path.exists(my_cache_now):
        os.mkdir(my_cache_now)

    p21c.config['direc'] = my_cache_now

    print("starting to run coeval for this params", astro_params_now)
    try:
        coeval = p21c.run_coeval(
            redshift=coeval_zs,
            astro_params=astro_params_now,
            cosmo_params=cosmo_params_now,
            flag_options=flag_options,
            user_params=user_params,
            regenerate=False,
            random_seed=init_seed_now,
            write=my_cache_now,
            direc=my_cache_now,
            **global_params,
        )
    except InfinityorNaNError:
        print("coevals failed with these params", astro_params_now)
        continue
    try:
        for z, c in enumerate(coeval):
            container.add_coevals(coeval_zs[z], c)
    except ValueError:
        print("Already sampled")
        continue
    del coeval      
    print("ended coeval, starting lightcone with these astros", astro_params_now)
    try:
        lightcone,PS = p21c.run_lightcone(
            redshift=4.9,
            max_redshift=35, #note the change
            user_params=user_params,
            cosmo_params=cosmo_params_now,
            astro_params=astro_params_now,
            flag_options=flag_options,
            #rotation_cubes=False,
            coeval_callback=lambda x: ps_coeval(x, 50),
            lightcone_quantities=lightcone_quantities,
            random_seed=init_seed_now,
            global_quantities=global_quantities,
            write = False,
            direc = my_cache_now,
            **global_params,
        )
    except InfinityorNaNError:
        print("There was some error with photon-conservation", astro_params_now)
        continue
    print("ended lightcone, starting frest")
    if lightcone is not None:
        container.add_PS(
            PS, lightcone.node_redshifts  # post-processing is done in save.py
        )
        container.add_global_xH(
            lightcone.global_xH
        )
        container.add_lightcones(lightcone)
        container.add_lightcone_redshifts(lightcone.lightcone_redshifts)
####LF part

    for index_uv, z_uv in enumerate(lf_zs_saved):
        mturnovers = 10 ** interp1d(np.array(lightcone.node_redshifts)[::-1], np.array(lightcone.log10_mturnovers))(
            z_uv
        )
        mturnovers_mini = 10 ** interp1d(
                np.array(lightcone.node_redshifts)[::-1], np.array(lightcone.log10_mturnovers_mini)[::-1]
            )(z_uv)

        if not hasattr(mturnovers, '__len__'):
            mturnovers = [mturnovers]
        if not hasattr(mturnovers_mini, '__len__'):
            mturnovers_mini = [mturnovers_mini]

        Muv, mhalo, lfunc = p21c.compute_luminosity_function(
                mturnovers=mturnovers,
                mturnovers_mini=mturnovers_mini,
                redshifts=[z_uv],
                astro_params=astro_params_now,
                flag_options=flag_options,
                cosmo_params=cosmo_params_now,
                user_params=user_params,
                nbins=n_muv_bins,
            )

        Muv = [m[~np.isnan(lf)] for lf, m in zip(lfunc, Muv)]
        mhalo = [m[~np.isnan(lf)] for lf, m in zip(lfunc, mhalo)]
        lfunc = [m[~np.isnan(lf)] for lf, m in zip(lfunc, lfunc)]

        container.add_UV((Muv, lfunc, mhalo), z_uv)

        if z_uv in lf_zs:
            datafile = [
                path.join(
                    path.dirname(py21cmmc.__file__),
                    "data",
                    "LF_lfuncs_z%d.npz" % z_uv,
                )
            ]
            noisefile = [
                path.join(
                    path.dirname(py21cmmc.__file__),
                    "data",
                    "LF_sigmas_z%d.npz" % z_uv,
                )
            ]
            data = []
            for fl in datafile:
                if not path.exists(fl):
                    raise ValueError
                else:
                    data.append(dict(np.load(fl, allow_pickle=True)))
            noise = []
            for fl in noisefile:
                if not path.exists(fl):
                    raise ValueError
                else:
                    noise.append(dict(np.load(fl, allow_pickle=True)))
            for key, val in data[0].items():
                if not hasattr(val[0], "__len__"):
                    data[0][key] = np.atleast_2d(val)
            for key, val in noise[0].items():
                if not hasattr(val[0], "__len__"):
                    noise[0][key] = np.atleast_2d(val)

            data = data[0]
            noise = noise[0]
            lnl = 0
            try:
                model_spline =  InterpolatedUnivariateSpline(
                    Muv[0][::-1], lfunc[0][::-1]
                )
            except ValueError:    #quick hack for interpolation, don't know why it's different from 21CMMC
                model_spline = InterpolatedUnivariateSpline(
                    Muv[0][::1], lfunc[0][::1]
                )
            lnl += -0.5 * np.sum(
                (
                    (data["lfunc"] - 10 ** model_spline(data["Muv"]))
                    ** 2
                    / noise["sigma"] ** 2
                )[data["Muv"] > mag_brightest]
            )
            container.add_uvlf_likelihood(lnl, z_uv)
            print("This it uvlnl for ", z_uv, ":" ,lnl)


####FOREST

    for index_forest, z_forest in enumerate(forest_zs):
        targets = (data_forest["zs"] > z_forest - 0.1) * (
                data_forest["zs"] <= z_forest + 0.1
        )
        nlos = sum(targets)
        bin_size = 50 / cosmo_params_now.hlittle

        if lightcone is not None:
            lightcone_redshifts = lightcone.lightcone_redshifts
            lightcone_distances = lightcone.lightcone_distances
            total_los = lightcone.user_params.HII_DIM ** 2
            index_right = np.where(
                lightcone_distances
                > (
                        lightcone_distances[
                            np.where(lightcone_redshifts > z_forest)[0][
                                0]
                        ]
                        + bin_size / 2
                )
            )[0][0]
            index_left = np.where(
                lightcone_distances
                > (
                        lightcone_distances[
                            np.where(lightcone_redshifts > z_forest)[0][
                                0]
                        ]
                        - bin_size / 2
                )
            )[0][0]
            if index_left == 0:
                index_right = np.where(
                    lightcone_distances > (lightcone_distances[0] + bin_size)
                )[0][0]

            tau_eff = np.zeros([n_realization, nlos])

            if log10_f_rescale_now is not None:
                f_rescale_proper = 10 ** log10_f_rescale_now
                f_rescale_proper += (z_forest-5.7) * f_rescale_slope_now
            else:
                f_rescale_proper = 1.0

            for jj in range(n_realization):
                gamma_bg = lightcone.Gamma12_box[:, :, index_left:index_right].reshape(
                    [total_los, index_right - index_left]
                )[jj:: int(total_los / nlos)][: nlos]
                delta = (
                        lightcone.density[:, :, index_left:index_right].reshape(
                            [total_los, index_right - index_left]
                        )[jj:: int(total_los / nlos)][: nlos]
                        + 1.0
                )
                temp = (
                        lightcone.temp_kinetic_all_gas[:, :,
                        index_left:index_right].reshape(
                            [total_los, index_right - index_left]
                        )[jj:: int(total_los / nlos)][: nlos]
                        / 1e4
                )
                tau_lyman_alpha = tau_GP(
                    gamma_bg, delta, temp,
                    lightcone_redshifts[index_left:index_right], cosmo_params_now
                )

                tau_eff[jj] = -np.log(
                    np.mean(np.exp(-tau_lyman_alpha * f_rescale_proper), axis=1))

            tau_Lyman_alpha_lc = tau_GP(
                lightcone.Gamma12_box,
                lightcone.density,
                lightcone.temp_kinetic_all_gas,
                lightcone_redshifts,
                cosmo_params_now
            )

            container.add_tau_lc(
                tau_Lyman_alpha_lc
            )

        tau_range = [0,8]
        hist_bin_width = 0.1
        hist_bin_size = int(
            (tau_range[1] - tau_range[0]) / hist_bin_width
        )
        datafile = [
            path.join(path.dirname(py21cmmc.__file__), "data/Forests/Bosman18/data.npz")
        ]
        noisefile = [
            path.join(
                path.dirname(py21cmmc.__file__),
                "data/Forests/Bosman18/PDF_ErrorCovarianceMatrix_GP/z%s.npy"
                % str(z_forest).replace(".", "pt"),
            )
        ]
        data  = []
        for fl in datafile:
            if not path.exists(fl):
                raise ValueError
            else:
                data.append(dict(np.load(fl, allow_pickle=True)))
        targets = np.where(
            (data[0]["zs"] > (z_forest - 0.1))
            * (data[0]["zs"] <= (z_forest + 0.1))
        )[0]
        pdfs_data = np.zeros([2, hist_bin_size])

        pdfs_data[0] = np.histogram(
            data[0]["tau_lower"][targets], range=tau_range,
            bins=hist_bin_size
        )[0]

        pdfs_data[1] = np.histogram(
            data[0]["tau_upper"][targets], range=tau_range,
            bins=hist_bin_size
        )[0]
        noise = []
        for fl in noisefile:
            if not path.exists(fl):
                raise ValueError
            else:
                try:
                    noise.append(dict(np.load(fl, allow_pickle=True)))
                except ValueError:
                    noise.append(np.load(fl, allow_pickle=True))
        ErrorCovarianceMatrix_GP = noise[0]

        pdfs_model = np.zeros([n_realization, hist_bin_size])
        for jj in range(n_realization):
            pdfs_model[jj] = np.histogram(
                tau_eff[jj], range=tau_range, bins=hist_bin_size
            )[0]
        tau_eff_model = np.mean(pdfs_model, axis=0)
        ecm_cosmic = np.cov(pdfs_model.T)
        noise = (
            ErrorCovarianceMatrix_GP + ecm_cosmic + np.diag(np.ones(hist_bin_size) * 1e-5)
        )
        container.add_forest_pdfs(pdfs_model, z_forest)
        det = np.linalg.det(noise)
        if det == 0 or tau_eff is None:
            lnl = -np.inf
        diff = tau_eff_model - pdfs_data[0]
        for ii in np.where(pdfs_data[0] != pdfs_data[1])[0]:
            if tau_eff_model[ii] < pdfs_data[0][ii]:
                diff[ii] = min(0, tau_eff_model[ii] - pdfs_data[1][ii])
        diff = diff.reshape([1, -1])
        lnl = (
            -0.5 * np.linalg.multi_dot([diff, np.linalg.inv(noise), diff.T])[0, 0]
        )
        if det < 0:
            lnl = -np.inf
        container.add_forest_likelihood(lnl, z_forest)

###STARTING CMB
    if lightcone is not None:
        redshifts_CMB, xHI_CMB = np.sort(np.array([lightcone.node_redshifts, lightcone.global_xHI]))
        neutral_frac_func = InterpolatedUnivariateSpline(redshifts_CMB, xHI_CMB, k=1)
        z_extrap = np.linspace(z_extrap_min, z_extrap_max, n_z_interp)
        xHI_CMB = neutral_frac_func(z_extrap)
        np.clip(xHI_CMB, 0, 1, xHI_CMB)
        tau_value = lib.compute_tau(
            user_params=user_params,
            cosmo_params=cosmo_params_now,
            redshifts=z_extrap,
            global_xHI=xHI_CMB,
        )
        container.add_tau(tau_value)
        tau_e_likelihood = -0.5 * np.square(tau_mean_CMB - tau_value) / (
                tau_sigma_u_CMB * tau_sigma_l_CMB
                + (tau_sigma_u_CMB - tau_sigma_l_CMB) * (
                            tau_value - tau_mean_CMB)
        )
        container.add_tau_likelihood(tau_e_likelihood)

    ###END OF CMB, CLEANING EVERYTHING UP

    del container, lightcone
    container = None
    for file_rm in glob.glob(my_cache_now + '/*' + str(init_seed_now) + '.h5'):
        os.system('rm '+ file_rm)
    os.system('rmdir ' + my_cache_now)
