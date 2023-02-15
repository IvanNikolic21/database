import os
import time
import glob
#initialize environment variables
time_start = time.time()
#os.environ['OPENBLAS_NUM_THREADS'] = str(1)
#os.environ['OMP_NUM_THREADS'] = str(1)

import numpy as np
from astropy import cosmology
import sys
sys.path.append('/mnt/lustre/users/inikoli/run_directory/')
import save as save
import logging, sys

from py21cmmc import mcmc
from py21cmmc import LikelihoodNeutralFraction
from py21cmmc import CoreLightConeModule
from py21cmmc import LikelihoodLuminosityFunction, CoreLuminosityFunction
from py21cmmc import LikelihoodForest, CoreForest
from py21cmmc import LikelihoodPlanck
from py21cmmc import LikelihoodBase
from py21cmmc import CoreCoevalModule
from py21cmfast import AstroParams, CosmoParams
#set logger from 21cmFAST

logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

#main parameter combinations:
user_params = {
    'HII_DIM':128,
    'BOX_LEN':256.0,  #note the change
    'USE_INTERPOLATION_TABLES': True,
    'USE_FFTW_WISDOM': True,
    'PERTURB_ON_HIGH_RES': True,
    'N_THREADS': 16,
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
    'EVOLVING_R_BUBBLE_MAX': True, #This parameter is not present in master!
    'USE_TS_FLUCT': True,
    'USE_MINI_HALOS': True,
}

global_params = {
    'Z_HEAT_MAX': 15.0, 
    'T_RE': 2e4,
    'ALPHA_UVB': 2.0,
    'PhotonConsEndCalibz':3.5
}

import py21cmfast as p21c

my_cache='/home/inikoli/lustre/run_directory/_cache'    #update this to the desired _cache directory.
if not os.path.exists(my_cache):
    os.mkdir(my_cache)

p21c.config['direc'] = my_cache

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

lf_zs_saved = [6,7,8,9,10,12,15]
lf_zs = [6, 7, 8, 10] 
forest_zs = [5.4, 5.6, 5.8, 6.0] # note the change in redshifts
coeval_zs = [5,6,7,8,9,10]
container = None

while True:
    seed_now = np.random.randint(low=0, high=2**32-1)
    np.random.seed(seed = seed_now)
    model_name = "database" + str(seed_now)
    p21c.global_params.Z_HEAT_MAX = 15.0

    params_full = np.loadtxt('/home/inikoli/params.txt')
    index_this = np.random.randint(0, np.shape(params_full)[0])
    params_this = params_full[index_this]

    cosmo = cosmology.FlatLambdaCDM(H0=cosmo_params['hlittle']*100, Om0=cosmo_params['OMm'],Tcmb0=2.725,Ob0=cosmo_params['OMb'])
    lightcone_quantities = (
       "xH_box",
       "density",
       "velocity",
       "brightness_temp",
    )
    output_dir = '/home/inikoli/lustre/run_directory/output/'

    astro_params = {
        'F_STAR10' : params_this[0],
        'ALPHA_STAR' : params_this[1],
        't_STAR' : params_this[2],
        'F_ESC10' : params_this[3],
        'ALPHA_ESC' : params_this[4],
        'F_STAR7_MINI' : params_this[6],
        'F_ESC7_MINI' : params_this[7],
        'L_X' : params_this[8],
        'NU_X_THRESH' : params_this[9],
    }
#    astro_params = {
#        'F_STAR10' : -1.30,
#        'ALPHA_STAR' : 0.5,
#        't_STAR' : 0.44,
#        'F_ESC10': -1.3,
#        'ALPHA_ESC' : -0.1,
#        'F_STAR7_MINI' : -2.20,
#        'F_ESC7_MINI' : -2.1,
#        'L_X' : 41.0,
#        'NU_X_THRESH' : 700,
#    }
    cosmo_params['SIGMA_8'] = params_this[5]
    log10_f_rescale_now = params_this[10]
    f_rescale_slope_now = params_this[11]
    #log10_f_rescale_now = 0.0
    #f_rescale_slope_now = 0.0
    #cosmo_params['SIGMA_8'] = 0.8118
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

    except AttributeError as e:

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

    init_seed_now = np.random.randint(low=0, high=2**32-1)
    container.add_rstate(init_seed_now)
    print("starting to run coeval")
    coeval = p21c.run_coeval(
        redshift=coeval_zs,
        astro_params=astro_params_now,
        cosmo_params=cosmo_params_now,
        flag_options=flag_options,
        user_params=user_params,
        regenerate=False,
        random_seed=init_seed_now,
        write=my_cache,
        direc=my_cache,
        **global_params,
    )
    for z, c in enumerate(coeval):
        container.add_coevals(self.redshift[z], c)
    print("ended coeval, starting lightcone")
    lightcone,PS = p21c.run_lightcone(
        redshift=4.9,
        max_redshift=15,
        user_params=user_params,
        cosmo_params=cosmo_params_now,
        astro_params=astro_params_now,
        flag_options=flag_options,
        rotation_cubes=False,
        coeval_callback=lambda x: ps_coeval(x, 50),
        lightcone_quantities=lightcone_quantities,
        random_seed=init_seed_now,
        global_quantities=lightcone_quantities,
        write = my_cache,
        direc = my_cache,
        **global_params,
    )
    print("ended lightcone, starting frest")
    if lightcone is not None:
        container.add_PS(
            PS, lightcone.node_redshifts  # post-processing is done in save.py
        )
        container.add_global_xH(
            lightcone.global_xH
        )
        container.add_lightcones(lightcone)


