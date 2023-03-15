import os
import time
import glob
#initialize environment variables
time_start = time.time()
#os.environ['OPENBLAS_NUM_THREADS'] = str(1)
#os.environ['OMP_NUM_THREADS'] = str(1)

import numpy as np
from astropy import cosmology
import logging, sys

from py21cmmc import mcmc
from py21cmmc import LikelihoodNeutralFraction
from py21cmmc import CoreLightConeModule
from py21cmmc import LikelihoodLuminosityFunction, CoreLuminosityFunction
from py21cmmc import LikelihoodForest, CoreForest
from py21cmmc import LikelihoodPlanck
from py21cmmc import LikelihoodBase
from py21cmmc import CoreCoevalModule
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
    'N_THREADS': 8,
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


while True:
    seed_now = int((time.time()- time_start) * 1000000)
    print(seed_now)
    np.random.seed(seed = seed_now)
    model_name = "database" + str(seed_now)
    core = [
        CoreLightConeModule(
            redshift=4.9,
            max_redshift=15,
            user_params=user_params,
            cosmo_params=cosmo_params,
            flag_options=flag_options,
            global_params=global_params,
            regenerate=False,
            initial_conditions_seed = seed_now,
            cache_dir=my_cache,
            output_dir = r'/home/inikoli/lustre/run_directory/output/',
            cache_mcmc=False)
        ,
    ] + [
#        CoreCoevalModule(
#            redshift = coeval_zs,
#            user_params = user_params,
#            cosmo_params = cosmo_params,
#            flag_options = flag_options,
#            global_params = global_params,
#            regenerate = False,
#            initial_conditions_seed  = seed_now,
#            cache_dir = my_cache,
#            output_dir = r'/home/inikoli/lustre/run_directory/output/',
#            cache_mcmc = False,
#        ), 
    ] + [
        CoreLuminosityFunction(
            redshift=redshift,
            sigma=0,
            name='lfz%d'%redshift,
            user_params=user_params,
            cosmo_params=cosmo_params,
            flag_options=flag_options,
            global_params=global_params,
            initial_conditions_seed = seed_now,
            regenerate=False,
            cache_dir=my_cache,
            output_dir = r'/home/inikoli/lustre/run_directory/output/',
            cache_mcmc=False)
        for redshift in lf_zs_saved #note that these are more than calculated in likelihood.
    ] + [
        CoreForest( 
            redshift=redshift,
            name='bosman%s'%(str(redshift).replace('.', 'pt')), 
            n_realization=150,
            user_params=user_params,
            cosmo_params=cosmo_params,
            flag_options=flag_options,
            global_params=global_params,
            initial_conditions_seed = seed_now,
            regenerate=False,
            cache_dir=my_cache,
            output_dir = r'/home/inikoli/lustre/run_directory/output/',
            cache_mcmc=False
        ) for redshift in forest_zs
    ]

    likelihood = [ 
        LikelihoodPlanck(),   # no LikelihoodNeutralFraction!
    ] + [
        LikelihoodLuminosityFunction(name='lfz%d'%redshift, simulate = False,)
        for redshift in lf_zs
    ] + [
        LikelihoodForest(name='bosman%s'%(str(redshift).replace('.', 'pt')))
        for redshift in forest_zs
    ]

    params_full = np.loadtxt('/home/inikoli/params.txt')
    index_this = np.random.randint(0, np.shape(params_full)[0])
    params_this = params_full[index_this]
    
    tolerance = 0.000001 #in general not important, but lower number are prefered.
    param_dict = {
        'F_STAR10' : [params_this[0], params_this[0]-tolerance, params_this[0]+tolerance, tolerance],
        'ALPHA_STAR': [params_this[1], params_this[1]-tolerance, params_this[1]+tolerance, tolerance],
        't_STAR' : [params_this[2], params_this[2]-tolerance, params_this[2]+tolerance, tolerance],        #removed M_turn as it's already calculated.
        'F_ESC10' : [params_this[3], params_this[3]-tolerance, params_this[3]+tolerance, tolerance],
        'ALPHA_ESC' : [params_this[4], params_this[4]-tolerance, params_this[4]+tolerance, tolerance],
        'SIGMA_8' : [params_this[5], params_this[5]-tolerance, params_this[5]+tolerance, tolerance], #Gaussian initiall ball here corresponds to Planck 68% CI
        'F_STAR7' : [params_this[6], params_this[6]-tolerance, params_this[6]+tolerance, tolerance],
        'F_ESC7' : [params_this[7], params_this[7]-tolerance, params_this[7]+tolerance, tolerance], #Based on YQ+20
        'L_X' : [params_this[8], params_this[8]-tolerance, params_this[8]+tolerance, tolerance],
        'NU_X_THRESH' : [params_this[9], params_this[9]-tolerance, params_this[9]+tolerance, tolerance],
        'log10_f_rescale' : [params_this[10], params_this[10]-tolerance, params_this[10]+tolerance, tolerance],
        'f_rescale_slope' : [params_this[11], params_this[11]-tolerance, params_this[11]+tolerance, tolerance],
    }

    mcmc_options = {'n_live_points': 1,
            'importance_nested_sampling': False,
            'sampling_efficiency': 0.8,
            'evidence_tolerance': 0.5,
            'multimodal': True,
            'max_iter': 1,
            'n_iter_before_update': 1,
            'write_output': False}
    print("starting_mcmc")
    chain = mcmc.run_mcmc(
        core, likelihood,
        datadir='./',
        model_name=model_name,
        params = param_dict ,
        use_zeus = False,
        use_multinest=True,
        create_yaml=False,
        **mcmc_options
    )
    for name in glob.glob(my_cache + str(seed_now) + r'.h5'):
        os.system('rm ' + name) 
