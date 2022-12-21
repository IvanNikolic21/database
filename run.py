import argparse

#so far only number of workers is to be called.
#All else will be specified in the bash script.

parser = argparse.ArgumentParser()
parser.add_argument("-- workers", type = int, default = 1)
parser.add_argument("-- threads_per_sim", type = int, default = 1)
parser.agg_argument("-- initial_seed", type = int, default = 1950)
inputs = parser.parse_args()

import os

#initialize environment variables

os.environ['OPENBLAS_NUM_THREADS'] = str(inputs.workers)
os.environ['OMP_NUM_THREADS'] = str(inputs.workers)

import numpy as np
from astropy import cosmology
import logging, sys

#21cmmc imports:

from py21cmmc import mcmc
from py21cmmc import LikelihoodNeutralFraction
from py21cmmc import CoreLightConeModule
from py21cmmc import LikelihoodLuminosityFunction, CoreLuminosityFunction
from py21cmmc import LikelihoodForest, CoreForest
from py21cmmc import LikelihoodPlanck
from py21cmmc import LikelihoodBase

#set logger from 21cmFAST

logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

#main parameter combinations:
user_params = {
    'HII_DIM':200.0,
    'BOX_LEN':400.0,  #note the change
    'USE_INTERPOLATION_TABLES': True,
    'USE_FFTW_WISDOM': True,
    'PERTURB_ON_HIGH_RES': True,
    'N_THREADS': inputs.threads_per_sim,
    'OUTPUT_ALL_VEL': False,  #for kSZ need to save all velocity components.
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
    'PHOTON_CONS': True,
    'EVOLVING_R_BUBBLE_MAX': True, #This parameter is not present in master!
    'USE_TS_FLUCT': True,
    'USE_MINI_HALOS': True,
}

global_params = {
    'Z_HEAT_MAX': 15.0, 
    'T_RE': 1e4,
    'ALPHA_UVB': 2.0,
    'PhotonConsEndCalibz':3.5
}

#21cmFAST cache settings
import py21cmfast as p21c

my_cache='./_cache/'    #update this to the desired _cache directory.
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

core = [
    CoreCoevalModule(
        redshift = redshift,
        user_params = user_params,
        cosmo_params = cosmo_params,
        flag_options = flag_options,
        global_params = global_params,
        regenerate = False,
        initial_conditions_seed  = inputs.initial_seed,
        cache_dir = my_cache,
        cache_mcmc = False,
    ) for redshift in coeval_zs
] + [ 
    CoreLightConeModule(
        redshift=4.9,
        max_redshift=15,
        user_params=user_params,
        cosmo_params=cosmo_params,
        flag_options=flag_options,
        global_params=global_params,
        regenerate=False,
        initial_conditions_seed = inputs.initial_seed,
        cache_dir=my_cache,
        cache_mcmc=False)
    ,
] + [
    CoreLuminosityFunction(
        redshift=redshift,
        sigma=0,
        name='lfz%d'%redshift,
        user_params=user_params,
        cosmo_params=cosmo_params,
        flag_options=flag_options,
        global_params=global_params,
        regenerate=False,
        initial_conditions_seed = inputs.initial_seed,
        cache_dir=my_cache,
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
        regenerate=False,
        initial_conditions_seed = inputs.initial_seed,
        cache_dir=my_cache,
        cache_mcmc=False
    ) for redshift in forest_zs
]

#likelihood modules
likelihood = [ 
    LikelihoodPlanck(),   # no LikelihoodNeutralFraction!
] + [
    LikelihoodLuminosityFunction(name='lfz%d'%redshift, simulate = False,),
    for redshift in lf_zs
] + [
    LikelihoodForest(name='bosman%s'%(str(redshift).replace('.', 'pt')))
    for redshift in forest_zs
]

#HERA likelihood to be added.

#parameters with their Gaussian initial ball options
param_dict = {
    'F_STAR10' : [-1.3, -2, -0.5, 0.5],
    'ALPHA_STAR' : [0.5, 0.0, 1.0, 0.5],
    't_STAR' : [0.5, 0.01, 1.0, 0.2],        #removed M_turn as it's already calculated.
    'F_ESC10' : [-1.0, -3.0, 0.0, 1.0],
    'ALPHA_ESC' : [-0.2, -1.0, 1.0, 0.8],
    'SIGMA_8' : [0.8118, 0.75, 0.85, 0.01], #Gaussian initiall ball here corresponds to Planck 68% CI
    'F_STAR7' : [-2.75, -4.0, -1.0, 1.0],
    'F_ESC7' : [-1.2, -3.0, -1.0, 1.0], #Based on YQ+20
    'L_X' : [40.5, 38.0, 42.0, 0.1],
    'NU_X_THRESH' : [500, 100, 1500, 300],
    'log10_f_rescale' : [0.0, -5.0, 5.0, 5],
    'f_rescale_slope' : [0.0, -2.5, 2.5, 2.5],
}

#MCMC options
mcmc_options = {
    'nsteps': 10000,   #Since zeus sampler has convergence criteria, any number will do
    'nwalkers': 18 * 13, #the choice of walkers is based on 13 parameters and the fact that 1 node 
                         #vega hosts 256 cpus. However, very likely that this number will change 
    'ndim': len(param_dict.keys()),
    'mpi_chains': 32,    #number of chains is also gonna depend on the choice. Probably a chain per
                         #node makes sense.
    'store_progress': True,
    'continue_sampling': False, # to be changed with further submissions
    'save_after_iter': 1,       #TODO, implement this!
    'folder' : r'./',           #to be changed.
    'param_names': param_dict.keys(),
}

chain = mcmc.run_mcmc(
    core, likelihood,
    datadir=mcmc_options['folder'],
    model_name=model_name,
    params = param_dict ,
    use_zeus = True,
    **mcmc_options
)


