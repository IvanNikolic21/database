import scipy.stats as stats
import numpy as np

param_dict = {
    'F_STAR10' : [-1.3, -2, -0.5, 0.2],
    'ALPHA_STAR' : [0.5, 0.0, 1.0, 0.2],
    't_STAR' : [0.5, 0.01, 1.0, 0.3],        #removed M_turn as it's already calculated.
    'M_TURN' : [8.7, 8.0, 10.0, 0.4],
    'F_ESC10' : [-1.3, -3.0, 0.0, 0.5],
    'ALPHA_ESC' : [0.0, -1.0, 1.0, 0.6],
    'SIGMA_8' : [0.8118, 0.75, 0.85, 0.01], #Gaussian initiall ball here corresponds to Planck 68% CI
    'F_STAR7' : [-2.75, -4.0, -1.0, 1.0],
    'F_ESC7' : [-1.2, -3.0, -1.0, 1.0], #Based on YQ+20
    'L_X' : [40.5, 38.0, 42.0, 1.0],
    'NU_X_THRESH' : [500, 100, 1500, 300],
#    'log10_f_rescale' : [0.0, -5.0, 5.0, 1.5],
#    'f_rescale_slope' : [0.0, -2.5, 2.5, 2.0],
}

param_dict_names_listed = [
    'F_STAR10', 
    'ALPHA_STAR',
    't_STAR',
    'M_TURN',
    'F_ESC10', 
    'ALPHA_ESC', 
    'SIGMA_8',
    'F_STAR7',
    'F_ESC7',
    'L_X',
    'NU_X_THRESH',
#    'log10_f_rescale',
#    'f_rescale_slope']
] 
n_combs = 10000000
params = np.zeros((n_combs, len(param_dict.keys())))
#for i in range(n_combs):
#    comb_now = np.asarray(
#            [
#                stats.truncnorm.rvs(
#                    param_dict[param_name][1] - param_dict[param_name][0] / param_dict[param_name][-1],
#                    param_dict[param_name][2] - param_dict[param_name][0] / param_dict[param_name][-1],
#                    loc = param_dict[param_name][0],
#                    scale = param_dict[param_name][-1],
#                    size = 1, 
#                ).item()
#                for param_name in param_dict_names_listed
#            ]
#        )
#    params[i] = comb_now

comb_now = np.asarray(
        [
            stats.truncnorm.rvs(
                (param_dict[param_name][1] - param_dict[param_name][0]) / param_dict[param_name][-1],
                (param_dict[param_name][2] - param_dict[param_name][0]) / param_dict[param_name][-1],
                loc = param_dict[param_name][0],
                scale = param_dict[param_name][-1],
                size = n_combs,
            )
            for param_name in param_dict_names_listed
        ]
    ).T



folder_to_add = r'/home/inikoli/'
#    with open(folder_to_add + 'params.txt', 'a') as f:
#        f.write(str(comb_now)
np.savetxt(folder_to_add + 'params.txt', comb_now)
