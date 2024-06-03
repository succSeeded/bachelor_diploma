from model import *
from numpy import set_printoptions
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from math_utils import *
import random
import phaseportrait as phsprt
import numpy as np

set_printoptions(suppress=True, precision=6)
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dev", action='store_true', help="Enables development mode. Calculations are done on a less dense net.")
args = vars(parser.parse_args())

model_params = {'r_1': .01, 'c_1': 8.8265e+5, 'e_1': 1e+4, 'alpha_1': 1.5,
                'alpha_2': .12, 'k_1': 2.7e+4, 'r_2': .3307, 'c_2': 1e+6, 'a_1': .1163, 
                'k_4': 1.05e+4, 'e_2': 1e+4, 'alpha_3': .0194, 'k_2': 2.7e+4, 'a_2': .25,
                'k_5': 2e+3, 'mu_1': .007, 'alpha_4': .1694, 'k_3': 3.3445e+5, 's_1': 6.3305e+4,
                'b_1': 5.75e-6, 'mu_2': 6.93, 'b_2': 1.02e-4, 'mu_3': .102}

model = eq_model(params=model_params, dev=args['dev'])
# model.plot_transitions()
model.find_eqpoints()
# model.integrate_at_point(np.array([10000.0,10000.0,10000.0,model.s_1/model.mu_2,50.0]))
random.seed()
f = open("init_points.txt", "a")
point = np.array([random.random()*model.Ox1, random.random()*model.Ox2,\
            random.random()*model.Ox3, model.Ux4 + random.random()*(model.Ox4-model.Ux4), random.random()*model.Ox5])
f.write(f'Point {point}\n')
f.close()
model.integrate_at_point(point)
# set_printoptions(suppress=True, precision=6)
# for i in range(model.eqpoints.shape[0]):
#     model.eqpoint_condtitions(model.eqpoints[i])
#     print(f'Loss: {model.dxdt(model.eqpoints[i])}.')
#     print(f'Jacobian matrix at ({model.eqpoints[i]}):\n{model.jacobian(model.eqpoints[i])}')
#     print(f'Eigenvalues of J({model.eqpoints[i]}) are:\n{model.eigs(model.eqpoints[i])}\n')
    # if i > 1:
    #     model.integrate_on_set(bounds = np.array([[0.0, model.Ox1],[0.0, model.Ox2],[0.0, model.Ox3],[model.Ux4, model.Ox4],[0.0, model.Ox5]]))
# P1 = np.array([0,0,0,model_params["s_1"]/model_params["mu_2"],0],dtype="float64")
# croots = model.cond_roots()
# print(f'Roots of condition (4)-2: {croots}')
bounds = np.array([[0.0, model.Ox1], 
                        [0.0, model.Ox2], 
                        [0.0, model.Ox3], 
                        [model.Ux4, model.Ox4], 
                        [0.0, model.Ox5]])
model.integrate_on_set(bounds)
plt.show()
