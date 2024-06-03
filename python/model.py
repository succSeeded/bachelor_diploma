import numpy as np
import numpy.linalg as LA 
import matplotlib.pyplot as plt
import scipy.integrate as scip
import random
from utils import *
from math_utils import *

class eq_model:
    
    def __init__(self, params=None, dev=None):
        self.__params_set = False
        if dev:
            self.dev = True
        else:
            self.dev = False
        if params and isinstance(params, dict):
            self.__set_parameters(params)
            
    def __set_parameters(self, params=None):
        if params and isinstance(params, dict):
            print('Setting parameters...')
            for param in [*params.items()]:
                self.__setattr__(param[0], param[1])
            self.__set_variables_at_eqpoints()
            self.__set_dequations()
            self.__set_jacobian()
            self.__set_attr_bounds()
            self.__params_set = True
            print('Parameters set!')
        else:
            raise ValueError('System parameters should be a non-empty dict')
            
    def __set_variables_at_eqpoints(self):
        self.__setattr__('x_4', lambda x_1: (self.b_1*x_1 + self.s_1)/self.mu_2)
        self.__setattr__('x_3', lambda x_1: self.a_2*x_1*(x_1+self.k_3)/((self.k_5+self.x_4(x_1))*(self.alpha_4*x_1+self.mu_1*(x_1+self.k_3))))
        self.__setattr__('x_5', lambda x_1: self.b_2*self.x_3(x_1)/self.mu_3)
        self.__setattr__('x_2', lambda x_1: (self.r_1*(1.0-x_1/self.c_1)*(self.x_4(x_1)+self.e_1)*(x_1+self.k_1)-self.alpha_2*self.x_3(x_1))/self.alpha_1)
    
    def __target_function(self, x_1):
        return self.r_2*self.x_2(x_1)*(self.c_2-self.x_2(x_1))*(x_1+self.k_1)*(self.k_4+self.x_5(x_1))*(self.x_4(x_1)+self.e_2)+self.a_1*self.c_2*(x_1+self.k_1)-self.alpha_3*self.c_2*x_1*self.x_2(x_1)*(self.k_4+self.x_5(x_1))*(self.x_4(x_1)+self.e_2)
    
    def __set_dequations(self):
        def dxdt(X):
            try:
                if X.shape[0] != 5:
                    raise ValueError('Incorrect argument dimensions.')
            except:
                if isinstance(X,list) and isinstance(X[0],np.array):
                    if X[0].shape != 5:
                        raise ValueError('Incorrect argument dimensions.')
                else:
                    raise TypeError('Input is neither list of np.array nor an np.array.')
                 
            return np.array(
                [self.r_1*X[0]*(1.0-X[0]/self.c_1)-(self.alpha_1*X[1]+self.alpha_2*X[2])*X[0]/(X[3]+self.e_1)/(X[0]+self.k_1), 
                self.r_2*X[1]*(1-X[1]/self.c_2)+self.a_1*X[4]/((self.k_4+X[4])*(self.e_2+X[3]))-self.alpha_3*X[0]*X[1]/(X[0]+self.k_2), 
                self.a_2*X[0]/(X[3]+self.k_5)-self.mu_1*X[2]-self.alpha_4*X[0]*X[2]/(X[0]+self.k_3), 
                self.s_1+self.b_1*X[0]-self.mu_2*X[3], 
                self.b_2*X[2]-self.mu_3*X[4]], dtype='float64')      
        
        self.__setattr__('dxdt', dxdt)

    def __set_jacobian(self):
        self.__setattr__('jacobian', lambda X: np.transpose(np.array([partial(self.dxdt, X, n_arg=j) for j in range(1,6)], dtype='float64')))
        
    def __set_attr_bounds(self):
        self.__setattr__('Ox1', self.c_1)
        self.__setattr__('Ox4', self.s_1/self.mu_2 + self.b_1*self.Ox1)
        self.__setattr__('Ux4', self.s_1/self.mu_2)
        self.__setattr__('Ox3', self.a_2*self.Ox1*(self.Ox1 + self.k_2)/(self.k_5+self.Ux4)/self.mu_1/self.k_2)
        self.__setattr__('Ox5', self.b_2*self.Ox3/self.mu_3)
        self.__setattr__('Ox2', self.c_2*.5 + np.sqrt(self.c_2*self.c_2*.5 + self.a_1*self.Ox5/self.k_4/(self.Ux4+self.e_2)))
        
        print(f'\nAttractor-containing set boundaries (Oxi -- upper bound, Uxi -- lower bound):\n\nOx1 = {self.Ox1}')
        print(f'Ox2 = {self.Ox2}')
        print(f'Ox3 = {self.Ox3}')
        print(f'Ux4 = {self.Ux4}')
        print(f'Ox4 = {self.Ox4}')
        print(f'Ox5 = {self.Ox5}\n')

    def eigs(self, point):
        return np.sort(LA.eigvals(self.jacobian(point)))

    def find_eqpoints(self):
        self.eqpoints = np.vstack((np.array([0.0,0.0,0.0,self.s_1/self.mu_2,0.0]),\
            np.array([0.0,self.c_2,0.0,self.s_1/self.mu_2,0.0])))
        
        print(f'Searching for zeros...')
        zeros = []
        
        if self.dev == True:
            interval = (self.c_1-100000, self.c_1)
        else:
            interval = (1e-6, self.c_1)
            
        start = time.time()    
        zeros, zero_times = zero_localizer(self.__target_function, interval, d=1e-1)
        end = time.time()
        for x_1 in zeros:
            self.eqpoints = np.vstack(\
                (self.eqpoints,\
                np.array([x_1, self.x_2(x_1), self.x_3(x_1), self.x_4(x_1), self.x_5(x_1)]))\
                )
        print(f'Total time elapsed: {round(end-start, 6)}s')
        print(f'Time elapsed while find_zero: {[f"{round(time,6)}s" for time in zero_times]}')
                    
        if len(zeros) == 1:
            print(f'One zero found. x_1 = {zeros[0]}')
        elif len(zeros) > 0:
            print(f'{len(zeros)} zeros found.')
            [print(f'X = {zero}') for zero in zeros]
        else:
            print('No zeros found on a given interval.')
            
        print('\nEquilibrium points:\n')
        for i in range(self.eqpoints.shape[0]):
            print(f'P = {self.eqpoints[i,:]}')
        print('\n')
        
        return None
        
    def eqpoint_condtitions(self, point):
        cond_1 = self.r_1*(self.c_1-point[0])*(point[3]+self.e_1)*(point[0]+self.k_1) - self.alpha_2*self.c_1*point[2] > 0
        cond_2 = (0 < point[0]) and (point[0] < self.c_1)
        if cond_1 and cond_2:
            print(f'Point X = {point} satisifies condtitions for a equilibrium point.')
        else:
            print(f'Point X = {point} does not satisfy condtitions for a equilibrium point.')
        return cond_1 and cond_2
    
    def cond_roots(self):
        d = np.array([-self.c_1, (self.s_1+self.mu_2*self.e_1)/self.b_1, self.k_1, (self.s_1+self.mu_2*self.k_5)/self.b_1, self.mu_1*self.k_3/(self.alpha_4+self.mu_1)])
        f = self.alpha_2*self.c_1*self.a_2*self.mu_2*self.mu_2/self.r_1/self.b_1/self.b_1/(self.alpha_4+self.mu_1)
        P = 1
        for i in range(5):
            P = np.polymul(P,(1, d[i]))
        P = np.polyadd(P, [f, f*self.k_3, 0])
        croots = np.roots(P)
        print(f'Коэффициенты многочлена из (4): {P}')
        print(f'{np.polyval(P, np.roots(P))}')
        for j in range(croots.shape[0]):
            point = [croots[j], self.x_2(croots[j]), self.x_3(croots[j]), self.x_4(croots[j]), self.x_5(croots[j])]
            val = self.r_1*(self.c_1-point[0])*(point[3]+self.e_1)*(point[0]+self.k_1) - self.alpha_2*self.c_1*point[2]
            print(f'(4)-2 в точке x_1 = {croots[j]}: {val}')
        return croots
    
    def integrate_at_point(self, point, T = 2000.0):
        
        plt.rcParams['text.usetex'] = False
        
        ax = [plt.figure().add_subplot(projection='3d') for i in range(3)]
        sol = scip.solve_ivp(lambda t, X: self.dxdt(X), [0.0,T], point, rtol=1e-7, atol=1e-6)
        X = sol.y
        for i in range(len(ax)):
            ax[i].plot(X[(i + 2*(i//5))%5,:], X[(i+1)%5,:], X[(i+2-2*(i//5))%5,:], color='black')    
            ax[i].scatter(point[(i + 2*(i//5))%5], point[(i+1)%5], point[(i+2-2*(i//5))%5], color='black')    
            ax[i].scatter(self.eqpoints[1:,(i + 2*(i//5))%5], self.eqpoints[1:,(i+1)%5], self.eqpoints[1:,(i+2-2*(i//5))%5], color='r')
            ax[i].text(self.eqpoints[1,(i + 2*(i//5))%5], self.eqpoints[1,(i+1)%5], self.eqpoints[1,(i+2-2*(i//5))%5], 'P_2')
            ax[i].text(self.eqpoints[2,(i + 2*(i//5))%5], self.eqpoints[2,(i+1)%5], self.eqpoints[2,(i+2-2*(i//5))%5], 'P_3')
            ax[i].set_xlabel(f'x_{(i + 2*(i//5))%5+1}')
            ax[i].set_ylabel(f'x_{(i+1)%5+1}')
            ax[i].set_zlabel(f'x_{(i+2-2*(i//5))%5+1}')
        return None 
    
    def quiver(self, plot_area, N = np.array([5, 5, 5, 5, 5])):
        
        ax = plt.figure().add_subplot(projection='3d')

        # Make the grid
        x1, x2, x3, x4, x5 = np.meshgrid(np.linspace(plot_area[0,0], plot_area[0,1], num = N[0]),
            np.linspace(plot_area[1,0], plot_area[1,1], num = N[1]),
            np.linspace(plot_area[2,0], plot_area[2,1], num = N[2]),
            np.linspace(plot_area[3,0], plot_area[3,1], num = N[3]),
            np.linspace(plot_area[4,0], plot_area[4,1], num = N[4]))

        # Make the direction data for the arrows
        u1, u2, u3, u4, u5 = self.dxdt(np.array([x1, x2, x3, x4, x5])) 

        ax.quiver(x1, x2, x3, u1, u2, u3, length = 1)
        ax.scatter(self.eqpoints[2,0], self.eqpoints[2,1], self.eqpoints[2,2], color='r')
        
                 
    def integrate_on_set(self, bounds, T = 2000.0, N = np.array([5,5,5,5,5])):
        
        # working under assmption that the first element of self.eqpoints is an equilibrium point that lies inside of the set D 
        
        if bounds.shape != (5,2):
            raise ValueError('Incorrect bounds!')
        
        x1 = np.linspace(bounds[0,0], bounds[0,1], num = N[0])
        x3 = np.linspace(bounds[2,0], bounds[2,1], num = N[2])
        x5 = np.linspace(bounds[4,0], bounds[4,1], num = N[4])
        
        ax = plt.figure().add_subplot(projection='3d')
        points = np.array([np.array([x1[i], 0.0, x3[j], self.s_1/self.mu_2, x5[k]]) for i in range(len(x1)) for j in range(len(x3))  for k in range(len(x3))])
        for i in range(len(points)):
            sol = scip.solve_ivp(lambda t, X: self.dxdt(X), [0.0,T], points[i], rtol=1e-7, atol=1e-6)
            X = sol.y
            ax.plot(X[0,:], X[2,:], X[4,:], color='black')    
            ax.scatter(points[i,0], points[i, 2], points[i, 4], color='black')    
        ax.scatter(self.eqpoints[1:,0], self.eqpoints[1:,2], self.eqpoints[1:,4], color='r')
        ax.text(self.eqpoints[1,0], self.eqpoints[1,2], self.eqpoints[1,4], 'P_2')
        ax.text(self.eqpoints[2,0], self.eqpoints[2,2], self.eqpoints[2,4], 'P_3')
        ax.set_xlabel(f'x_{1}')
        ax.set_ylabel(f'x_{3}')
        ax.set_zlabel(f'x_{5}')
        
        return None
                
    def integrate_near_point(self, point, T = 2000.0, N = np.array([5,5,5,3,5])):
        
        # working under assmption that the first element of self.eqpoints is an equilibrium point that lies inside of the set D 
        
        x1 = np.linspace(0.0, self.Ox1, N[0]) 
        x2 = np.linspace(0.0, self.Ox2, N[1])
        x3 = np.linspace(0.0, self.Ox3, N[2])
        x4 = np.linspace(self.Ux4, self.Ox4, N[3])
        x5 = np.linspace(0.0, self.Ox5, N[4])
        
        x1, x2, x3, x4, x5 = np.meshgrid(x1, x2, x3, x4, x5)
        
        NN = np.prod(N)
        traj_beg = np.zeros((NN,5))
        traj_end = np.zeros((NN,5))
        for i in range(NN):
            progress_bar(i, NN)
            coord1 = i%N[0] 
            coord2 = i//N[0]%N[1] 
            coord3 = i//(N[0]*N[1])%N[2]
            coord4 = i//(N[0]*N[1]*N[2])%N[3] 
            coord5 = i//(N[0]*N[1]*N[2]*N[3])%N[4]
            sol = scip.solve_ivp(lambda t, X: self.dxdt(X), [0.0,T], np.array([x1[coord1, coord2, coord3, coord4, coord5], x2[coord1, coord2, coord3, coord4, coord5], x3[coord1, coord2, coord3, coord4, coord5], x4[coord1, coord2, coord3, coord4, coord5], x5[coord1, coord2, coord3, coord4, coord5]]), rtol=1e-7, atol=1e-6)
            traj_beg[i,:] = sol.y[:,0]
            traj_end[i,:] = sol.y[:,-1]
            # if i == 100:
            #     print(f'\nДля начальных точек: \n{traj_beg[89:99,:]}')
            #     print(f'\nРасстояния от конечных точек траекторий до внутреннего положения равновесия: \n{np.linalg.norm(traj_end[89:99] - self.eqpoints[2],2,axis=1)}')
            #     print(f'\nРасстояния от конечных точек траекторий до устойчивого граничного положения равновесия: \n{np.linalg.norm(traj_end[89:99] - self.eqpoints[1],2,axis=1)}')
            #     print(f'\nРасстояния от конечных точек траекторий до неустойчивого граничного положения равновесия: \n{np.linalg.norm(traj_end[89:99] - self.eqpoints[0],2,axis=1)}')
            #     return None
        
        Dmax = np.argmax(np.linalg.norm(traj_end - self.eqpoints[2],2))    
        progress_bar(NN, NN)
        return Ds[Dmax]
    
    def plot_transitions(self, T = 2000.0):
        plt.rcParams['text.usetex'] = True
        ax1 = [plt.figure().add_subplot(), plt.figure().add_subplot(), plt.figure().add_subplot(), plt.figure().add_subplot(), plt.figure().add_subplot()]
        ax2 = [plt.figure().add_subplot(), plt.figure().add_subplot()]
        random.seed()
        init_point = np.array([random.random()*self.Ox1, random.random()*self.Ox2,\
            random.random()*self.Ox3, self.Ux4*.5, random.random()*self.Ox5])
        sol = scip.solve_ivp(lambda t, X: self.dxdt(X), [0.0,T], init_point, rtol=1e-7, atol=1e-6)
        ax1[0].grid()
        ax1[1].grid()
        ax1[2].grid()
        ax1[3].grid()
        ax1[4].grid()
        ax1[0].plot(sol.t, sol.y[0,:])
        ax1[1].plot(sol.t[:500], sol.y[1,:500])
        ax1[2].plot(sol.t[:500], sol.y[2,:500])
        ax1[3].plot(sol.t[:50], sol.y[3,:50])
        ax1[4].plot(sol.t[:500], sol.y[4,:500])
        ax1[0].set_title(r'$x_1$')
        ax1[1].set_title(r'$x_2$')
        ax1[2].set_title(r'$x_3$')
        ax1[3].set_title(r'$x_4$')
        ax1[4].set_title(r'$x_5$')
        ax1[0].set_xlabel(r't, c')
        ax1[1].set_xlabel(r't, c')
        ax1[2].set_xlabel(r't, c')
        ax1[3].set_xlabel(r't, c')
        ax1[4].set_xlabel(r't, c')
        ax1[0].set_ylabel(r'$x_1$')
        ax1[1].set_ylabel(r'$x_2$')
        ax1[2].set_ylabel(r'$x_3$')
        ax1[3].set_ylabel(r'$x_4$')
        ax1[4].set_ylabel(r'$x_5$')
        random.seed()
        init_point = np.array([0.0, random.random()*self.Ox2,\
            0.0, self.Ux4*.5, 0.0])
        sol = scip.solve_ivp(lambda t, X: self.dxdt(X), [0.0,T], init_point, rtol=1e-7, atol=1e-6)
        # ax2[0].plot(sol.t, sol.y[0,:]+sol.y[2,:]+sol.y[4,:])
        ax2[0].grid()
        ax2[1].grid()
        ax2[0].set_title(r'$x_2$')
        ax2[1].set_title(r'$x_4$')
        ax2[0].plot(sol.t[:100], sol.y[1,:100])
        ax2[1].plot(sol.t[:30], sol.y[3,:30])
        # ax2[0].set_xlabel(r't, c')
        ax2[0].set_xlabel(r't, c')
        ax2[1].set_xlabel(r't, c')
        # ax2[0].set_ylabel(r'$x_1+x_3+x_5$')
        ax2[0].set_ylabel(r'$x_2$')
        ax2[1].set_ylabel(r'$x_4$')
        plt.show()
        return None