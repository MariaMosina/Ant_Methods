import numpy as np
import random
from matplotlib import pyplot as plt

class VM():
    def __init__(self, v):
        self.v = v

class Task():
    def __init__(self, compl):
        self.compl = compl

class System_ACS():
    def __init__(self, tasks, dep, vms, n_ants, n_it):
        self.tasks = tasks
        self.n_t = len(tasks)
        self.vms = vms
        self.n_vms = len(vms)
        self.dep = dep
        self.n_it = n_it
        self.ants = [Ant_ACS(self) for i in range(n_ants)]
        
        self.edges = np.random.rand(len(self.tasks)*len(self.vms), 
                              len(self.tasks)*len(self.vms)) * 10**(-8)
        self.exec_t = np.zeros((len(tasks), len(vms)))
        for i_t, t in enumerate(tasks):
            for i_vm, vm in enumerate(vms):
                self.exec_t[i_t][i_vm] = t.compl/vm.v
        
        self.alpha = 1
        self.beta = 0.5
        self.rho = 0.1
        self.q0 = 0.7
        self.ksi = 0.1
        self.tau0 = 10**(-3)
        
    def start(self):
        vm_ids = np.random.randint(0, len(self.vms), len(self.ants))
        if len(self.dep[0]):
            av_tasks_ids = np.array([t_id for t_id in range(len(self.tasks)) if t_id not in self.dep.T[1]])
        else: 
            av_tasks_ids =  np.array(list(range(len(self.tasks))))
        tasks_ids = np.array([random.choice(av_tasks_ids) for i in range(len(self.ants))])
        np.random.shuffle(tasks_ids)
        for i in range(len(self.ants)):
            self.ants[i].set_pos(tasks_ids[i], vm_ids[i])
            
    def movement(self):
        for t in range(self.n_t-1):
            for i, ant in enumerate(self.ants):
                ant.step()
        
    def global_update(self):
        times = [ant.time for ant in self.ants]
        min_t = min(times)
        for ant in self.ants:
            if ant.time == min_t:
                for i_st in range(self.n_t-1):
                    self.edges[ant.route_t[i_st]*self.n_vms + ant.route_vm[i_st]]\
                    [ant.route_t[i_st+1]*self.n_vms + ant.route_vm[i_st+1]] \
                    *= (1 - self.rho)
                    self.edges[ant.route_t[i_st]*self.n_vms + ant.route_vm[i_st]]\
                    [ant.route_t[i_st+1]*self.n_vms + ant.route_vm[i_st+1]] \
                    += 1/min_t

    def aco(self):
        times = []
        for i in range(self.n_it):
            self.start()
            self.movement()
            self.global_update()
            times.append(np.mean([ant.time for ant in self.ants]))
        return np.array(times)

class Ant_ACS():
    def __init__(self, system):
        self.system = system
        
    def set_pos(self, pos_t, pos_vm):
        self.route_t = np.array([], dtype = int)
        self.route_vm = np.array([], dtype = int)
        if len(self.system.dep[0]):
            self.taboo = self.system.dep.T[1]
        else:
            self.taboo = np.array([])
        self.time = 0
        self.taboo = np.append(self.taboo, pos_t)
        self.route_t = np.append(self.route_t,  pos_t)
        self.route_vm = np.append(self.route_vm,  pos_vm)
        self.time += self.system.exec_t[pos_t][pos_vm]
        if len(self.system.dep[0]):
            if pos_t in self.system.dep.T[0]:
                ids = np.where(self.system.dep.T[0] == pos_t)
                for id in ids:
                    self.taboo = np.delete(self.taboo, np.where(self.taboo == self.system.dep.T[1][id]))

    def step(self):
        n_t, n_vm = self.system.n_t, self.system.n_vms
        P = np.zeros_like(self.system.edges)
        pos_t = self.route_t[-1]
        pos_vm = self.route_vm[-1]
        for i_t, t in enumerate(self.system.tasks):
            if i_t not in self.taboo:
                for i_vm, vm in enumerate(self.system.vms):
                    tau = self.system.edges[pos_t*n_vm+pos_vm][i_t*n_vm+i_vm]
                    eta = 1/self.system.exec_t[i_t][i_vm]
                    P[i_t][i_vm] = (tau**(self.system.alpha)) * (eta**(self.system.beta))
        q = random.random()
        if q <= self.system.q0:
            pos_t, pos_vm = np.unravel_index(np.argmax(P), P.shape)
        else:
            l_ii, l_jj, l_P = [], [], []
            for ii in range(len(P)):
                for jj in range(len(P[0])):
                    if P[ii][jj] > 0:
                        l_ii.append(ii)
                        l_jj.append(jj)
                        l_P.append(P[ii][jj])
            idx = random.choices(np.arange(len(l_P)),  weights=l_P)[0]
            pos_t, pos_vm = l_ii[idx], l_jj[idx]
        
        self.taboo = np.append(self.taboo, pos_t)
        if len(self.system.dep[0]):
            if pos_t in self.system.dep.T[0]:
                ids = np.where(self.system.dep.T[0] == pos_t)
                for id in ids:
                    self.taboo = np.delete(self.taboo, np.where(self.taboo == self.system.dep.T[1][id]))
        self.route_t = np.append(self.route_t, pos_t)
        self.route_vm = np.append(self.route_vm, pos_vm)
        self.time += self.system.exec_t[pos_t][pos_vm]
        self.local_update()
        
    def local_update(self):
        self.system.edges[self.route_t[-2]*self.system.n_vms + self.route_vm[-2]]\
                  [self.route_t[-1]*self.system.n_vms + self.route_vm[-1]]\
                  *= (1 - self.system.ksi)
        self.system.edges[self.route_t[-2]*self.system.n_vms + self.route_vm[-2]]\
                  [self.route_t[-1]*self.system.n_vms + self.route_vm[-1]]\
                  += self.system.ksi * self.system.tau0