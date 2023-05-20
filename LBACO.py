import numpy as np
import random
from matplotlib import pyplot as plt

class VM2():
    def __init__(self, mips, n_pr, bw):
        self.mips = mips  #v
        self.n_pr = n_pr
        self.bw = bw
        self.EV = n_pr * mips + bw

class Task2():
    def __init__(self, tasklength, inputfilesize):
        self.tasklength = tasklength
        self.inputfilesize = inputfilesize

class System_LBACO():
    def __init__(self, tasks, dep, vms, n_ants, n_it):
        self.tasks = tasks
        self.n_t = len(tasks)
        self.vms = vms
        self.n_vms = len(vms)
        self.dep = dep
        self.n_it = n_it
        self.ants = [Ant_LBACO(self) for i in range(n_ants)]
        
        self.edges = np.array([[vm.EV for vm in self.vms]*self.n_t])
        self.edges = np.repeat(self.edges, self.n_t*self.n_vms, axis = 0) * 10**(-4)
        self.res = np.zeros((len(tasks), len(vms)))
        for i_t, t in enumerate(tasks):
            for i_vm, vm in enumerate(vms):
                self.res[i_t][i_vm] = t.tasklength/vm.EV + t.inputfilesize/vm.bw
        self.lastAver_res = 0
        self.LB = np.ones((self.n_t, self.n_vms))

        self.alpha = 1
        self.beta = 1
        self.gamma = 1
        self.rho = 0.1
        self.D = 5
        
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
        self.lastAver_res = min_t/self.n_t
        self.LB = 1 - ((self.res - self.lastAver_res)/(self.res+self.lastAver_res))
        for ant in self.ants:
            for i_st in range(self.n_t-1):
                self.edges[ant.route_t[i_st]*self.n_vms + ant.route_vm[i_st]]\
                [ant.route_t[i_st+1]*self.n_vms + ant.route_vm[i_st+1]] \
                *= (1 - self.rho)
                if ant.time == min_t:
                    self.edges[ant.route_t[i_st]*self.n_vms + ant.route_vm[i_st]]\
                    [ant.route_t[i_st+1]*self.n_vms + ant.route_vm[i_st+1]] \
                    += self.D/min_t
                else: 
                    self.edges[ant.route_t[i_st]*self.n_vms + ant.route_vm[i_st]]\
                    [ant.route_t[i_st+1]*self.n_vms + ant.route_vm[i_st+1]] \
                    += 1/ant.time

    def aco(self):
        times = []
        for i in range(self.n_it):
            self.start()
            self.movement()
            self.global_update()
            times.append(np.mean([ant.time for ant in self.ants]))
        return np.array(times)
    
class Ant_LBACO():
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
        self.time += self.system.res[pos_t][pos_vm]
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
                    P[i_t][i_vm] = (tau**(self.system.alpha)) \
                    * (vm.EV**(self.system.beta)) \
                    * (self.system.LB[i_t][i_vm]**self.system.gamma)
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
        self.time += self.system.res[pos_t][pos_vm]
        self.local_update()
        
    def local_update(self):
        pass