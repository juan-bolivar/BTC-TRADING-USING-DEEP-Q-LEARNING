"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import pdb
import time


class QLearner(object):
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.s_prev = {}
        
####################################CUSTOM DEFINITIONS############################################################################################################
        self.t1 = time.time()
        self.alphar = 0.65
        self.num_states=num_states
        self.num_actions =num_actions 
        self.alpha = alpha 
        self.gamma = gamma 
        self.rar   = rar 
        self.radr  = radr
        self.Q_s_a = np.random.rand(num_states,num_actions)
        self.dyna = dyna
        if( dyna >0):
            self.Tc_s_a= np.array([[[0.000001]*num_states]*num_actions]*num_states)
            self.R_s_a = np.random.rand(num_states,num_actions)*1-1#-num_states
            self.T_s_a = np.array([[[0.000001/sum(self.Tc_s_a[0,0,:])]*num_states]*num_actions]*num_states)
        
        
    def author(self):
        return 'Juan'
    
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        #action = rand.randint(0, self.num_actions-1)
        action = self.Q_s_a[s,:].argmax()
        if self.verbose: print "s =", s,"a =",action
        self.a = action
        return action

    def query(self,s_prime,r):
        
        """
        
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        <s,a,s',r>
        
        """
        ## Simulate the random probability deca                                 ###
        ## rar = rar*radr                                                       ###
        
        self.rar  = self.rar*self.radr
        self.Q_s_a[self.s,self.a] = (1-self.alpha)*self.Q_s_a[self.s,self.a] + self.alpha*(r + self.gamma*self.Q_s_a[s_prime,:].max())
        
        if(np.random.binomial(1,self.rar)):
            
            
            action = rand.randint(0, self.num_actions-1)            
            #self.Q_s_a[self.s,self.a] = (1-self.alpha)*self.Q_s_a[self.s,self.a] + self.alpha*(r + self.gamma*self.Q_s_a[s_prime,self.Q_s_a[s_prime,:].argmax()])
            
        else:
            
            action = self.Q_s_a[s_prime,:].argmax()
            
            
        ### Q-LEARNING#########################################################################################    
        if(self.dyna>0):
            
            self.Tc_s_a[self.s,self.a,s_prime] += 1
            self.R_s_a[self.s,self.a] = (1-self.alphar)*self.R_s_a[self.s,self.a] + self.alphar*r
            self.T_s_a[self.s,self.a,:] = self.Tc_s_a[self.s,self.a,:]/sum(self.Tc_s_a[self.s,self.a,:])
            self.s_prev.setdefault(self.s,[]).append(self.a)
            
            for i in range(int(len(self.s_prev.keys())*0.3)+1):
                
                #if(np.random.binomial(1,0.1)):
                #    s = np.random.choice(self.s_prev.keys()) #rand.randint(0,self.num_states-1)
                #    a = np.random.choice(self.s_prev[s])
                    
                #else:
                a = rand.randint(0,self.num_actions-1)
                s = rand.randint(0,self.num_states-1)
                
                r_prime = self.R_s_a[s,a]
                s_prime_dyna = np.random.choice(range(self.num_states),1,p = self.T_s_a[s,a,:])[0]
                self.Q_s_a[s,a] = (1-self.alpha)*self.Q_s_a[s,a] + self.alpha*(r_prime + self.gamma*self.Q_s_a[s_prime_dyna,:].max())
                
                
            #self.Q_s_a[self.s,self.a] = (1-self.alpha)*self.Q_s_a[self.s,self.a] + self.alpha*(r + self.gamma*self.Q_s_a[s_prime,:].max())
            self.Q_s_a[self.s,self.a] = (1-self.alpha)*self.Q_s_a[self.s,self.a] + self.alpha*(r + self.gamma*self.Q_s_a[s_prime,:].max())
            action = self.Q_s_a[s_prime,:].argmax()
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        self.a,self.s = (action,s_prime)
        self.s_prev.setdefault(s_prime,[]).append(action)
        #print(time.time()-self.t1)
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
