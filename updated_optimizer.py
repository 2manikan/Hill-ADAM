# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:18:49 2025

@author: manid
"""

from torch.optim import Optimizer
import torch
import time
import math
import copy




#goal: create a new state dict storing the parameter values. perform steps on this state dict. then load that into param groups.
class NewOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=0.01, betas=(0.9,0.999), maximize=False):
        #THIS LINE HERE creates the state_dict. This state dict (get from <optim_instance>.state_dict()) is fundamental to an optimizer
        #This links the parameters to the id's. This is only required for the optimizer here.
       
       
        #bunch of "NONE" values here. that makes sense because this '__init__" function is called much before loss.backward()
        super(NewOptimizer, self).__init__(params, defaults={'lr':lr, 'weight_decay':weight_decay, 'betas':betas})
       
        #create a list of moving averages. for EACH SET OF WEIGHTS.
        self.epsilon=1e-8
        self.num_params=len(self.state_dict()['param_groups'][0]['params'])        
        self.moving_average=[0] * self.num_params #create a moving average for each parameter
        self.multiple_moments=[0] * self.num_params
        self.weight_decay=weight_decay
       
        self.parameters=[]
        params=self.param_groups[0]["params"]
        for i in range(self.num_params):
            self.parameters.append(params[i])
            self.state[params[i]]["step"]=torch.zeros(())
            self.state[params[i]]["first_moment"]=torch.zeros_like(params[i])
            self.state[params[i]]["second_moment"]=torch.zeros_like(params[i])
           
           
        self.prev_loss=None
        self.new_loss=None
       
        self.maximize=maximize
       
        if self.maximize==True:
            self.direction=1
        else:
            self.direction=0
       
        self.direction_changed=0
        self.direction_changed_2=0
       
       
        if self.maximize==True:      
           self.least_error=-999999999
        else:
            self.least_error=999999999
           
       
        self.state_copy=None
        self.not_here=1
       
        self.f=[]
        self.count=0
       
       

       

    #loss.backward will have calculated the gradients for the required tensors
    #update the weight values by ONE STEP based on LEARNING RATE, WEIGHT DECAY, AND GRAD.
        #update the weights in PARAM GROUPS using a FOR LOOP -- NOT FOR EACH (bc this changes the actual stored values not just copies of them)
    #no need to return anything
    def step(self, loss_value, new_state):
        self.count+=1
        self.new_loss=loss_value
        
        # for debugging
        # print(self.new_loss)
        # print(new_state)
        # print("----------------")
        
        
        
       
        #changing direction
        if self.prev_loss!=None:
           
            if abs(self.new_loss-self.prev_loss)<1e-4:
                #print(self.new_loss,self.prev_loss)
                #print("----------------------------------")
               
                if self.direction==0:
                    self.direction=1
                elif self.direction==1:
                    self.direction=0
               
                self.direction_changed=1
               
               
               
                self.f.append((self.count, self.direction))
               
                self.prev_loss=None
                self.new_loss=None
               
        if self.new_loss != None:
            if self.new_loss>200:   #default is 1e4
                #copy the weights of most recent local minima here
                self.direction=0
                self.f.append((self.count, self.direction))
                if self.not_here==1:
                   self.direction_changed_2=1
                self.prev_loss=None
                self.new_loss=None
                self.not_here=0
            else:
                self.not_here=1
               
        if self.new_loss != None:   
            if self.new_loss<-1e4:
                #copy the weights of most recent local minima here
                self.direction=1
                self.f.append((self.count, self.direction))
                if self.not_here==1:
                   self.direction_changed_2=1
                self.prev_loss=None
                self.new_loss=None
                self.not_here=0
            else:
                self.not_here=1
       
       
        for i in range(self.num_params):
            parameter=self.parameters[i]
           
           
            #what happens when the direction changes
            if self.direction_changed==1 or self.direction_changed_2:
                self.state[parameter]['first_moment']=torch.zeros_like(self.state[parameter]['first_moment'])
                self.state[parameter]['second_moment']=torch.zeros_like(self.state[parameter]['second_moment'])
                self.state[parameter]['step']=torch.tensor(0)
               
           
           
            first_m=self.state[parameter]['first_moment']
            second_m=self.state[parameter]['second_moment']
            weight_decay=self.state_dict()['param_groups'][0]['weight_decay']
            betas=self.state_dict()['param_groups'][0]['betas']
            lr=self.state_dict()['param_groups'][0]['lr']
           
           
            #step update
            self.state[parameter]['step']=torch.tensor(self.state[parameter]['step'].item()+1)
            step_number=self.state[parameter]['step']
           
            #weight decay
            if(weight_decay!=0):
                parameter.grad.add(parameter, alpha=weight_decay)
           
           
               
            #update moving average
            first_m.lerp_( parameter.grad,(1-betas[0]))
            second_m.mul_(betas[1]).addcmul_(parameter.grad, parameter.grad.conj(), value=1 - betas[1])
           
           
            #bias correction and update
            first_bias=1-betas[0] ** step_number
            second_bias= 1-betas[1] **step_number
            step_size=lr/first_bias
            second_bias_sqrt=second_bias ** 0.5
            denominator=(second_m.sqrt() /second_bias_sqrt).add_(self.epsilon)
           
            if self.direction==0:
               parameter.data.addcdiv_(first_m, denominator, value=-step_size)
            elif self.direction==1:
               parameter.data.addcdiv_(first_m, denominator, value=step_size)
       
        self.prev_loss=self.new_loss
        self.direction_changed=0
        self.direction_changed_2=0
       
       
        if self.new_loss!=None:
           #update loss
           if self.least_error>self.new_loss:
                  self.least_error=self.new_loss
                  self.state_copy=new_state
           
       
        
       
        #return self.least_error
        return self.state_copy, self.least_error
