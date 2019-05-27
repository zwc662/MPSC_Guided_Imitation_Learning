#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:40:23 2018

@author: Weichao Zhou
"""
import numpy as np
import math

class Car():
    
    def __init__(self, posx = 0., posy = 0., vel = 10., theta = 0.):

        self.posx = posx
        self.posy = posy
        self.vel = vel
        self.theta = theta

        self.gamma = gamma
        self.beta = math.tanh(self.gamma)
        self.b = 3
        self.x = np.asarray([self.posx, self.posy, self.vel, self.alpha, self.beta])
        self.x_init = self.x.copy()

        self.u1 = 0
        self.u2 = 0
        self.u = np.asarray([self.u1, self.u2])

        self.dt = 0.05

        self.G = {'posy': (-0.1, 0.1),
                  'vel': (-0.1, 0.1),
                  'alpha': (-0.1, 0.1),
                  'beta': (-0.1, 0.1)
                  }

        self.I = {'y': (-2, 2),
                  'v': (-2, 2),
                  'alpha': (-1, 1),
                  'beta': (-1, 1)
                  }

    def reset(self):
        self.x = self.x_init.copy()
    

        
    def nextStep(self, u):
        self.u1 = u[0]
        self.u2 = u[1]

        self.posx_ = self.posx + self.vel * math.cos(self.alpha) * self.dt
        self.posy_ = self.posy + self.vel * math.sin(self.alpha) * self.dt
        self.vel_ = self.vel + self.u1 * self.dt
        self.alpha_ = self.alpha + self.dt * self.vel * self.beta/self.b
        self.beta_ = self.beta + self.dt * self.u2

        self.posx = self.posx_
        self.posy = self.posy_
        self.vel = self.vel_
        self.alpha = self.alpha_
        self.beta = self.beta_
        
        self.x = np.asarray([self.posx, self.posy, self.vel, self.alpha, self.beta])
        return self.x


    def getState(self):
        self.x = np.asarray([self.posx, self.posy, self.vel, self.alpha, self.beta])
        return self.x
    

class space():
    def __init__(self, shape, n):
        self.shape = shape
        self.n = n


class env():
    def __init__(self):
        self.car = Car()
        self.observation_space = space(self.car.x.shape, self.car.x.shape[0])
        self.action_space = space(self.car.u.shape, self.car.u.shape[0])
        self.max_step = 200
        self.i_step = 0
    
    def seed(self, seed = 1):
        self.seed_shuffle = seed

    def reset(self):
        self.car.reset()
        self.i_step = 0
        return self.car.x(


    def step(self, action):
        action = np.asarray(action)
        assert action.shape == self.action_space.shape
        x = self.car.nextStep(action)
        self.i_step += 1
        if x[1] >= self.car.G['posy'][0] and x[1] <= self.car.G['posy'][1] \
                 and x[2] >= self.car.G['vel'][0] and x[2] <= self.car.G['vel'][1]\
                 and x[3] >= self.car.G['alpha'][0] and x[3] <= self.car.G['alpha'][1]\
                 and x[4] >= self.car.G['beta'][0] and x[4] <= self.car.G['beta'][1]:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
        if self.i_step >= self.max_step:
            done = True
        return (x, reward, done, {})

    def render():
        pass

def make():
    return env)
