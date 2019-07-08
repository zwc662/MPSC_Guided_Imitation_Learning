#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:40:23 2018

@author: Weichao Zhou
"""
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from ..dynamics import BatchAutoDiffDynamics, tensor_constrain
from ..cost import Cost

class MountaincarDynamics(BatchAutoDiffDynamics):
   
    """Car auto-differentiated dynamics model."""

    def __init__(self, 
                 dt = 0.01,
                 constrain = True,
                 min_bounds = np.array([-0.0015]),
                 max_bounds = np.array([ 0.0015]),
                 l = 1.0,
                 **kwargs):
        """Car dynamics.

        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for actions [N, rad].
            max_bounds: Maximum bounds for actions [N, rad].
            wheel diameter
            **kwargs:

            Note:
                state: [posx, posy, v, theta]
                action: [v_dot, theta_dot]
        """

        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        def f(x, u, i):
            # Constrain action space.
            if constrain:
                u = tensor_constrain(u, min_bounds, max_bounds)

            pos = tensor_constrain(x[..., 0], -1.2, 0.6)
            v = tensor_constrain(x[..., 1], -0.07, 0.07)
            
            force = u[..., 0]

            #pos_ = T.switch(T.lt(pos, -1.2) , pos, pos + v)
            #v_ = T.switch(T.le(pos, -1.2), 0.001, v + 0.0015 * force - 0.0025 * np.cos(3 * pos)) 

            pos_ = pos + v
            v_ = 0.001, v + 0.0015 * force - 0.0025 * np.cos(3 * pos)
            

            return T.stack([
                pos_,
                v_,
                ]).T 


        super(MountaincarDynamics, self).__init__(f, state_size=2,
                                             action_size=1,
                                             **kwargs)


    @classmethod
    def augment_action(cls, action):
        return np.asarray(action)
    @classmethod
    def augment_state(cls, state):
        return np.asarray(state)
    @classmethod
    def reduce_action(cls, action):
        return np.asarray(action)
    @classmethod
    def reduce_state(cls, state):
        return np.asarray(state)


class BarrierCost(Cost):
    """Quadratic Regulator Instantaneous Cost for trajectory following and barrier function."""

    def __init__(self, A, b, q1, q2, x_barrier_u = None, x_barrier_l = None):
        """Constructs a Quadratic Cost with barrier function.

        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            q: Linear state cost matrix [state_size, 1]

            R: Quadratic control cost matrix [action_size, action_size].
            r: Linear control cost matrix [action_size, 1]

            F: Quadratic barrier cost matrix [state_size, state_size]
            f: Linear barrier cost matrix [state_size, 1]

            x_nominal: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """
        self.A = np.array(A)
        self.b = np.array(b)
        self.q1 = np.array(q1)
        self.q2 = np.array(q2)
        self.F = np.zeros([self.A.shape[-1], self.A.shape[-1]))
        self.f = np.zeros([self.A.shape[-1], 1])

        self.x_barrier_l = x_barrier_l
        self.x_barrier_u = x_barrier_u


        assert self.b.shape[0] == self.A.shape[0], "Barrier A & b mismatch"
        assert self.q1.shape[0] == self.A.shape[0], "Barrier A & q1 mismatch"
        assert self.q2.shape[0] == self.A.shape[0], "Barrier A & q2 mismatch"


        self._F_plus_F_T = self.F + self.F.T

        super(CarCost, self).__init__()

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if x[0] is None:
            print("x is None!!!!!!!")

        if self.x_barrier_l(x) is None or self.x_barrier_u(x) is None:
            print("barrier functions at {} return None!!!!!!!!".format(x[0]) )

        if self.x_barrier_l is not None:
            x_barrier_l = x - self.x_barrier_l(x)

        if self.x_barrier_u is not None:
            x_barrier_u = x - self.x_barrier_u(x)
       
        constant_x_cost = self.q1[0] * np.exp(self.q2[0] * (self.A[0].dot(x_barrier_u) - self.b[0])) - self.q1[0]\
                + self.q1[1] * np.exp(self.q2[1] * (self.A[1].dot(x_barrier_l) - self.b[1])) - self.q[1]

        F = self.q1[0] * self.q2[0]**2 * np.exp(self.q2[0] * (self.A[0].dot(x_barrier_u) - self.b[0])) * self.A[0].T * self.A[0]
        f = self.q1[0] * self.q2[0] * np.exp(self.q2[0] * (self.A[0].dot(x_barrier_u) - self.b[0])) * self.A[0].T
        
        F = F + self.q1[1] * self.q2[1]**2 * np.exp(self.q2[1] * (self.A[1].dot(x_barrier_l) - self.b[1])) * self.A[1].T * self.A[1]
        f = f + self.q1[1] * self.q2[1] * np.exp(self.q2[1] * (self.A[1].dot(x_barrier_l) - self.b[1])) * self.A[1].T

        #for i in range(2, self.A.shape[0]):
        #    F = F + self.q1[i] * self.q2[i]**2 * np.exp(self.q2[i] * (self.A[i].dot(x_dist) - self.b[i])) * self.A[i].T * self.A[i]
        #    f = f + self.q1[i] * self.q2[i] * np.exp(self.q2[i] * (self.A[i].dot(x_dist) - self.b[i])) * self.A[i].T
        self.F = F
        self.f = f

        squared_x_cost = 0.5 * x.T.dot(F).dot(x)

        linear_x_cost = x.T.dot(f)

        return squared_x_cost + u_diff.T.dot(R).dot(u_diff) + linear_x_cost + constant_x_cost

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """

        if self.x_barrier_l is not None:
            x_barrier_l = x - self.x_barrier_l(x)

        if self.x_barrier_u is not None:
            x_barrier_u = x - self.x_barrier_u(x)


        F = self.q1[0] * self.q2[0]**2 * np.exp(self.q2[0] * (self.A[0].dot(x_barrier_u) - self.b[0])) * self.A[0].T * self.A[0]
        f = self.q1[0] * self.q2[0] * np.exp(self.q2[0] * (self.A[0].dot(x_barrier_u) - self.b[0])) * self.A[0].T
        
        F = F + self.q1[1] * self.q2[1]**2 * np.exp(self.q2[1] * (self.A[1].dot(x_barrier_l) - self.b[1])) * self.A[1].T * self.A[1]
        f = f + self.q1[1] * self.q2[1] * np.exp(self.q2[1] * (self.A[1].dot(x_barrier_l) - self.b[1])) * self.A[1].T
        #for i in range(2, self.A.shape[0]):
        #    F = F + self.q1[i] * self.q2[i]**2 * np.exp(self.q2[i] * (self.A[i].dot(x_dist) - self.b[i])) * self.A[i].T * self.A[i]
        #    f = f + self.q1[i] * self.q2[i] * np.exp(self.q2[i] * (self.A[i].dot(x_dist) - self.b[i])) * self.A[i].T
        self.F = F
        self.f = f


        self._F_plus_F_T = self.F + self.F.T
        F_plus_F_T = self._F_plus_F_T


        return x.T.dot(F_plus_F_T) + self.f.T

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        return 0.0


    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """


        if self.x_barrier_l is not None:
            x_barrier_l = x - self.x_barrier_l(x)

        if self.x_barrier_u is not None:
            x_barrier_u = x - self.x_barrier_u(x)


        F = self.q1[0] * self.q2[0]**2 * np.exp(self.q2[0] * (self.A[0].dot(x_barrier_u) - self.b[0])) * self.A[0].T * self.A[0]
        f = self.q1[0] * self.q2[0] * np.exp(self.q2[0] * (self.A[0].dot(x_barrier_u) - self.b[0])) * self.A[0].T
        
        F = F + self.q1[1] * self.q2[1]**2 * np.exp(self.q2[1] * (self.A[1].dot(x_barrier_l) - self.b[1])) * self.A[1].T * self.A[1]
        f = f + self.q1[1] * self.q2[1] * np.exp(self.q2[1] * (self.A[1].dot(x_barrier_l) - self.b[1])) * self.A[1].T
        #for i in range(2, self.A.shape[0]):
        #    F = F + self.q1[i] * self.q2[i]**2 * np.exp(self.q2[i] * (self.A[i].dot(x_dist) - self.b[i])) * self.A[i].T * self.A[i]
        #    f = f + self.q1[i] * self.q2[i] * np.exp(self.q2[i] * (self.A[i].dot(x_dist) - self.b[i])) * self.A[i].T
        self.F = F
        self.f = f


        self._F_plus_F_T = self.F + self.F.T
        F_plus_F_T = self._F_plus_F_T
        return F_plus_F_T

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return 0.0

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        return 0.0


