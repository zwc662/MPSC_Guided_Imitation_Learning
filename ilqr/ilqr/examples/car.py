#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:40:23 2018

@author: Weichao Zhou
"""
import numpy as np
import theano.tensor as T
from ..dynamics import BatchAutoDiffDynamics, tensor_constrain
from ..cost import Cost

class CarDynamics(BatchAutoDiffDynamics):
   
    """Car auto-differentiated dynamics model."""

    def __init__(self, 
                 dt,
                 constrain = True,
                 min_bounds = np.array([-1.0, -1.0]),
                 max_bounds = np.array([1.0, 1.0]),
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

            posx = x[..., 0]
            posy = x[..., 1]
            v = x[..., 2]
            theta = x[..., 3]
            
            v_dot = u[..., 0]
            tan_delta = u[..., 1]


            # Define dynamics model as Jianyu Chen's 
            # Constrained Iterative LQR for On-Road Autonomous Driving Motion Planning

            theta_dot = v * tan_delta/l
            posx_ = posx + T.cos(theta) * (v * dt + 0.5 * v_dot * dt**2)
            posy_ = posy + T.sin(theta) * (v * dt + 0.5 * v_dot * dt**2)
            v_ = v + v_dot * dt
            theta_ = theta + theta_dot * dt

            return T.stack([
                posx_,
                posy_,
                v_,
                theta_
                ]).T 


        super(CarDynamics, self).__init__(f, state_size=4,
                                             action_size=2,
                                             **kwargs)
    @classmethod
    def augment_action(cls, action):
        """
            [v_dot, delta] - > [v_dot, tan_delta]
        """
        if state.ndim == 1:
            v_dot, delta = action
        else:
            v_dot = action[..., 0].shape(-1, 1)
            delta = action[..., 1].shape(-1, 1)
        return np.hstack([v_dot, np.tan(delta)])


class CarCost(Cost):
    """Quadratic Regulator Instantaneous Cost for trajectory following and barrier function."""

    def __init__(self, Q, q, R, r, A, b, q1, q2, x_nominal, x_barrier_u = None, x_barrier_l = None, u_path=None, Q_terminal=None):
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
        self.Q = np.array(Q)
        self.q = np.array(q)

        self.R = np.array(R)
        self.r = np.array(r)

        self.A = np.array(A)
        self.b = np.array(b)
        self.q1 = np.array(q1)
        self.q2 = np.array(q2)
        self.F = np.zeros(self.Q.shape)
        self.f = np.zeros(self.q.shape)

        self.x_nominal = np.array(x_nominal)
        self.x_barrier_l = x_barrier_l
        self.x_barrier_u = x_barrier_u

        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_nominal.shape[0]

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if u_path is None:
            self.u_path = np.zeros([path_length - 1, action_size])
        else:
            self.u_path = np.array(u_path)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.q.shape[0] == self.Q.shape[0], "q mismatch"

        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        assert self.r.shape[0] == self.R.shape[0], "r mismatch"


        assert self.A.shape[-1] == self.Q.shape[0], "Barrier A & Q mismatch"
        assert self.A.shape[0] == self.b.shape[0], "Barrier A & b mismatch"
        assert self.q1.shape[0] == self.A.shape[0], "Barrier A & q1 mismatch"
        assert self.q2.shape[0] == self.A.shape[0], "Barrier A & q2 mismatch"


        assert state_size == self.x_nominal.shape[1], "Q & x_nominal mismatch"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + 1, \
                "x_nominal must be 1 longer than u_path"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T
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
        Q = self.Q_terminal if terminal else self.Q
        r = self.r
        q = self.q
        R = self.R

        if x[0] is None:
            print("x is None!!!!!!!")

        if self.x_barrier_l(x[0]) is None:
            print("barrier_l {} returns None!!!!!!!!".format(x[0]) )

        x_diff = x - self.x_nominal[i]

        if self.x_barrier_l is not None:
            x_barrier_l = x - np.array([0.0, self.x_barrier_l(x[0])[1], 0.0, 0.0])
        else:
            x_dist = x_diff

        if self.x_barrier_u is not None:
            x_barrier_u = x - np.array([0.0, self.x_barrier_u(x[0])[1], 0.0, 0.0])
        else:
            x_dist = x_diff

       
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

        squared_x_cost = x_diff.T.dot(Q).dot(x_diff) + 0.5 * x.T.dot(F).dot(x)
        if terminal:
            return squared_x_cost

        u_diff = u - self.u_path[i]
        linear_x_cost = x_diff.T.dot(q) + u_diff.T.dot(r) + x.T.dot(f)
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
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_nominal[i]

        if self.x_barrier_l is not None:
            x_barrier_l = x - np.array([0.0, self.x_barrier_l(x[0])[1], 0.0, 0.0])
        else:
            x_dist = x_diff

        if self.x_barrier_u is not None:
            x_barrier_u = x - np.array([0.0, self.x_barrier_u(x[0])[1], 0.0, 0.0])
        else:
            x_dist = x_diff


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


        return x_diff.T.dot(Q_plus_Q_T) + x.T.dot(F_plus_F_T) + self.f.T + self.q.T

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
        if terminal:
            return np.zeros_like(self.u_path)

        u_diff = u - self.u_path[i]
        return u_diff.T.dot(self._R_plus_R_T) + self.r.T

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


        x_diff = x - self.x_nominal[i]
        if self.x_barrier_l is not None:
            x_barrier_l = x - np.array([0.0, self.x_barrier_l(x[0])[1], 0.0, 0.0])
        else:
            x_dist = x_diff

        if self.x_barrier_u is not None:
            x_barrier_u = x - np.array([0.0, self.x_barrier_u(x[0])[1], 0.0, 0.0])
        else:
            x_dist = x_diff


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
        return self._F_plus_F_T + self._Q_plus_Q_T_terminal if terminal else self._F_plus_F_T + self._Q_plus_Q_T

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
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

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
        if terminal:
            return np.zeros_like(self.R)

        return self._R_plus_R_T
