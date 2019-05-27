import sparsegrad.forward as forward
import numpy as np
import osqp
import scipy.sparse as sparse


class MPC():
    def __init__(self, x0, v0, theta0, thetadot0):
        # Horizon is 50 time steps
        self.N = 20
        # 4 state variables and 1 action variable
        self.NVars  = 5
        # 8s in real time for 50 time steps
        T = 8.0
        # Time span for each step
        self.dt = T/self.N
        dt = self.dt
        self.dtinv = 1./dt
	# Px = sparse.eye(N)
	# sparse.csc_matrix((N, N)) 

	# The different weigthing matrices for x, v, angle, angluar v and external force

        # NXN diagonal identity matrix with 0.05 on the diagonal
        reg = sparse.eye(self.N)*0.05

        # NXN diagonal identity matrix with 1 on the diagonal
        z = sparse.bsr_matrix((self.N, self.N))
        # sparse.diags(np.arange(N)/N)
        
        # NXN diagonal matrix with elements being evenly spaced samples from 1 to 7
        pp = sparse.diags(np.linspace(1,20,self.N)) # sparse.eye(self.N)

        # 5NX5N diagonal matrix with blocks arranged on the diagonal  
        P = sparse.block_diag([reg, 10*reg ,pp, reg, 10*reg]) #1*reg,1*reg])

        #P[N,N]=10
        self.P = P
        THETA = 2


        # The goal states initially are in a 5XN all zero matrix
        q = np.zeros((self.NVars, self.N))
        # The 3rd row, the pole angle, is initially pi???????
        q[THETA,:] = np.pi
        
        # The 1st row, the cart position, is initially 0.5
        q[0,:] = 0.0 #0.5
        #q[N,0] = -2 * 0.5 * 10

        # q in one dimension with 5*N elements where the same type of varialbes are put together
        q = q.flatten()

        # P is 5NX5N, q is 5NX1
        # min 0.5P(x - q)^2=> min 0.5 x^TPx - Pqx where q is originally the true weight
        # -Pq replaces q in the osqp solver
        q= -P@q
        #u = np.arr

        # Randomly create a 5*N-element variable
        self.x = np.random.randn(self.N, self.NVars).flatten()
        #x = np.zeros((N,NVars)).flatten()
        #v = np.zeros(N)
        #f = np.zeros(N)

        '''
        ### Zero cost
        P = np.zeros((self.NVars * self.N, self.NVars * self.N))
        P = sparse.block_diag([P])
        self.P = P
        q = np.zeros((self.NVars * self.N))
        self.q = q
        ### Zero cost
        '''
        # x_0 is the position, v_0 is the vel, theta0 is the angle, thetadot0 is the angular vel
        # l\leq Ax \leq u
        A, l, u = self.getAlu(self.x, x0, v0, theta0, thetadot0)
 
        self.m = osqp.OSQP()
        self.m.setup(P=P, q=q, A=A, l=l, u=u , time_limit=0.1, verbose=False)# , eps_rel=1e-2 #  **settings # warm_start=False, eps_prim_inf=1e-1
        self.results = self.m.solve()
        #print(self.results.x)

        # Update for 100 times for the initial step
        #for i in range(100):
        #    self.update(x0, v0, theta0, thetadot0)

    def calcQ(self):
        # recalculate q for the OSQP solver in min P^TxP/2 + qx where q replaces -Pq in min 0.5P(x - q)^2
        THETA = 2
        # Initially q is still the true weights
        q = np.zeros((self.NVars, self.N))

        # Find the angle variables 0~N-1 in self.x
        thetas = np.reshape(self.x, (self.NVars, self.N))[THETA,:]

        # Set target angle to the next np.pi position
        thetas = thetas - thetas % (2*np.pi) + np.pi
        q[THETA,:] = thetas[0] #np.pi #thetas
        # Set target position to 0.5??? middle 0.0???
        q[0,:] = 0.0
        #q[N,0] = -2 * 0.5 * 10
        q = q.flatten()
        # Builde -Pq for the OSQP solver 
        q = -self.P@q
        return q

    def update(self, x0, v0,theta0, thetadot0, a0 = None):
        A, l, u = self.getAlu(self.x, x0, v0, theta0, thetadot0, a0)
        #print(A.shape)
        #print(len(A))
        q = self.calcQ()
        
        # Ax is the renewed value for A
        self.m.update(q=q, Ax=A.data, l=l, u=u)
        self.results = self.m.solve()
        if self.results.x[0] is not None:
            self.x = np.copy(self.results.x)
            self.m.update_settings(eps_rel=1e-3)
        else:
            #self.x += np.random.randn(self.N*self.NVars)*0.1 # help get out of a rut?
            self.m.update_settings(eps_rel=1.1)
            return None
        # The action is the last variable
        return self.x[4*self.N+1]


    def constraint(self, var, x0, v0, th0, thd0):
        #x[0] -= 1
        #print(x[0])
        g = 9.8
        L = 2.0
        gL = g * L
        m = 1.0 # doesn't matter
        I = L**2 / 3
        Iinv = 1.0/I
        dtinv = self.dtinv
        N = self.N

        # Position is in 0~N-1
        x = var[:N]
        # Velocity is in N~2N-1 
        v = var[N:2*N]
        # Angle is in 2N~3N-1
        theta = var[2*N:3*N]
        # Anglur velocity is in 3N~4N-1
        thetadot = var[3*N:4*N]
        # Action is in 4N~5N-1
        a = var[4*N:5*N]
        # All state variables
        dynvars = (x,v,theta,thetadot)
        # Average between each two consecutive variables
        xavg, vavg, thetavg, thdotavg = map(lambda z: (z[0:-1]+z[1:])/2, dynvars)
        # Gradient of postion, velocity, angle, anglur velocity at each time step after the 1st
        dx, dv, dthet, dthdot = map(lambda z: (z[1:]-z[0:-1])*dtinv, dynvars)
        # Gradient of velocity(acceleration) - action
        vres = dv - a[1:]
        # Gradient of position(transient velocity) - average velocity
        xres = dx - vavg
        # Composition of the gravity and external force
        torque = (-gL*np.sin(thetavg) + a[1:]*L*np.cos(thetavg))/2
        # Angular acceleration -  3 * torque/L**2
        thetdotres = dthdot - torque*Iinv
        # Angular gradient(transient anglur velocity) - average anglur velocity
        thetres = dthet - thdotavg
       
        # First 4 only consider the 1st update  while the remaining 4 consider all the variables
        return x[0:1]-x0, v[0:1]-v0, theta[0:1]-th0, thetadot[0:1]-thd0, xres,vres, thetdotres, thetres
        #return x[0:5] - 0.5

        #print(cons)


    def getAlu(self, x, x0, v0, th0, thd0, a0 = None):
        N = self.N
        # Build upper bound (gt=greater than) for the first 2 variables, the position and velocity
        gt = np.zeros((2,N))
        gt[0,:] = -1.5 # 0.15 # x is greaer than 0.15
        gt[1,:] = -3 #-1 #veclotu is gt -1m/s
        #gt[4,:] = -10
        
        # Do not constrain at most 3 steps or 0.1/self.dt steps
        control_n = max(3, int(0.1 / self.dt)) # I dunno. 4 seems to help
        #print(control_n)

        # All variables are larger than -100
        gt[:,:control_n] = -100
        #gt[1,:2] = -100
        #gt[1,:2] = -15
        #gt[0,:3] = -10
        gt = gt.flatten()

        # Build lower bound (lt=less than) for the first 2 variables, the position and velocity
        lt = np.zeros((2,N))
        lt[0,:] = 1.5 #0.75 # x less than 0.75
        lt[1,:] = 3 #1 # velocity less than 1m/s
        #lt[4,:] = 10
        # All variables are lower than -100
        lt[:,:	control_n] = 100
        #lt[1,:2] = 100
        #lt[0,:3] = 10
        #lt[1,:2] = 15
        
        lt = lt.flatten()

        # For inequility l\leq ineqAx \leq u  
        # Each z for a variable from time 0 to N-1
        z = sparse.bsr_matrix((N, N))
    
        # Form a 2N X 5N matrix
        #  The left 2N x 2N matrix is identity and the remainings are all zero
        ineqA = sparse.bmat([[sparse.eye(N),z,z,z,z],[z,sparse.eye(N),z,z,z]]) #.tocsc()
        #print(ineqA.shape)
        #print(ineqA.todense())

        # The gradients are evaluated by using the previously predicted state
        cons = self.constraint(forward.seed_sparse_gradient(x), x0, v0, th0, thd0)
        # The equality constraint is temp \leq Ax - totval \leq temp
        # A include the gradients of all the outputs w.r.t all the variables. 
        # Therefore, each output gradient has 5N elements(for all inputs)
        A = sparse.vstack(map(lambda z: z.dvalue, cons)) #  y.dvalue.tocsc()
        #print(A)
        #print(A.shape)

        # totval is based on the previously predicted state 
        totval = np.concatenate(tuple(map(lambda z: z.value, cons)))


        # Ax_t - x_t = Ax_{t+1}
        # To find x_{t+1} is to find the x that satisfies Ax=Ax_t-x_t=Ax-totval
        # A is composed of the gradient of x
        temp = A@x - totval

        # Initial action is external
        A_ = np.zeros([1, self.x.shape[0]])
        if a0 is not None:
            A_[0, 4*self.N + 1] = 1
        else:
            a0 = 0
        temp = np.concatenate(([a0], temp))
        A = sparse.vstack((A_, A))


        # Combine Eq constraint and Ineq constraint
        A = sparse.vstack((A,ineqA)).tocsc()

        #print(tuple(map(lambda z: z.value, cons)))
        #print(temp.shape)
        #print(lt.shape)
        #print(gt.shape)
        u = np.concatenate((temp, lt))
        l = np.concatenate((temp, gt))

        return A, l, u
