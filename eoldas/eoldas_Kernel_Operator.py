from kernels import *
#from eoldas_obs import *
from eoldas_Lib import *
from eoldas_Operator import Operator
'''
Kernels model interface to eoldas
'''
class Kernel_Operator(Operator):
    
    def preload_prepare(self):
        '''
            Here , we use preload_prepare to make sure 
            the x & any y data are NOT gridded for this
            operator. 
            
            This method is called before any data are loaded, 
            so ensures they are not loaded as a grid.
            '''
        # mimic setting the apply_grid flag in options
        self.y_state.options.y.apply_grid = False
    
    def postload_prepare(self):
        '''
        This is called on initialisation, after data have been read in.
            
        In this method, we have a linear model, so we can
        pre-calculate the kernels (stored in self.K)
        
        '''
        # This is not gridded data, so we have explicit
        # information on self.y.control
        self.mask = self.y.control\
                [:,self.y_meta.control=='mask'].astype(bool) 
        vza = self.y.control[:,self.y_meta.control=='vza'] 
        vaa = self.y.control[:,self.y_meta.control=='vaa'] 
        sza = self.y.control[:,self.y_meta.control=='sza'] 
        saa = self.y.control[:,self.y_meta.control=='saa'] 
        self.kernels = Kernels(vza,sza,vaa-saa,RossHS=False,MODISSPARSE=True,RecipFlag=True,normalise=1,doIntegrals=False,LiType='Sparse',RossType='Thick')
        K = numpy.ones([len(vza),3])
        K[:,1] = self.kernels.Ross[:]
        K[:,2] = self.kernels.Li[:]
        # here, we know the full number of states and their names
        names = self.x_meta.state 
        bands = self.y_meta.state
        nb = len(bands)
        nn = len(names)
        ns = len(vza)
        M = np.eye(nb).astype(bool)
        I = np.eye(nb).astype(float)
        M1 = np.zeros((3,3*nb,nb)).astype(bool)
        I1 = np.zeros((3*nb,nb)).astype(float)
        for i in xrange(3):
            M1[i,i*nb:(i+1)*nb,:] = True

        self.K = np.zeros((ns,nn,ns,nb))  
        for i in xrange(ns):
            this = I1.copy()
            for jj in xrange(3):
                that = this[M1[jj]].reshape((nb,nb))
                that[M] = K[i,jj]
                this[M1[jj]] = that.flatten()
            self.K[i,:,i,:] = this
        self.K = np.matrix(self.K.reshape((ns*nn,ns*nb)))
        self.nb = nb
        self.nn = nn
        self.ns = ns
        # we can form self.H so that y = self.H x
        # to do this, we need to know the location data
        # in self.x  
        testh = self.H(self.x.state)
        self.linear.H_prime = np.matrix(self.K)
        self.isLinear = True

    
    def H(self,x):
        '''
        The reflectance from the kernels
        '''
        x = x.flatten()
        self.Hx = np.array((x * self.K).reshape(self.ns,self.nb))
        return self.Hx
            

if __name__ == "__main__":
	self = tester()
	
