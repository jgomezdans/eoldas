#!/usr/bin/env python
import pdb
import numpy as np
#from eoldas_model import subset_vector_matrix
from eoldas_State import State
from eoldas_ParamStorage import ParamStorage
from eoldas_Operator import *

class Observation_Operator ( Operator ):
    '''
       The EOLDAS Observation Operator Class
        
    This is designed to allow 'plug-n-play' operators provided
    they are spectral models, i.e. model reflectance or radiance
    as a function of the control variables and wavelength.
    Control variables are terms such as vza, vaa, but could for instance
    include polarisation state. Alternatively, terms such as polarisation
    could be included as pseudowavelengths (see Kernels_Operator for 
    non spectral models) 
    
    To use the Observation_Operator class, the user must declare 
    
        operator.obs.name=Observation_Operator 
        
    where obs is an arbitrary name for this operator. 
        
    There should also be a section [operator.obs.rt_model]
        
    This must at least declare `model` e.g.
        
    operator.obs.rt_model.model = rtmodel_ad_trans2
        
    where rtmodel_ad_trans2 is a python class or a shared object library
    (i.e. rtmodel_ad_trans2.so in unix). Other terms pertinent to 
    running the rt model can also be declared, e.g.:
        
    operator.obs.rt_model.use_median=True
        
    which means that the full band pass functions are not used, but
    the median of the bandpass, which is generally much faster.
        
    operator.obs.rt_model.spectral_interval = 1
        
    which declares the spectral interval of the model. This is assumed to
    be in nm.
        
    operator.obs.rt_model.ignore_derivative = False
        
    Set to True to override loading any defined derivative functions in the library and use numerical approximations instead
        
    The declared class (e.g. rtmodel_ad_trans2) is loaded in the method postload_prepare, which happens after all data and configuration information have been loaded into the operator (at the end of __init__). If this fails for any reason (e.g. it does not exist in the PYTHONPATH) then an exception is raised as the eoldas cannot continue. For those interested in where items are stored, this will be in self.options.rt_model.model.
        
    At this point, we also load methods from the rt_model. This is done by the (private) method _load_rt_library() and loads the methods in this class as `self.rt_library.<method_name>`
        
    These must include:
        rt_model
    
    It may also include:    
    
        rt_modelpre, rt_modelpost
        
    which are setup functions (e.g. memory allocation/deallocation) for the rt
        model that are called in the class constructir and destructor 
        respectively.

    If the rt model can calculate adjoints, this is declared via the methods:
        
        rt_modeld, rt_modeldpre, rt_modeldpost, rt_modelpred
        
    where `rt_modeld` calculated the adjoint, `rt_modeldpre` is a method for setting up the mode (e.g. memory allocation) that is called after `rt_modelpre`, `rt_modeldpost` is called after `rt_modelpost`.
        
    `rt_modelpred` is normally equivalent to rt_model but may be called in its place if adjoints are used.
        
    '''

    def H( self, x):
        '''
        The forward model of the operator
            
        Load the state vector from self.state.x and calculate BRF for all 
        observations and wavebands, either using a full bandpass function or
        a single wavelength (it's median), and maybe initialising the adjoint
        code if that's required.

        '''
        
        self.state = x.reshape(self.x.state.shape)
        #  initialise to 0 
        self.linear.H = self.linear.H*0.
	if not self.isLoaded:
	    # we only have to load these once
            try:
                self.doy = self.y.location[:,self.y_meta.location == 'time']
            except:
                pass
            try:
                self.mask = self.y.control\
                    [:,self.y_meta.control=='mask'].astype(bool) 
                self.vza = self.y.control[:,self.y_meta.control=='vza'] 
                self.vaa = self.y.control[:,self.y_meta.control=='vaa'] 
                self.sza = self.y.control[:,self.y_meta.control=='sza'] 
                self.saa = self.y.control[:,self.y_meta.control=='saa'] 
	        self.isLoaded = True
            except:
                self.logger.error('error loading control information in %s'%\
                                'Observation_Operator')
                self.logger.error(\
                                'Check the configuration for mask,vza,sza,vaa,saa')
                raise Exception('error loading control information')

        # bands_to_use is of dimensions (nbands,nl)
        # and contains the normalised bandpass functions
        # All spectral information is contained in
        # self.y_meta.spectral  self.rt_library.rt_modelpred
                # rt_modelpred
        for i_obs in np.where(self.mask)[0]:
            self.linear.H[i_obs] = self.model_reflectance(i_obs)
	return self.linear.H

    def J(self):
	'''
	The cost function for the observation operator
	'''
        x,Cx1,xshape,y,Cy1,yshape = self.getxy()
        self.Hx = self.H(x)
	diff = (y -  self.Hx.flatten()) 
        part1 = diff * Cy1
	self.linear.brf_ad = (-part1).reshape(self.Hx.shape)
        self.linear.J = 0.5*np.dot(part1,diff)
	return self.linear.J	

    def J_prime_prime(self):
	'''
	The second order differntial
	
	For this operator, this is independent for each observation
	so we can do it numerically, but changing each observation
	
	'''
	x,Cx1,xshape,y,Cy1,yshape = self.getxy()
	J,J_prime0 = self.J_prime()
	xshape = self.x.state.shape
        if not 'linear' in self.dict():
            self.linear = ParamStorage()
        if not 'J_prime_prime' in self.linear.dict():
            self.linear.J_prime_prime = \
                np.zeros(xshape*2)
        else:
            self.linear.J_prime_prime[:] = 0

	for i in xrange(xshape[-1]):
	    ww = np.where(deriv)
	    ww2 = tuple(ww[:-1]) + tuple([ww[0]*0+i]) + tuple([ww[-1]] )+ tuple([ww[0]*0+i])
	    self.linear.J_prime_prime[ww2] = deriv[ww]
	return J,J_prime,J_prime_prime	

    def J_prime_full( self):
	'''
	The derivative of the cost function for the observation
        operator.

        This is achieved with an adjoint if available
        '''
        J = self.J()
        x,Cx1,xshape,y,Cy1,yshape = self.getxy()
        state = x.reshape(self.x.state.shape)
        for i_obs in np.where(self.mask)[0]:
            self.linear.J_prime[i_obs] = self.rt_library.rt_modeld(i_obs+1,\
		state[i_obs], [self.vza[i_obs]], [self.vaa[i_obs]], \
		self.sza[i_obs], self.saa[i_obs], self.linear.brf_ad[i_obs],\
		self.y_meta.spectral.bands_to_use )			
        return J,self.linear.J_prime

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
	# method J_prime_approx_1 is appropriate here
	self.J_prime_approx = self.J_prime_approx_1
 
    def postload_prepare(self):
        '''
        This is called on initialisation, after data have been read in
            
        '''
        try:
            self.rt_model = self.options.rt_model
            self.rt_class = self.rt_model.model
            self._load_rt_library ( rt_library=self.rt_class)
        except:
            raise Exception('rt library %s could not be loaded in %s'%\
                      (self.options.rt_model.model,'Observation_Operator'))   
        try:
            self.setup_rt_model()
        except:
            raise Exception('Error initialising the rt model %s'%\
                            self.options.rt_model.model)
		
    def setup_rt_model  ( self ):
        """
        This sets up the RT model (and adjoint if available)
        by calling any preparation methods.
            
        """
        if not 'linear' in self.dict():
            self.linear = ParamStorage()
        if 'y' in self.dict():
            self.linear.H = np.zeros(self.y.state.shape)
        else:
            self.linear.H = np.zeros(self.x.state.shape)
        self.nv = 1
        self.npt = len(self.y.state)

        self.linear.J_prime = np.zeros(self.x.state.shape)

        if not self.rt_model.use_median:
            self.bandIndex = self.y_meta.spectral.all_bands
        else:
            self.bandIndex = self.y_meta.spectral.median_bands
        self.linear.brf_ad = np.ones((self.npt,len(self.bandIndex)))

        self.rt_library.rt_modelpre(np.array(self.bandIndex) + 1 )
        self.rt_library.rt_modeldpre (self.npt )

        self.x_orig = self.x.state.copy()
        if self.rt_model.use_median:
            bands_to_use = self.y_meta.spectral.median_bands_to_use
            bandpass_library = self.y_meta.spectral.median_bandpass_library
            index = self.y_meta.spectral.median_bandpass_index
        else:
            bands_to_use = self.y_meta.spectral.bands_to_use
            bandpass_library = self.y_meta.spectral.bandpass_library
            index = self.y_meta.spectral.bandpass_index
        
        self.y_meta.spectral.bands_to_use = \
                    np.zeros((len(bands_to_use),len(self.bandIndex)))
        for (i,bandname) in enumerate(self.y_meta.spectral.bandnames):
            fullb = bandpass_library[bandname]
            this = fullb[index[bandname]]
            this = this/this.sum()
            ww = np.where(np.in1d(self.bandIndex,index[bandname]))[0]
            self.y_meta.spectral.bands_to_use[i,ww] = this
                              
    _nowt = lambda self, *args : None
    
    def _load_rt_library ( self, rt_library ):
        """
        A method that loads up the compiled RT library code and sets some 
        configuration options. This method tries to import all the methods and
        make them available through `self.rt_library.<method_name>`. It is also
        a safe importer: if some functions are not available in the library,
        it will provide safe methods from them. Additionally, while importing,
        a number of configuration options in the class are also updated or set 
        to default values.
        
        Parameters
        -----------
        rt_library : string
        This is the name of the library object (.so file) that will be 
        loaded
            
            """
        from eoldas_Lib import sortopt
        import_string = "from %s import " % ( rt_library )
        self.logger.debug ("Using %s..." % rt_library )
        
        self.rt_library = sortopt(self,'rt_library',ParamStorage())
        # 1. Import main functionality
        try:
            self.logger.debug ("Loading rt_model")
            exec ( import_string + "rt_model" )
            self.rt_library.rt_model = rt_model
        except ImportError:
            self.logger.info(\
                "Could not import basic RT functionality: rt_model")
            self.logger.info(\
                "Check library paths, and whether %s.so is available" % \
                rt_library)
            raise Exception('error importing library %s'%rt_library)
                            
        # 1a. Import conditioning methods that can be ignored
        try:
            exec ( import_string + 'rt_modelpre' )
            self.rt_library.rt_modelpre = rt_modelpre
        except:
            self.rt_library.rt_modelpre = self._nowt
        try:
            exec ( import_string + 'rt_modelpre' )
            self.rt_library.rt_modelpost = rt_modelpre
        except:
            self.rt_library.rt_modelpost = self._nowt    
                        
        # 2. Try to import derivative
        self.rt_model.ignore_derivative = sortopt(self.rt_model,\
                                'ignore_derivative',False)

        if self.rt_model.ignore_derivative == False:
            try:
                exec ( import_string + \
                    "rt_modeld, rt_modeldpre, rt_modeldpost" + \
                    ", rt_modelpred" )
                self.rt_model.have_adjoint = True
		self.J_prime = self.J_prime_full
                self.rt_library.rt_modeld = rt_modeld
                self.rt_library.rt_modeldpre = rt_modeldpre
                self.rt_library.rt_modeldpost = rt_modeldpost
                self.rt_library.rt_modelpred = rt_modelpred
            except ImportError:
                self.logger.info("No adjoint. Using finite differences approximation.")
                self.rt_model.have_adjoint = False
        else:
            self.logger.info("ignoring adjoint. Using finite differences approximation.")
            self.rt_model.have_adjoint = False
                            
        self._configure_adjoint ()
                            
        try:
            exec( import_string + "rt_model_deriv")
            self.rt_library.rt_model_deriv = rt_model_deriv
        except ImportError:
            self.rt_library.rt_model_deriv = None

 
    def _configure_adjoint ( self ):
        """
            A method to configure the adjoint code, whether the _true_ adjoint or
            its numerical approximation is used.        
            """
        if not self.rt_model.have_adjoint:
            self.rt_library.rt_modeld = None
	    self.J_prime = self.J_prime_approx_1
            self.rt_library.rt_modeldpre = lambda x: None
            self.rt_library.rt_modeldpost = lambda : None
            self.rt_library.rt_modelpred = self.rt_library.rt_model
        
    model_reflectance = lambda self,i_obs : self.rt_library.rt_modelpred( i_obs+1, \
		self.state[i_obs], [self.vza[i_obs]], [self.vaa[i_obs]], \
		self.sza[i_obs], self.saa[i_obs],self.y_meta.spectral.bands_to_use )
    model_reflectance.__name__ = 'model_reflectance'
    model_reflectance.__doc__ =  """
            Calculate the cost of a single observation\
            
            i_obs is the observation index (1-based)
	    
	    It is in this function that we interface with
	    any rt libraries
            
        """
        
    def hessian(self,use_median=True,epsilon=1.e-5,linear_approx=False):
        """
            observation operator hessian
            """
        self.logger.info ("Starting Hessian ")
        nparams = self.setup.config.params.n_params 
        if not hasattr ( self, "hessian_m"):
            self.hessian_m = np.zeros([self.setup.grid_n_obs*self.setup.grid_n_params,self.setup.grid_n_obs*self.setup.grid_n_params])
        else:
            self.hessian_m[:,:] = 0
        for i_obs in xrange( self.obs.npt ):
            nbands = self.obs.nbands[i_obs]
            if self.obs.qa[i_obs] > 0:
                # get the parameters (full set)
                x = self.setup.store_params[self.setup.obs_shift[i_obs],:]
                # form a mask of which of these vary
                mask = (self.setup.fix_params[self.setup.obs_shift[i_obs], :] == 1) + (self.setup.fix_params[self.setup.obs_shift[i_obs], :] == 2) 
                dobs_dx = np.matrix(self.dobs_dx_single_obs (x[mask], i_obs, use_median=use_median, epsilon=epsilon,linear_approx=linear_approx))
                # this is the same as self.hprime[i_obs,:.nbands,mask]
                # so it has dimensions [nbands,n_params] 
                # The call to self.dobs_dx_single_obs also calculates the brf for that sample, in 
                # self.store_brf[i_obs, :nbands] 
                thism = dobs_dx * self.obs.real_obsinvcovar[i_obs] * dobs_dx.T 
                #thism = dobs_dx * self.obs.obsinvcovar[i_obs] * dobs_dx.T
                # try to put in the second order term
                #try:
                #    thatm = self.obs.real_obsinvcovar[i_obs] * np.matrix(self.h_prime_prime[i_obs,:nbands,mask]).T
                #    diff = np.matrix(self.obs.observations[ i_obs, :nbands] - self.store_brf[i_obs, :nbands])
                #    thism = thism - diff * thatm
                #except:
                #    pass
                # thism is a n_params x n_params matrix
                # pack it into the big one
                # so, which time does this obs belong to?
                this = self.setup.obs_shift[i_obs]
                nv = thism.shape[0]
                self.hessian_m[this*nv:(this+1)*nv,this*nv:(this+1)*nv] = self.hessian_m[this*nv:(this+1)*nv,this*nv:(this+1)*nv] + thism
        oksamps = np.where(self.setup.fix_params.flatten()>0)[0]
        xx = self.hessian_m[oksamps]
        return xx[:,oksamps]
    
    def set_crossval ( self, leave_out = 0.2 ):
        """
            This method allows to set a cross validation strategy in the data by
            selecting a fraction of the samples that can be left out from those
            that have a good QA. This is expressed as a fraction, by default we
            use 20%. The location of these samples is random.
            
            Parameters
            ------------
            leave_out : float
            The fraction of samples to "leave out"
            """
        from random import sample
        ndoys = self.obs.qa.sum()
        leave_out_xval = int ( np.ceil(ndoys*leave_out) )
        self.logger.info ("Cross validation: Leaving %d samples out" % \
                          leave_out_xval )
        xval_locs = np.array([np.where ( self.obs.doys==i) \
                              for i in sample(self.obs.doys[ self.obs.qa==1], \
                                              leave_out_xval)]).squeeze()
        self.xval[ xval_locs ] = -1
    
    
    def set_crossval_single ( self, xval_loc ):
        """
            This method allows to set a cross validation strategy in the data by
            selecting a single observation to be "left out". Looping over all the
            observations is required for Generalised cross validation strategies.
            
            Parameters
            -----------
            xval_loc : integer
            location of the sample left out.
            """
        self.logger.info ("Cross validation: Leaving %d-th sample out" % \
                          xval_loc )
        self.xval = self.obs.qa.copy()
        self.xval[ xval_loc ] = -1
    
    
    
    def calculate_crossval_err ( self ):
        """
            Calcualte the crossvalidation error. Once the DA process has completed,
            this method calculates the mismatch between the observations that were
            left out of the DA optimisation and their estimates. They are weighted
            by the observational uncertainty in each case.
            
            """
        xval_err = 0.0
        for i in xrange( self.obs.npt ):
            if self.xval[i] == -1:
                difference = self.obs.observations[ i, :self.obs.nbands[i] ] - \
                    self.rt.brf[ i, :self.obs.nbands[i] ]
                inv_obs_covar = self.obs.obsinvcovar[i]
                part1 = np.dot ( inv_obs_covar, difference )
                xval_err += np.dot( part1, difference )
                
                for j in xrange ( self.obs.nbands[i] ):
                    info_str = "Doy: %d Obs #%d: " % ( self.obs.doys[i], i )
                    info_str = info_str + "B%d Obs: %g Model: %g Dif: %g" % \
                        ( j+1, self.obs.observations[ i, j ], \
                         self.rt.brf [i,j],  self.obs.observations[ i, j ] - \
                         self.rt.brf [i,j] )
                    self.logger.info ( info_str )
        return ( xval_err, self.params_x, self.rt.brf, self.obs.observations )
    

