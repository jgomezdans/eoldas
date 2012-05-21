#!/usr/bin/env python
import pdb
import sys
import numpy as np
import os

from eoldas_ParamStorage import ParamStorage
#from eoldas_conf import ConfFile

def sortopt(options,key,value):
    '''
    A utility to set the dictionary entry to
    value if it is not set, and in any case return the
    entry.
        
    '''
    if not hasattr(options,key):
        options[key] = value
    return options[key]

def isfloat(s):
    '''
    Returns True if the value is a float or int, False otherwise
    '''
    try:
        i = float(s)
        return True,i
    except ValueError, TypeError:
        return False,0


class eoldas_setup(object):
    """
    This is redundant

    """
    def __init__(self,datafile,options):
      
        self.options = options 
        # sort logging
        self.options.general.logdir         \
            = self.__getopt(self.options.general,'logdir',"logs")
        self.options.general.logfile        \
            = self.__getopt(self.options.general,'logfile',"logfile.log") 
        # 1.- set up logging for this particular run
        self.logger = set_up_logfile(self.options.general.logfile,\
                        name="eoldas_setup",logdir=self.options.general.logdir)

        try:
            self.setup()            
        except:
            self.logger.error("Unable to access critical elements of the options. See help(eoldas_setup.setup) for details")
            sys.exit(-1)
        # read conf file(s)
        self.configfile = data_file
        #config = ConfFile(self.configfile,dirs=dirs,log_name=self.logger)
        if len(config.infos) == 0:
            self.fail = True
            return
        # not sure what to do if multiple config files???
        # just take the first one at the moment
        self.config  = config.infos[0]
        # update with cmd line options
        self.config.update(self.options,combine=True)

        self.logger.info("Model sd scaling by %f over that defined in the config file" % self.config.general.model_sd)
        self.ok = self.process_config_file()

    def setup(self):
        '''
        Access and set up critical elements of the problem.

        This includes:
        
        The existence of:
            options.parameter.names (a list)

        '''
        options = self.options
        options.parameter.n_params = len(options.parameter.names)
        

        self.options = options

    def __getopt(self,options,key,value):
        if not hasattr(options,key):
            options[key] = value
        return options[key]

    def __pcheck(self,thisdict,name):
        '''
        Check that name exists in thisdict
        '''
        try:
            if name in thisdict.dict():
                return True
            else:
                return False
        except:
            return False

    def __min(self,a,b):
        '''
        Min utility for 2 numbers, ignoring None
        '''
        if a == None:
            out = b
        elif b == None:
            out = a
        else:
            out = np.min([a,b])
        if out == None:
            return 0
        else:
            return out     

        # the next critical thing is some observations
        obs = load_brdf_file (brf,self.config,bandpass_names={})
        if obs == False:
            return False

        self.config.operator.obs.update(obs,combine=True)

        # sets up an initial version of x_init
        # which is in the observation 'space' (ie one per obs)
        for n_par in xrange ( self.config.params.n_params ):
            #self.default_vals[n_par] = prior_mean[n_par]
            if np.all( self.obs.x_init[ :, n_par] == 0 ): # No
                self.obs.x_init [ :, n_par ] = self.default_vals [ n_par ]
        # try brfinit_files
        # which can overwrite x_init
        try:
            if self.options.preload != []:
                brfinit_files = self.options.preload
                self.brfinit_files['override'] = brfinit_files
        except:
            if self.options.preload != []:
                self.brfinit_files = ParamStorage ()
                self.brfinit_files['override'] = self.options.preload
                # this is a hack to get the same structure
                self.brfinit_files = self.brfinit_files.dict()
        thisdoys = None
        if self.brfinit_files is not None:
            # this is not consistent with having multiple files
            # and is a bit of a mess 
            for key in self.brfinit_files.keys():
                if type(self.brfinit_files[key]) == type([]):
                    initfile = self.brfinit_files[key][0]
                else:
                    initfile = self.brfinit_files[key]
                #(acovar, abandwidth, abands, anpt, anbands_max, alocation, \
                #    awhichfile, anbands, adoys, aqa, atheta_v, atheta_i,aphi_v, \
                #    aphi_i, aisobs, aobs, aobscovar, aparams_x) = \
                #    load_brdf_file(initfile)
                (thisdoys,thisparams) = self.read_parameters(initfile,confdir=confdir)
                # if fail, thisdoys is None
                #self.obs.x_init[:,:] = aparams_x[:,:]

        if thisdoys == None:
            self.brfinit_files = None
        # For convenience, we can invert the observation covariance matrices
        self.obs.obsinvcovar = []
        self.obs.real_obsinvcovar = []

        for sample_no in xrange( self.obs.npt ):
            temp_mtx = np.matrix( self.obs.obscovar[ sample_no ] ).I
            if self.config.params.scale_cost:
                self.logger.info ("Scaling obs by %f" % \
                    float(self.obs.npt*self.obs.nbands[0] ) )
                self.obs.obsinvcovar.append ( \
                        temp_mtx/float((self.obs.npt*self.obs.nbands[sample_no] )))
            else:
                self.obs.obsinvcovar.append( temp_mtx )
            self.obs.real_obsinvcovar.append (temp_mtx)    

        # if there is anything non zero in x_init, set params_x to that
        if self.obs.x_init.sum() > 0:
            self.params_x = self.obs.x_init.copy()
        else:
            self.params_x = np.zeros ((self.obs.npt, \
                                        self.config.params.n_params))
        # determine which params to fix, based primarily on solve_for flags
        fix_params = define_fixparams(self.parameters, \
            solve_for=self.solve_for,prior_sd=self.prior_sd,model_unc_cfg=self.model_unc_cfg)

        self.config.params.n_model_params = np.sum(fix_params==3) + np.sum(fix_params==4)

        # set up the grid based on the span of unique doys
        self.unique_doys, self.quantised_doys, self.obs_shift = quantise_time ( self.obs.doys, \
                                                self.time_quant ,grid=grid)
        self.grid_n_obs = self.unique_doys.shape[0]

       

        self.fix_params = np.tile(fix_params, self.grid_n_obs).reshape((self.grid_n_obs,self.config.params.n_params))

        self.logger.info ("%d days, %d quantised days" % ( len(self.unique_doys), \
            len(self.quantised_doys) ) )
        self.grid_n_params = fix_params.shape[0]

        # set up a grid model representation from self.params_x
        # we will use then when loading
        # self.params_x is a full representation in obs space
        # so we expand it to the model grid space
        self.store_params = self.get_x(self.params_x,self.fix_params*0.)
 
        # but this may contain zeros if a parameter has not been defined so should be set to the default value
        # or maybe interpolations is better
        udoys = np.unique(self.obs.doys)
        try:
            where_udoys = np.in1d(self.unique_doys,udoys)
        except:
            where_udoys = np.zeros_like(self.unique_doys).astype(np.bool)
            for i in udoys:
                w = np.where(self.unique_doys == i)
                where_udoys[w] = True
        for i in xrange(self.grid_n_params):
            self.store_params[:,i] = np.interp(self.unique_doys,self.unique_doys[where_udoys],self.store_params[where_udoys,i])
        
        # override this with data from brfinit_files
        if self.brfinit_files is not None:
            # zeroth ...
            # pull out elements of thisdoys that appear in  self.unique_doys
            
            # first interpolate thisparams onto the grid
            store_params = self.store_params*0.
            new_thisdoys = np.zeros( self.store_params.shape[0]).astype(np.int)
            # loop over thisdoys and load where appropriate
            for (i,j) in enumerate(thisdoys):
                ww = np.where(j == self.unique_doys)
                store_params[ww,:] = thisparams[i,:] 
                new_thisdoys[ww] = j
            thisdoys = new_thisdoys
            udoys = np.unique(thisdoys)
            try:
                where_udoys = np.in1d(thisdoys,udoys)
            except:
                where_udoys = np.zeros_like(thisdoys).astype(np.bool)
                for i in udoys:
                    w = np.where(where_udoys  == i)
                    where_udoys[w] = True
            for i in xrange(self.grid_n_params):
                self.store_params[:,i] = np.interp(self.unique_doys,self.unique_doys[where_udoys],store_params[where_udoys,i])

        # deal with model uncert
        self.model_unc = np.ones((self.fix_params.shape[1])) 
        for ( i, k ) in enumerate ( self.parameters ):
            if self.model_unc_cfg [ k ] > 0:
                self.model_unc[i] = self.model_unc[i] * self.model_unc_cfg [ k ]
        self.prior_m = np.array([self.prior_mean[k] for k in self.parameters ])
        self.prior_std = np.array([self.prior_sd[k] for k in self.parameters ])

        return #( prior_mean, prior_sd, model_unc, abs_tol, scale_cost)

    def get_x( self, x_obs,  x_model, summer=False):
        """
        return an x_model representation which has parameter values for the 
        complete model grid. The array x_obs has a representation of the 
        parameter values only for observation points, whereas x_model is 
        typically defined over the whole assimilation period/region.

        When loading parameters in this way (from observation space
        to model space, only the parameter associated with the first 
        observation at a particular point is taken (summer=False)

        When loading derivatives (e.g. when using the adjoint) we need
        to sum over all observation grid points (summer=True)
        
        Parameters
        -----------
        x_obs : array-like
            The state vector representation that corresponds to the observations
        x_model : array-like
            The state vector representation that corresponds to the assimilation
            interval.
        """
        if summer == False:
            for i in np.unique(self.obs_shift).astype(np.int):
                w = np.where(self.obs_shift == i)[0][0]
                x_model[i,:] = x_obs[w,:]
        else:
            x_model[:,:] = 0.
            for i in np.unique(self.obs_shift).astype(np.int):
                w = np.where(self.obs_shift == i)[0]
                for j in w:
                    x_model[i,:] = x_model[i,:] + x_obs[j,:]
        return x_model

    def write_parameters(self,filename,params,ofmt='ASCII'):
        """
        Write the parameters out to filename
        """
        if ofmt == 'ASCII':
            self.logger.info ( "Saving parameters to %s" % filename)
            fp = open(filename,'w')
            fp.write("# PARAMETERS %s\n" % "".join ( [ "%s " % i  for i in self.parameters])) 
            for i in xrange(self.grid_n_obs):
                 fp.write("%f %s\n" % (self.unique_doys[i],"".join ( [ "%s " % j  for j in params[i,:]])))
            fp.close()

def sortlog(self,logfile,logger,name="eoldas",logdir=None,debug=True ):
    '''
    A safe interface to logging
    for passing log information between lots of classes
    '''
    import logging
    import time
    from eoldas_Lib import set_up_logfile,dummyprint

    if type(self).__name__  == 'SpecialVariable' and logger != None:
        return logger
    try:
        if 'logger' in self.dict():
            return self.logger
    except:
        pass
    try:
        if 'logger' in self.keys():
            return self.logger
    except:
        pass
    try:
        if type(self).__name__ != 'SpecialVariable':
            this = self.logger
            return self.logger
    except:
        pass

    if logdir == None:
        logdir = '.'

    if name == None:
        name = type(self).__name__ + '.' + str(time.time())

    if logger:
        if type(self).__name__  == 'SpecialVariable':
            return logger
        logger.info('Setting up logger for %s'%name)
        logger = logging.getLogger(name)
	logger.info('Set up')
        return logger
        
    if logfile == None or name == None:
        logger = ParamStorage()
        logger.info  = lambda x:dummyprint( "Info: %s"%x)
        logger.debug = lambda x:dummyprint( "Debug: %s"%x)
        logger.error = lambda x:dummyprint( "Error: %s"%x)
        return logger
    logger = set_up_logfile(logfile,name=name,logdir=logdir,debug=debug)
    return logger

def dummyprint(x):
    print x

def set_up_logfile ( logfile, name="eoldas",logdir=None,debug=True ):
    """
    A convenience function to set up the logfiles
        
    The logfile is generated in logfile which may be in
    logdir if it is not an absolute filename.
        
    The item 'name' is used as an identifier in the log.
        
    The debug sets the level of logging (not curretly in use)
        
    """
    from os import makedirs,sep
    from os.path import join, isabs, dirname,exists
    import logging
    #try:
    #    logging.shutdown()
    #except:
    #    pass
    logger = logging.getLogger(name)
    if logdir == None:
        # get the dir name from logfile
        logdir = dirname(logfile)    
    else:
        if not isabs(logfile):
            logfile = join(logdir, logfile)
    # Test whether the log directory exists, and if not, create it
    logdir = os.path.dirname(logfile)
    if not exists( logdir ):
        try:
            os.makedirs( logdir )
        except OSerror:
            raise Exception(str(OSerror),"Prevented from creating logdir %s" % logdir)

    if debug == True:
        logger.setLevel( logging.DEBUG )    
    fh = logging.FileHandler(logfile)
    if debug == True:
        fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - " + \
            "%(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("starting logging")
    return logger
   
def get_filename(fname,datadir=['.','~/.eoldas'],env=None,multiple=False): 
    '''
    A very useful utility method to get a full list of potential
    filenames.

    Arguments:
        fname    : filename (relative or absolute)
    
    Options:
        datadir=[]  : list of directories to search in
        env=None : environment variable to give a colon (:)
                   separated string to provide a list of directories
                   to search in

    The filename fname is first expanded to take account of ~ or 
    other symbols.
    
    If the resultant filename is an absolute filename (i.e. starts 
    with '/' on Unix or related on windows, and it is confirmed to 
    be a file, and is readable, then only this filename is returned.

    If the fname is absolute, but it is not a file, then the basename 
    of fname is assumed to be what the user wanted to refer to as fname. 
    The basename is the second half of what is returned by split(fname). 
    
    If the filename is a relative filename then the list dirs is 
    first searched for readable files fname .
        
    If the option multiple is set True, then a complete list of
    readable files is returned. Otherwise, just the first readable
    entry encountered.

    '''
    import os
    allfiles = []
    orig_fname = fname
    fname = os.path.expanduser(fname)
    if os.path.isabs(fname):
        if os.path.isfile(fname) and os.access(fname,os.R_OK):
            if multiple:
                allfiles.append(fname)
            else:
                return fname,(0,'')
        # only take the basename if its absolute
        fname = os.path.basename(fname)
    for d in datadir:
        this = os.path.expanduser('%s/%s' % (d,fname))
        if os.path.isfile(this) and os.access(this,os.R_OK):
            if multiple:
                allfiles.append(this)
            else:
                return this,(0,'')
    if env == None:
        if not multiple or not len(allfiles):
            return '',(1,"Error: file %s not found in %s \
                   and environment variable not set" % \
                   (orig_fname,str(datadir)))
    thisenv = os.getenv(env)
    if thisenv == None:
        if not multiple or not len(allfiles):
            return '',(1,"Error: file %s not found in %s \
                   and environment variable %s set \
                   but contains nothing" % \
                   (orig_fname,str(datadir),str(env)))
    else:
        for d in thisenv.split(':'):
            filenames.append(os.path.expanduser('%s/%s' % (d,fname)))
            if os.path.isfile(this) and os.access(this,os.R_OK):
                if multiple:
                    allfiles.append(this)
                else:
                    return this,(0,'')
        if not multiple or not len(allfiles):
            return '',(1,"Error: file %s not found in %s \
                   and environment variable %s set \
                   but file not found in directories %s" % \
                   (orig_fname,str(datadirs),str(env),str(thisenv)))
    return allfiles,(0,'')

set_default_limits = lambda location : [[0,None,1]] * len(location)
set_default_limits.__name__ = 'set_default_limits'
set_default_limits.__doc__ = '''
    For a given location list of len len(location)
    return a set of default limits information [0,None,1] of
    the right length.
    '''

check_limits_valid = lambda limits : [[i[0],i[1],max(1e-20,i[2])] \
                                      for i in list(limits)]
check_limits_valid.__name__ = 'check_limits_valid'
check_limits_valid.__doc__ = '''
    Return a limits list that doesn't contain zero in the step term.
    (if it is zero, it is set to 1e-20, i.e. a small number)
    '''

quantize_location = lambda value, limits : \
    [int(round(((value[i]-limits[i][0])/limits[i][2]))) \
    for i in xrange(len(limits))]
quantize_location.__name__='quantize_location'
quantize_location.__doc__ = """
    Quantize a value by limits and return the quantized value.
    """
dequantize_location = lambda qvalue, limits : \
    [limits[i][2]*qvalue[i]+limits[i][0] for i in xrange(len(limits))]
dequantize_location.__name__='dequantize_location'
dequantize_location.__doc__='''
    Dequantize a value by limits and return the unquantized value.
    '''
get_col = lambda index,liner : float(len(index) and liner[index])
get_col.__name__ = 'get_col'
get_col.__doc__ = '''
    For a list liner, return a float representation of the value
    in column index, or 0 if the len(index) is zero.
    '''


                

class ObservationOperator ( object ):
    '''
       Depreciated 
    '''
    def __init__ ( self, nbands, nbands_max, npt, bandwith, obscovar, \
                        location, whichfile, doys, qa, theta_v, theta_i, \
                        phi_v, phi_i, isobs,  params_x, \
                        obs=0.0, bandpass_library=False ) :
                           
        # Define a configuration container. Makes everything look like Java
        # Containers can then be useful for quickly listing all variables :)
        self.config = ParamStorage ()
        self.config.spectral = ParamStorage ()
        self.config.rt_model = ParamStorage ()
        self.observations = ParamStorage ()
        
        self.config.rt_model.nparams = rt_getnparams ()
        
        self.config.npt = npt # Number of points
        self.config.nv = 1 # Always set to 1. No questions asked
        
        self._setup_spectral_config ( nbands, nbands_max, bandwith )
        self._setup_rt_model ()
        self._setup_bandpass_funcs ( bandwidth, bandpass_library )
        self._setup_geometry ( theta_v, phi_v, theta_i, phi_i )
        
        self.params_x = np.zeros (( self.config.npt, \
                        self.config.rt_model.nparams ) )
        self.observations.brf = np.zeros([self.config.npt, \
                                self.config.spectral.nbands_max] )
        self.observations.obs = np.zeros([self.config.npt, \
                                self.config.spectral.nbands_max] )
        # obs will be set to zero if not loaded
        self.observations.obs[:,:] = obs
        # setting self.brf_ad to 1 means that we calculate the model derivative 
        # by default
        self.brf_ad = np.ones([self.config.npt, \
                                self.config.spectral.nbands_max] )
        self._set_minmax ( )
        self._set_x( x )

    def _set_minmax ( self ):
        """
        Find out the boundaries for each parameter from the RT model using the
        rt_getminmax method
        """
        minmax = rt_getminmax().split( ":" )[ :self.config.rt_model.nparams ]
        self.x_min = [ float(s.split(";")[0]) for s in minmax ]
        self.x_max = [ float(s.split(";")[1]) for s in minmax ]
        self.x_min = np.array ( self.x_min )
        self.x_max = np.array ( self.x_max )
        
        
    def destroy ( self ):
        """Some memory free-ing
        
        Deallocates arrays in the RT code, and hence frees memory
        """
        rt_modelpost ()
        rt_modeldpost ()
        
  
    def rt_model ( self ):
        """Run the fwd model
        """
        for i_obs in xrange ( self.config.npt ):
            wavebands = self._get_bands ( i_obs )
            self.observations.brf [ i_obs, : ] = rt_model ( i_obs, \
                                    self.params_x[i_obs, :], 
                                    [ self.observations.theta_v [ i_obs ]], \
                                    [ self.observations.phi_v [ i_obs ]], \
                                    self.observations.theta_i [ i_obs ], \
                                    self.observations.phi_i [ i_obs ], \
                                    wavebands )

    def rt_model_ad ( self ):
        """Run the adjoint model
        """
        for i_obs in xrange ( self.config.npt ):
            wavebands = self._get_bands ( i_obs )
            self.observations.x_ad [ i_obs, : ] = rt_modeld ( i_obs, \
                                    self.params_x[i_obs, :], 
                                    [ self.observations.theta_v [ i_obs ]], \
                                    [ self.observations.phi_v [ i_obs ]], \
                                    self.observations.theta_i [ i_obs ], \
                                    self.observations.phi_i [ i_obs ], \
                                    self.brf_ad[i_obs, :], \
                                    wavebands )

    def _set_x ( self, x, quiet=False, boundcheck=True ):
        xx = self.params_x
        if np.isscalar ( x ):
            if x is not False:
                self.params_x[:, :] = x
        else:
            if  x.size == self.config.rt_model.nparams*self.config.npt :
                self.params_x [:, :] = x
            else:
                if x.size == self.config.rt_model.nparams:
                    for i in xrange ( self.config.rt_model.nparams ):
                        self.params_x[ i, : ] = x
                else:
                    print "Error in self._set_x"
                    sys.exit(-1)
                        
    def _setup_geometry ( self, theta_v, phi_v, theta_i, phi_i ):
        """
        Update the geometry
        """
        self.observations.theta_v = theta_v
        self.observations.theta_i = theta_i
        self.observations.phi_v = phi_v
        self.observations.phi_i = phi_i
        
        
        
    def _setup_rt_model ( self ):
        """
        Reserve memory and generally speaking, set up the RT model
        """
        rt_modelpre()
        rt_modeldpre ( self.config.npt )
        
def sort_non_spectral_model(parameter,ops,logger=None):
    '''
        For a non-spectral model, e.g. Kernels observation operator
        we have to increase size of the model state vector by
        multiplying by the number of observation 'states' (wavebands).
        
        For example, if we have a MODIS kernel model, with 3 parameters
        
        Isotropic RossThick LiSparseModis
        
        and we have observations in 7 MODIS wavebands:
        
        465.6 553.6 645.5 856.5 1241.6 1629.1 2114.1
        
        Then we require 7 x 3 = 21 state variables per observation location
        (i.e. per day, row, col) to represent this.
        
        We do this by forming pseudo state variables e.g.
        
        Isotropic.465.6 Isotropic.553.6 etc
        
        but it would be rather tedious to have to define all of
        that in a problem configuration.
        
        This method sort_non_spectral_model is only called
        if there is a y State (an observation) that is declared
        to be non spectral. This is switched on by using:
        
        general.is_spectral = False
        
        in the configuration file. 
        
        Note that not all state variables are 'made spectral' (i.e.
        transformed to Isotropic.465.6 Isotropic.553.6 etc.) ... it is
        only those associated with the x state of operators that contain
        both x and y states. So, for example, we might have an operator
        (e.g. a regularisation operator) with a parameter gamma_time.
        Since this operator contains only x state, there is no
        need to 'make spectral' the gamma_time state. If however we had
        a prior operator, which would have both x and y states, the
        declared x state would be 'made spectral'.
        
        '''
    from eoldas_Lib import fixXstate
    names = parameter.names
    if logger:            
        logger.info('Non spectral model declared (general.is_spectral)')
        logger.info('adjusting x state vector accordingly')
    # get the parameter names
    new_states = {}
    for (k,v) in ops.dict().iteritems():
        if type(v) == ParamStorage and 'y' in v.dict() and 'x' in v.dict():
            # get the band names
            bands = v.y.names
            pnames = v.x.names
            old_state = np.array(np.array(pnames).tolist()*len(bands)).flatten().tolist()
            new_state = np.array([[p+'.'+b for p in pnames] for b in bands]).\
                flatten().tolist()
            thisdict =  dict(zip(new_state,old_state))
            for (k,v) in thisdict.iteritems():
                new_states[k] = v
    
    # Now, wherever we see pnames in any x states
    # we have to replace by new_states
    if len(new_states) == 0:
        return
    # split the dict into 2 lists
    toList = []
    fromList = []
    for (k,v) in new_states.iteritems():
        fromList.append(v)
        toList.append(k)
    if logger:
        logger.debug('... %s'%'parameter')
    parameter.thisname = 'parameter'+'.x'
    fixXstate(parameter,fromList,toList,logger=logger)
    for (opname,op) in ops.dict().iteritems():
        if type(op) == ParamStorage:
            if logger:
                logger.debug('... %s'%opname)
            op.thisname = opname+'.x'
            fixXstate(op,fromList,toList,logger=logger)
    
def fixXstate(op,fromList,toList,logger=None):
    '''
        A method to extend the state vector definition by 
        e.g. making each state have a value per waveband. This is 
        useful for non spectral models (e.g. MODIS kernels)
        
        Inputs:
        op      : a ParamStorage containing 'x'
        fromList: what the states were previously called
        toList  : what we will call them now
        
        '''
    from eoldas_Lib import sortopt
    if not 'x' in op.dict():
        if logger:
            logger.info('No x state found in %s'%op.thisname)
        return
    if logger:
        logger.debug('Extending x state in %s ... '%op.thisname)
    # store the lists in case we need them later
    op.is_spectral = False
    op.x.fromList = fromList
    op.x.toList = True
    # The terms to examine:
    # op.names, op.solve
    # op.x.names, op.x.sd, op.x.state, op.x.bounds, op.x.default
    
    toList = np.array(toList)
    fromList = np.array(fromList)
    for opd in [op, op.x]: 
        if 'names' in opd.dict():
            names = np.array(opd.names)
            for i in ['names', 'solve', 'sd', 'state', 'bounds', 'default']:
                if i in opd.dict() and opd[i] != None:
                    old_values = [[k] for k in opd[i]]
                    new_values = old_values
                    for j in np.unique(fromList):
                        ww = np.where(names == j)[0]
                        www = np.where(fromList == j)[0]
                        if i == 'names':
                            new_values[ww] = np.array(toList[www]).tolist()   
                        else:
                            new_values[ww] = old_values[ww]*len(www)
                    opd[i] = np.array(reduce(lambda x,y:x+y,new_values))



            
    
def demonstration(conf='default.conf'):      
    '''
    Need to develop a new demo
    '''
    from eoldas_Lib import eoldas_setup
    from os import getcwd
    from eoldas_ParamStorage import ParamStorage

    options = ParamStorage()
    options.here = getcwd()
    options.logdir = 'logs'
    options.logfile = "logfile.log"
    options.datadir = ['.','~/.eoldas']
    #self = eoldas_setup('default.conf',options)            
    #return self        
    return True

if __name__ == "__main__":
    
    depreciated = [ObservationOperator,\
                   eoldas_setup]
    useful = [set_up_logfile, sortopt, get_filename, \
              set_default_limits,  check_limits_valid,\
              sort_non_spectral_model, fixXstate,\
              quantize_location, dequantize_location, get_col]
    
    from eoldas_Lib import *
    for i in useful:
        help(i)
    
    self = demonstration(conf='default.conf')


