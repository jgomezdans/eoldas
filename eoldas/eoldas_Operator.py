#!/usr/bin/env python
import pdb
import numpy as np
#from eoldas_model import subset_vector_matrix
from eoldas_State import State
from eoldas_Lib import sortopt,sortlog
from eoldas_ParamStorage import ParamStorage
import resource
from eoldas_Spectral import Spectral
from eoldas_Files import writer
import os
from os.path import dirname

class Operator ( ParamStorage ):
    """
        A generic class for the EOLDAS operator 
        """
    def __init__ ( self,options,general,parameter=None,\
                  datatype={'x':'x','y':'y'},\
                  state={'x':None,'y':None},\
                  sd={'x':None,'y':None},\
                  names={'x':None,'y':None},\
                  control={'x':None,'y':None},\
                  location={'x':None,'y':None},\
                  name=None,\
                  env=None,logger=None,\
                  logdir=None,logfile=None,datadir=None,doY=None):
        """
            
            This sets up the data for an Operator
            
            The main purpose is to load self.x and self.y
            with associated metadata in self.x_meta and self.y_meta
            
            The Operator is set up to calculate, among other things, as cost:
            
            J = 0.5 (y - H(x))^T (C_y^-1 + C_H(x)^-1) (y - H(x))
            
            where y might be an observation and H(x) is a modelled version 
            of that observation.
            
            For example, for a prior Operator, we want:
            
            J = 0.5 (x_prior - x)^T C_prior^-1 (x_prior - x)
            
            so when we are solving for this, we load candidate state estimates
            into x, with C_x^-1 = 0 with C_y^-1 = C_prior^-1 and prior terms 
            into y, with H(x) = I(x), and identity operation.
            
            For some model e.g. x_model = A + B x, we have:
            
            J = 0.5 (x_model - x)^T C_model^-1 (x_model - x)
            
            For a linear model, x_model = A + B x so we have:
            
            J = 0.5 (A + (B -I)x)^T C_model^-1 (A + (B - I)x)
            
            so we load the vector A into y, with C_y^-1 = 0 and H(x) = (B-I)x
            with C_x^-1 = C_model^-1
            
            For an observation operator, we want:
            
            J = 0.5 (y_obs - Y(x))^T C_obs^-1 (y_obs - Y(x))
            
            so we load the observations into y, 
            and set H(x) = Y(x), the observation operator.
            
            Note that we always load the new state extimate into the x term.
            
            To summarise these cases wrt initial setup:
            
            For data loading:
            
            For the prior example:       y = x_prior, x = x_new, H(x) = Ix
            For the model example:       y = A,       x = x_new, H(x) = (B-I)x 
            For the observation example: y = y_obs,   x = x_new, H(x) = Y(x)
            
            The cost function derivative, the Jacobian wrt each of the model 
            state vector elements is, ignoring derivatives of C_H(x)^-1:
            
            J' = dJ/dx = (y - H(x))^T (C_y^-1) H'(x)
            
            where H'(x) = dH(x)/dx
            
            The second order differential, the Hessian is a square matrix:
            
            J'' = d^2J/(dx_idx_j) = H'(x)^T (C_y^-1) H'(x) +
            (y - H(x))^T (C_y^-1) H''(x)
            
            or if we ignore H''(x), simply:
            
            J'' = d^2J/(dx_idx_j) = H'(x)^T (C_y^-1) H'(x)
            
            
            We assume that a config file or command line parsing
            has been done and the required information is set up in 
            options. This will include 
            
            x.names   :   state variable names
            x.state   :   state mean
            x.sd      :   state sd
            
            and may include:
            
            x.control     :   control variables, e.g.
            [mask,vza,vaa,sza,saa]
            x.location    :   location variables e.g. [time]
            solve       :   Codes associated with solver options for the
            model parameters.
            An integer array of the same length as 
            names
            
            as well as any other information that may be required,
            generally available through setup.
            
            The information in self.options is used to initialise the
            x data from self.options.xstate and the y data from 
            self.options.ystate
            
            The result of initialising the Operator then is to:
            -- start logging
            -- load any data into self.x.state and self.y.state
            and possibly self.x.sd, self.y.sd 
            from various formats that state can be.
            -- load self.options
            -- load self.general
            -- load location and control information
            self.x.location
            self.x.control (& y data)
            -- load associated metadata
            self.x_meta.control etc
            
            """
        if name == None:
           import time
           thistime = str(time.time())
           name = type(self).__name__
           name =  "%s.%s" % (name,thistime)
        self.thisname = name

        if type(datatype) != dict:
            # maybe its a list
            if type(datatype) == list:
                datatype = dict(np.repeat(np.array(datatype),2)\
                                .reshape((len(datatype),2)))
            elif type(datatype) == str:
                datatype = {datatype:datatype}
            else:
                return
        self.options = options
        self.general = general
        
        this = {'datadir':['.'],'env':None,'logfile':None,'logdir':None}
        for (k,v) in this.iteritems():
            self.general[k] = sortopt(self.general,k,v)
        
        env     = env     or self.general.env
        datadir = datadir or self.general.datadir
        logfile = logfile or self.general.logfile
        logdir  = logdir  or self.general.logdir
        bounds = 'False'
        self.logger = sortlog(self,logfile,logger,name=self.thisname,\
			 logdir=logdir,debug=True)        
        self.isLinear = False
        #  set up x & y states 
        thisname = name
        kk = datatype.keys()
        kk.sort()
        # need to make sure y values are before the x ones
        # so sort & reverse     
        for t in kk[::-1]:
            if parameter != None and t in parameter.dict():
                t_meta = "%s_meta"%t
                name = {t:thisname}
                this = {'datatype':t,'name':name,\
                    'state':parameter[t].state,'sd':parameter[t].sd,\
                    'control':parameter[t_meta].control,\
                    'location':parameter[t_meta].location,\
                    'bounds':parameter[t_meta].bounds,\
                    'names':parameter[t_meta].state}
            else:
                name = {t:thisname}
                this = {'datatype':t,'name':'Billy no mates %s'%t,\
                    'state':None,'sd':None,\
                    'control':None,'location':None,\
                    'bounds':'False',\
                    'names':['dummy']}
            # tt is what we will store it as e.g. x_state
            for (k,v) in this.iteritems():
                kk = "%s%s"%(t,k)
                # if k exists in parameter, use that as v
                if parameter != None and not t in parameter.dict() and\
                    k in parameter.options.dict():
                        v = parameter.options[k]
                self.options[t][k] = \
                    sortopt(self.options[t],k,v)
                self.options[t][k] = (t in eval(k) and eval(k)[t]) \
                    or self.options[t][k]
                # need to tmp load it directly to options 
                if type(self.options[t][k]).__name__ != 'NoneType':
                    self.options[k] = self.options[t][k]
            tt = "%s_state"%t
            self.options.name = "%s-%s"%(self.options.name,t)
            self.options.bounds = self.options[t].bounds
            # Set up State 
            self[tt] = State(options,logger=logger,\
                             datadir=datadir,logfile=logfile,\
                             logdir=logdir,env=env,\
                             name=self.options.name,grid=self.general.grid)
	    # a flag you can use to see if 'something' is loaded
	    self.isLoaded = False
            self.preload_prepare()
            # load any state data   
            has_state = False
            if (type(self.options[t]) == ParamStorage and \
                'state' in self.options[t].dict().keys()) or\
                (type(self.options[t]) == dict and \
                 state in self.options[t].keys()):
                self[tt].state = self.options[t].state
		try:
		    if self[tt].state.size == 1 and \
			self[tt].state == None or self[tt].state == np.array(None):
		    #raise Exception('Failed to recognise state data ... check the log')
		        has_state = False
		    else:
                        has_state = True
	        except:
		    if self[tt].state.size > 1:
			has_state = True
            if has_state == True and self[tt].state.size == 1 and \
                self[tt].state == np.array(None):
                    has_state = False
            
            from_parameter = False
            if parameter != None and t == 'x' and self[tt].state.size == \
                parameter[tt].state.size:
                # must have loaded from parameter
                # so load into self.x.state
                self.loader(parameter)
                self.x_state.state = self.x.state
                from_parameter = True
            
            if from_parameter == False and has_state == False:
                # then we'd better set up a grid from the defaults
                if not self[tt].options[t].apply_grid:
                    self[tt].options[t].apply_grid = True
                    State.regrid(self[tt])
                has_state = True
	   
	    if has_state: 
	        self.options[t].sd = sortopt(self.options[t],'sd',np.zeros_like(self[tt].Data.state))
                self[tt].Data.sd = sortopt(self[tt].Data,'sd',\
                            np.array([float(v) for v in self.options[t].sd]))
		
            self[tt].Name.is_grid = False
            if has_state and 'apply_grid' in self[tt].options[t].dict() and \
                self[tt].options[t].apply_grid:
                if 'gridder' in self[tt].Data.dict():
                    # if gridded, load the grid data
                    self[tt].Data.state = self[tt].Data.gridder.grid
                    self[tt].Data.sd = self[tt].Data.gridder.sdgrid
                    self[tt].Name.is_grid = True
                if from_parameter:
                    self[tt].Name.is_grid = True
            
            if from_parameter or not self[tt].Name.is_grid:
                # override with options data if there
                self[tt].Data.sd = \
                    sortopt(self.options[t],'sd',self[tt].Data.sd)
                # check the size of sd
                sd1 = np.array(self[tt].Data.sd)
                r = self[tt].Data.state.astype(float)
            # ensure sd is float 
            self[tt].Data.sd = np.array(self[tt].Data.sd)
            shape = self[tt].Data.sd.shape
            sd = np.zeros_like(self[tt].Data.sd).astype(float).flatten()
            for (i,v) in enumerate(self[tt].Data.sd.flatten()):
                sd[i] = float(v)
            self[tt].Data.sd = sd.reshape(shape)
            sd1 = np.array(self[tt].Data.sd)
            
            if has_state and (not self[tt].Name.is_grid or from_parameter)\
                and sd1.size != r.size:
                    # try to match up sd with data shape & size
                    if not self[tt].Name.is_grid:
                        (n_samples,n_states) = r.shape
                    else:
                        n_states = r.shape[-1]
                        n_samples = r.size/n_states
                    
                    if parameter:
                        n = parameter.options["%ssd"%t].size
                    else:
                        n = r[:,0].size
                    if sd1.size == 1: 
                        self[tt].Data.sd = np.zeros_like(r) + sd1.flatten()[0]
                    elif n != n_states:
                        if 'logger' in self.dict():
                            self.logger.info("Warning: some isses with state" + \
                                 " and sd data. They dont look the same size " +\
                                "(%d vs %d) ... tiling accordingly"%(n,n_states))
                        self[tt].Data.sd = \
                            np.tile(np.array(sd1),n_samples)\
                                .reshape((n_samples,sd1.shape[0]))
                        if from_parameter:
                            self[tt].Data.sd = self[tt].Data.sd.reshape(r.shape)
            # sort inverse correlation matrix if its from sd
            
            if has_state:
                self[tt].Data.C1 = self[tt].Data.sd*0.
                ww = np.where(self[tt].Data.sd >0)
                self[tt].Data.C1[ww] = \
                    1./(self[tt].Data.sd[ww] * self[tt].Data.sd[ww])
            
            self[t]           = self[tt].Data   #  .to_dict()
            self["%s_meta"%t] = self[tt].Name   #  .to_dict()
        # If there is a y datatype, load spectral info
        
        kk = datatype.keys()
        kk.sort()
        for t in kk[::-1]:
            if t == 'y':
                self.y_meta.spectral = Spectral(self,name=self.thisname)                
	self.linear = ParamStorage()
        self.linear = ParamStorage()
        if 'y' in self.dict():
            self.linear.H = np.zeros(self.y.state.shape)
        else:
            self.linear.H = np.zeros(self.x.state.shape)

	self.epsilon = sortopt(self.general,'epsilon',0.01)
	# a hook to allow users not to have to write a whole new __init__
	self.J_prime_approx = None
	self.doPlot = sortopt(self.general,'doplot',False)
	self.plotMod = sortopt(self.general,'plotmod',10)
	self.plotMovie = sortopt(self.general,'plotmovie',False)
	self.showPlot = sortopt(self.general,'showplot',False)
        self.plotMovieFrame = 0
        self.postload_prepare()
	# check to see if there is anything worth plotting
	# in this operator
	# this is determined by whether a filename is 
	# specified for output
	try:
	    self.plot_filename = \
			sortopt(self.y_state.options.y.result,\
				'filename',self.thisname) + '.plot'
	except:
	    try:
		self.plot_filename = \
			sortopt(self.x_state.options.x.result,\
				'filename',self.thisname) + '.plot'
	    except:
		try:
		    self.plot_filename = \
			sortopt(self.options.result,\
				'filename',self.thisname) + '.plot'
		except:
		    self.doPlot = False
	if 'plot_filename' in self.dict():
  	    outdir = os.path.dirname(self.plot_filename)
	    try:
	        os.makedirs(outdir)
	    except:
	        pass
	self.Hcount = 0

        return
    
    def postload_prepare(self):
        '''
            A method that gets accessed after data
            are loaded. You should think of this as a hook
	    so that you don't have to write a whole new __init__ 
	    method.

	    In this default method, we use it to declare that
	    the operator is linear (self.isLinear = True)
	    and also to set up H_prime for an identity operator.

            '''
        self.isLinear = True   
	try:
	    self.linear.H_prime = np.eye(self.y.state.shape[0])
        except:
	    self.linear.H_prime = np.eye(self.x.state.shape[0]) 

    def preload_prepare(self):
        '''
            A method that gets accessed before data
            are loaded. You should think of this as a hook
            so that you don't have to write a whole new __init__ 
            method.

            '''
	# no approximate method for J_prime is appropriate here
	# as J_prime is trivial for an identity operator
        self.J_prime_approx = None
    
    def H(self,x):
        '''
            The default operator: an identity operator
            
            This should return self.linear.H with dimensions the same
            as y if there is a y state. Otherwise, it is assumed of size x.
            '''
        self.linear.H = x
        return self.linear.H
   
    def H_prime(self,x):
        '''
            The default differential operator
            
            dH(x)/dx
            
            Here, we return a full matrix but that is not always needed
            as it can be large for large problems
            '''
	if self.isLinear and 'H_prime' in self.linear.dict():
	    return self.linear.H_prime
        xfull,Cx1,xshape,y,Cy1,yshape = self.getxy()
        xshape = np.array(x.shape)
        xnparam = x.size/xshape.prod()
	ynparam = y.size/yshape.prod()
        xx = x.reshape((xshape.prod(),xnparam))
        out = np.zeros((np.array([xshape.prod(),xnparam,\
                                  yshape.prod(),ynparam])))
        bounds = self.options.x.bounds
        self.Hx = self.linear.H
        for i in xrange(xnparam):
            x1 = xx.copy()
            delta = self.delta[:,i]
            if delta != 0:
                x1[:,i] += delta
                ww = np.where(x1[:,i] > self.xmax[:,i])[0]
                x1[ww,i] = xx[ww,i] - delta
                H1x = self.H(x1.flatten())
                d = (H1x - self.Hx).reshape((yshape.prod(),ynparam))/delta
                d[ww,i] = -d[ww,i]
                ww = np.where(d[:,i] != 0)[0]
                for j in ww:
                    out[j,i,j,i] = d[j,i]
        self.linear.H_prime = out.reshape([xshape.prod()*xnparam,\
                                           yshape.prod()*ynparam])
        # check to see if we can simplfy
        #if xshape.prod()*xnparam == yshape.prod()*ynparam:
        #    dd = self.linear.H_prime.diagonal()
        #    out = self.linear.H_prime.reshape(xshape.prod()*xnparam*\
        #                                      yshape.prod()*ynparam)
        #   if out.sum() == dd.sum():
                # its diagonal
        #        self.linear.H_prime = dd
        return self.linear.H_prime
    
    def unloader(self,smallx,fullx,all=False,sum=False,M=False):
        '''
            Utility to return a full state vector representation 
            from one that may be only partial
            
            If sum is True, then if multiple items appear per location
            they are summed. That is a bit fiddly, but is needed
            e.g. for loading derivative data.
            
            If sum is False, then only the first instance we come across 
            is loaded.
            
            If M is True, then we are unloading a matrix
            
            '''
	#if self.thisname == 'eoldas.solver.eoldas.solver-obs': # and self.x.state[0].sum() == 1:
        if 'loaderMask' in self.dict():
            if M == True:
                MM = self.loaderMask.flatten()
                n = fullx.flatten().shape[0]
                out = np.zeros((n,n))
                ww = np.where(MM)[0]
                count = 0
                for i in ww:
                    this = self.unloader(smallx[count],fullx,sum=sum)
                    out[i,:] = this.flatten()
                    count += 1
                return out
            out = fullx*0.
            if 'xlookup' in self.dict():
                stmp = smallx.reshape(self.x.state.shape)
                if sum:
                    # This kept me puzzled for ages ...
                    # Its when when have multiple terms
                    # per observation period
                    # Lewis, UCL 23 Aug 2011
                    stmpt = stmp.T
                    # this reduces smallx to the size of the y state
                    smallx = stmp[self.xunlookup].T
                    for i in xrange(len(self.xunlookup)):
                        for j in xrange(1,len(self.xlookup[i])):
                            smallx[...,i] = smallx[...,i] + \
                                stmpt[...,self.xlookup[i][j]]
                    smallx = smallx.T
                else:
                    # load the first one
                    smallx = stmp[self.xunlookup]
            # then load others
            out[self.loaderMask] = smallx.flatten()
            return out
        raise Exception(\
                'Illegal call to unloader before loader has been called')
    
    def loader(self,fullx):
        '''
            Utility to load the state vector required here 
            (self.input_x) from fullx
            
            This is done using a mask, in loader_prep which can be 
            used in reverse in the unloader. The mask is stored
            as self.loaderMask
            
            '''
	from eoldas_Lib import isfloat
        if 'loaderMask' in self.dict():
            state = (fullx.x.state[self.loaderMask])\
                .reshape(self.unloaderMaskShape)
            if 'xlookup' in self.dict():
                for i in xrange(len(self.xlookup)):
                    self.x.state[self.xlookup[i]] = state[i]
            else:
                self.x.state = state
            return True
        if not 'x_meta' in self.dict():
            self.x_meta = ParamStorage()
            self.x_meta.state = self.x_state._state.name.state
            self.x_meta.location = self.x_state._state.name.location
            try:
                self.x_meta.is_grid = self.options.x.apply_grid
            except:
                self.x_meta.is_grid = self.options.x.apply_grid = False
            if not 'x' in self.dict():
                self.x = ParamStorage()
        to_names = self.x_meta.state
        from_names = fullx.x_meta.state
        self.x_state.logger.info("loading from %s"%str(from_names))
        self.x_state.logger.info("loading to   %s"%str(to_names))
        to_loc = self.x_meta.location
        from_loc = fullx.x_meta.location
        n_param = len(to_names)
        # the loader mask is the size of the input data
        # 
        self.loaderMask = np.zeros_like(fullx.x.state).astype(bool)
	# this is used in case a full size array is passed
	self.ploaderMask = np.in1d(fullx.x_meta.state,self.x_meta.state)
        if not (from_loc == to_loc).all():
            # cant load as location bases not the same 
            self.logger.error("Cannot load state vector as location bases are different")
            return False
        if self.x_meta.is_grid and fullx.x_meta.is_grid:
            # a straight load but may be different parameters
            if np.in1d(self.x_meta.state,fullx.x_meta.state).all() and \
                len(self.x_meta.state) == len(fullx.x_meta.state):
                self.x.state = fullx.x.state
                # full mask
                self.unloaderMaskShape = fullx.x.state.shape
                self.loaderMask = self.loaderMask + True
            else: # downsizing the gridded dataset
                xshape = np.array(fullx.x.state.shape)
                xshape[-1] = self.x_meta.state.size
                xshape = tuple(xshape)
                self.x.state = np.zeros(xshape)
                self.unloaderMaskShape = xshape
                for i in xrange(self.x_meta.state.size):
                    ww = np.where(self.x_meta.state[i] == fullx.x_meta.state)[0]
                    self.loaderMask[...,ww] = True
                    self.x.state[...,i] = fullx.x.state[...,ww].reshape(\
                                                    self.x.state[...,i].shape)
	    where_grid_data = tuple(np.array(self.loaderMask.shape)[:-1] *0)
	    if self.loaderMask[where_grid_data].sum() == 0 :
	        self.novar = True
	        self.logger.info('There appears to be no valid data ... ignoring this Operator')
                return False

        elif fullx.x_meta.is_grid:
            # so we are coming from a grid to a non grid
            # which is the most likely case
            # we only want to load those samples where
            # location matches
            # now set to_loc to the actual location data
            nd = len(from_loc)
            # we need to get the qlocation info from somewhere
            if not 'qlocation' in self.x and \
                not 'qlocation' in self.x_state:
                try:
                    for i in ['qlocation','location']:
                        self.x[i] = self.y[i]
                        self.x_meta[i] = self.y_meta[i]
                except:
                    self.logger.error('Illegal x state definition')
                    raise Exception('Illegal x state definition')
            
            try:
                to_loc = self.x.qlocation
                    #- fullx.x_meta.gridder.grid[0:nd]
            except:
                to_loc = self.x_state.qlocation 
                    #- fullx.x_meta.gridder.grid[0:nd]
                self.x.qlocation = self.x_state.qlocation
            
            try:
                self.unloaderMaskShape = self.x.state.shape
            except:
                # x state is not yet set, so we must work out itsd
                # shape by other means
                self.unloaderMaskShape = (len(self.x.qlocation)\
                                          ,len(self.x_meta.state))
                self.x.state = np.zeros(self.unloaderMaskShape)
            
            mask = np.zeros_like(self.x.state).astype(bool)
            self.x.fullstate = self.x.state
            for i in xrange(len(to_loc)):
                for j in xrange(n_param):
                    n = np.where(to_names[j] == from_names)[0][0]
                    tup = tuple(np.append(to_loc[i],n))
                    #tuple(np.concatenate([to_loc[i],[n]]))
                    if not self.loaderMask[tup]:
                        self.x.state[i,j] = fullx.x.state[tup]
                        mask[i,j] = True
                        self.loaderMask[tup] = True
            
            m = self.x.state[mask]
            state = m.reshape(m.size/n_param,n_param)
            
            lmask = np.array([mask[:,0].tolist()]*self.x.location.shape[1]).T
            m = self.x.location[lmask]
            location = m.reshape(m.size/self.x.location.shape[1],\
                                 self.x.location.shape[1])
            m = self.x.qlocation[lmask]
            qlocation = m.reshape(m.size/self.x.location.shape[1],\
                                  self.x.location.shape[1])
            
            # The length of the x vector is the same as the y
            # but y may contain multiple observations
            # for a given location
            # So, we form a list xlookup that contains the 
            # indices of the multiple y terms for x
            # If there is no y vector then they have one element
            # each, but more generally they will have
            # multiple y terms
            xlookup = [[] for i in xrange((m.size/self.x.location.shape[1]))]
            for i in xrange(len(xlookup)):
                thisloc = str(qlocation[i,:])
                for j in np.where(self.x.qlocation[:,0] \
                                  == qlocation[i,0])[0]:
                    if str(self.x.qlocation[j]) == thisloc:
                        xlookup[i].append(j)
            
            self.xlookup = xlookup
            # xunlookup is the opposite of xlookup but only
            # gives the index of the first element on the list
            # for each location
            self.xunlookup = np.array([i[0] for i in self.xlookup])
            # now expand state to self.x.state
            for i in xrange(len(self.xlookup)):
                self.xlookup[i] = np.array(self.xlookup[i])
                self.x.state[self.xlookup[i]] = state[i]
            self.unloaderMaskShape = state.shape

        # make sure we store some bounds info in a convenient manner
        # required eg for finite differerence approximations
        self.epsilon = sortopt(self.general,'epsilon',1e-5)
        try:
            bounds = self.x_meta.bounds
        except:
            try:
                bounds = self.options.bounds
            except:
                raise Exception('Bounds information not defined')
	if self.x_meta.is_grid and fullx.x_meta.is_grid:
	    ww = where_grid_data
	    self.logger.debug('Coming from a gridded dataset to a gridded dataset')
	elif fullx.x_meta.is_grid:
	    self.logger.debug('Coming from a gridded dataset to a non gridded dataset')
	    ww = tuple(to_loc[0])
	else:
	    self.logger.debug('Coming from a non gridded dataset to a non gridded dataset')
	    # it should be an int or float
	    ww = to_loc[0]
	    try:
	        ok,dummy = isfloat(ww)
	        if not ok:
	            # but it might not be for some strange reason
		    # shouldnt be here, but give it a go anyway
	            self.logger.error('Having trouble interpreting dataset')
	            self.logger.error(' ... have ended up somewhere unexpected, but will keep trying')
	            ww = tuple([0]*len(to_loc))
	    except:
		ww = tuple(ww*0)
		self.logger.error('Issues with this dataset for this operator, but will keep trying')
		ww = tuple([0]*len(to_loc))
	if 'x_state' in self.dict():
            # are there any data here?
	    nn = np.array(self.loaderMask[ww]).sum()
	    if nn == 0:
	        self.novar = True
	        self.logger.info('There appears to be no valid data ... ignoring this Operator')
		return False

            dd = self.epsilon*(bounds[:,1]-\
                bounds[:,0])
            if dd.size == self.loaderMask[ww].size:
                self.delta = dd[self.loaderMask[ww]]
                self.xmin = bounds[:,0][self.loaderMask[ww]]
                self.xmax = bounds[:,1][self.loaderMask[ww]]
            elif dd.size == self.loaderMask.size:
                self.delta = dd[self.loaderMask]
                self.xmin = bounds[:,0][self.loaderMask]
                self.xmax = bounds[:,1][self.loaderMask]
	        
	    if self.delta.size == 0:
		# no data here ..?
		self.novar = True
	 	self.logger.info('There appears to be no valid data ... ignoring this Operator')
		self.xranger = self.delta
		return False
            if self.delta.size == self.x.state.shape[-1]:
	        ww = tuple(np.array(self.x.state.shape)[:-1])
                self.delta = np.tile(self.delta,ww).reshape(self.x.state.shape)
                self.xmin = np.tile(self.xmin,ww).reshape(self.x.state.shape)
                self.xmax = np.tile(self.xmax,ww).reshape(self.x.state.shape)
	    try:
                self.xranger = self.xmax[0] - self.xmin[0]            
	    except:
		raise Exception('An error occurred setting eoldas bounds in the Operator')
        return True

    def H_prime_prime(self,x):
        '''
            Not yet implemented
            
            d^2H(x)/dxi dxj
            '''
        if isLinear and 'linear' in self.dict(): 
            return self.linear.H_prime_prime
        elif not 'linear' in self.dict():
            self.linear = ParamStorage()
        return None   

    def getxy(self,diag=False):
        
        xshape = self.x.state.shape
        x = self.x.state.astype(float).flatten()
        if 'x' in self.dict():
            Cx1 = self.x.C1.astype(float)
        else:
            Cx1 = 0.*x
          
        if 'y' in self.dict():
            y = self.y.state.astype(float).flatten()
            yshape = self.y.state.shape
            Cy1 = self.y.C1.astype(float)
        else:
            y = 0.*x
            Cy1 = Cx1
            yshape = xshape
	if diag:
	    # if diag flag is set, return sd, not C^-1
            try:
	    	Cx1 = np.diag(Cx1)
	    	Cx1 = 1./np.sqrt(Cx1)
	    except:
	    	pass
	    try:
            	Cy1 = np.diag(Cy1)
            	Cy1 = 1./np.sqrt(Cy1)
            except:
            	pass
	    if Cx1.size < x.size:
		Cx1 = np.tile(Cx1,yshape[0])
	    if Cy1.size < x.size:
		Cy1 = np.tile(Cy1,yshape[0])
	else:
	    Cx1 = Cx1.flatten()
	    Cy1 = Cy1.flatten()
        return x,Cx1,xshape,y,Cy1,yshape

    memory = lambda self : self.logger.info("Memory: %.3f GB"%\
                                ((resource.getrusage(0).ru_maxrss*1./\
                                  float(resource.getpagesize()))/(1024**3)))

    def write(self,filename,dataset,fmt='pickle'):
        '''
        A write method for outputting state variables 
            
        e.g. self.write('xxx.dat','x',fmt='PARAMETERS')
       
	Will also do basic plotting of state and observations
	if possible.
     
        '''
	try:
            op.plot(ploterr=True)
        except:
            # never mind
            pass
        state = '%s_state'%dataset
        if not state in self.dict():
            self.logger.error(\
                    'Cannot write state %s ... not in operator'%dataset)
            return
        writer(self[state],filename,dataset,fmt=fmt)

    def cost(self):
        '''
            If the Operator is a super operator
            it may contain other operators in self.operators
            in which case, cost() returns J and J_prime
            for all sub operators
            
            If not, then just calculate the cost in this operator.
            Note that in the case of a super operator, we do not
            count self cost.
            
            '''
        #self.memory()
        if not 'operators' in self.dict():
            op = self
            op.loader(self)
            J,J_prime_tmp = op.J_prime()
            J_prime = op.unloader(J_prime_tmp,self.x.state,sum=True)
            op.x_state.logger.info("J = %f"%Jtmp)
            #op.x_state.logger.info("J' = %s"%str(J_prime))
        
        for i,op in enumerate(self.operators):
            # load from self.x.state, the *full* representation
            # into that required by this operator
            op.loader(self)
            Jtmp,J_prime_tmp = op.J_prime()
            J_prime_tmp = op.unloader(J_prime_tmp,self.x.state,sum=True)
            #self.logger.info('operator %s'%op.options.thisname)
	    #self.logger.info("%d     J   = %f"%(i,Jtmp))
	    #self.logger.info("%d max |J'| = %s"%(i,str(\
            #                            np.max(np.abs(J_prime_tmp),axis=0))))

            op.x_state.logger.info("     J   = %f"%Jtmp)
            if 'debug' in self.dict() and self.debug > 1:
		op.x_state.logger.info("     J'  = %s"%str(J_prime_tmp))
            	op.x_state.logger.info("     x   = %s"%str(self.x.state))
            if op.doPlot and op.Hcount%op.plotMod == 0:
	        try:
                    op.plot()
		except:
		    # never mind but inform the user
		    op.logger.info("... didn't make it")
            op.Hcount += 1

            if i == 0:
                J = Jtmp
                J_prime = J_prime_tmp
            else:
                J += Jtmp
                J_prime += J_prime_tmp
        #self.memory()
        if self.doPlot and self.Hcount%self.plotMod == 0:
            try:
                self.plot(noy=True)
            except:
                # never mind but inform the user
                self.logger.info("... didn't make it all")
	self.Hcount += 1

        return J,J_prime.flatten()

    def invtrans(self,x,lim=[]):
	'''
	Apply the inverse transform to the state vector
	if it is defined
	'''
	# lots of reasons not to do this ...
	xshape = x.shape
	x = x.flatten()
	if lim != []:
	    lu = lim[1].flatten()
	    ll = lim[0].flatten()
	    ww = np.where(x>lu)
	    x[ww] = lu[ww]
	    ww = np.where(x<ll)
	    x[ww] = ll[ww]
	try:
	    invtransform = self.invtransform
	    if invtransform == None:
		return x.reshape(xshape)
	except:
	    return x.reshape(xshape)
	xout = x.copy().reshape(xshape)
	thisinvtransform = self.invtransform
	todo = np.in1d(thisinvtransform,self.x_meta.state)
	for i in np.where(~todo)[0]:
	     this = thisinvtransform[i].replace(self.x_meta.state[i],'xout[...,i]')
	     xout[...,i] = eval(this)
	return xout

    def plot(self,noy=False,ploterr=False):   
        '''
	Form a plot of the  x & y state
        '''
        import pylab
	if not 'y_state' in self.dict() and noy == False:
	    self.logger.error('Failed attempt to plot y data: non existent y state')
	    return
        x1,Cx1,xshape,y,Cy1,yshape = self.getxy(diag=True)
        # try various places to bet bounds info
	
        try:
            x_min = self.xmin[0]
            x_max = self.xmax[0]
        except:
            x_max = np.zeros(xshape[-1])
            x_min = np.zeros(xshape[-1])
            try:
                for i in xrange(xshape[-1]):
                    x_max[i] = self.x_meta.bounds[i,1]
                    x_min[i] = self.x_meta.bounds[i,0]
            except:
                for i in xrange(xshape[-1]):
                    x_max[i] = max(x[...,i])
                    x_min[i] = max(x[...,i])

	lu = np.tile(x_max,xshape[:-1])
        ll = np.tile(x_min,xshape[:-1])
        lim = (ll,lu)
	# are any transforms required ?
	x = self.invtrans(x1.reshape(xshape),lim=lim).flatten()
	if ploterr:
            dx1 = self.invtrans((x1+Cx1*1.96).reshape(xshape),lim=lim) 
	    dx2 = self.invtrans((x1-Cx1*1.96).reshape(xshape),lim=lim)
	    # dx1 should be upper bound of x
            tmp = dx1.copy()
	    ww = np.where(dx2 > dx1)
	    tmp[ww] = dx2[ww]
	    dx2[ww] = dx1[ww]
	    dx1 = tmp
	    dx1 = (dx1).reshape(xshape)
	    dx2 = (dx2).reshape(xshape)
	    dy1 = (y + Cy1*1.96).reshape(yshape)
            dy2 = (y - Cy1*1.96).reshape(yshape)
	self.logger.info('starting plots ...')
	pylab.rcParams.update(\
		{'axes.labelsize':5,\
		 'text.fontsize': 6,
		 'xtick.labelsize': 6,\
		 'ytick.labelsize': 6})
	fig = pylab.figure(1)
        axisNum = 0
        pylab.clf()
	x = x.reshape(xshape)
	y = y.reshape(yshape)
        label = self.x_meta.location[0]
        isgrid = True
        try:
            (locations,qlocations,state,sd) = \
                                self.x_state.ungrid(x,Cy1)
            location = locations
	except:
            isgrid = False
            try:
                location = self.x.location[:,0]
            except:
                try:
                    if self.y.location[:,0].size == x[:,0].size:
                        location = self.y.location[:,0]
                    else:
                        label = label + ' index'
                        location = \
                                np.arange(x[:,0].size)\
                                *self.x_meta.qlocation[0][2]\
                                + self.x_meta.qlocation[0][0]
                except:
                    label = self.x_meta.location[0] + ' index'
                    location = np.arange(x[:,0].size)\
                                *self.x_meta.qlocation[0][2]\
                                + self.x_meta.qlocation[0][0]

        x_min = self.invtrans(x_min)
        x_max = self.invtrans(x_max)
        tmp = x_max.copy()
        ww = np.where(x_min > x_max)
        tmp[ww] = x_min[ww]
        x_min[ww] = x_max[ww]
        x_max = tmp

        nn = xshape[-1]
	for i in xrange(xshape[-1]):    
	    axisNum += 1
            ax = pylab.subplot(nn, 1, axisNum)
            maxer = x_max[i]
            if x_max[i] == x_min[i]:
                maxer = x_max[i] + 0.001
            #ax.set_ylim(x_min[i],maxer)
	    #try:
            #	ax.set_xlim(np.min(location),np.max(location))
	    #except:
		#pass
	    ax2 = ax.twinx() 
	    yprops = dict(rotation=90,\
			horizontalalignment='right',\
			verticalalignment='center')
	    ax2.set_ylabel(self.x_meta.state[i],**yprops)
	    # some transform bug in plotting error bars 
	    # leave for now
	    if isgrid or True or x.shape[0] > 100:
               	try:
		    if ploterr:
		        ax.fill_between(location,y1=dx2[:,i],y2=dx1[:,i],facecolor='0.8')
                    ax.plot(location,x[:,i],'r')
	        except:
		    xx = np.arange(xshape[:-1]).flatten()
		    if ploterr:
		        ax.fill_between(xx,y1=dx2[:,i],y2=dx1[:,i],facecolor='0.8')
	            ax.plot(xx,x[:,i],'r')
	    else:
	 	if ploterr:
		    dx1 = dx1 - x
		    dx2 = x - dx2
		try:
		    if ploterr:
		        ax.errorbar(location, x[:,i], yerr=[dx2[:,i],dx1[:,i]], fmt='ro')
                    ax.plot(location,x[:,i],'ro')
                except:
		    xx = np.arange(xshape[:-1]).flatten()
		    if ploterr:
                        ax.errorbar(xx,x[:,i], yerr=[dx2[:,i],dx1[:,i]], fmt='ro')
		    ax.plot(xx,x[:,i],'ro')
	    #ax2 = ax.twinx()	
	    # set nice ticks
	    ax.yaxis.set_major_locator(pylab.MaxNLocator(3))
	    #ax2.yaxis.set_major_locator(pylab.MaxNLocator(2))
            ax2.set_yticks(ax.get_yticks())
	    ax2.set_yticklabels([])
	    pylab.xticks()
	    ax.xaxis.set_major_locator(pylab.MaxNLocator(9))
	    if i == xshape[-1]-1:
	        ax.set_xlabel(label)
	    else:
		ax.set_xticklabels([])
	    #for lab in ax.get_xticklabels():
	    #    lab.set_rotation(0)
	    #ax.set_ylim(ax2.get_ylim())
        if self.showPlot:
	    pylab.show()
        else:
	    if self.plotMovie and 'plot_filename' in self.dict():
                fig.savefig(self.plot_filename + '.%08d.x.png'%self.plotMovieFrame)
            fig.savefig(self.plot_filename + '.x.png')

	if noy ==  True:
	    if not self.doPlot:
	        if self.plotMovie:
                    self.logger.info("... written plots to " + \
                        self.plot_filename + ".%08d.x.png"%self.plotMovieFrame)
		    self.plotMovieFrame += 1
	        self.logger.info("... written plots to " + \
		    self.plot_filename + ".x.png")
	    return
        # plot the other way around?
        axisNum = 0
        nn1 = int(np.sqrt(yshape[1]))
        nn2 = yshape[1]/nn1 + 1
        fig = pylab.figure(3)
        pylab.clf()

	# fix issues encounterd with 1D arrays
	Hx = np.matrix(self.Hx.reshape(self.y.state.shape))
        hshape = np.array(Hx.shape)
        if hshape[1] == yshape[0] and hshape[0] == yshape[1]:
	    Hx = Hx.T
	Hx = np.array(Hx)
	# y plotting
	# This will fail if there is no Hx, so wrap try around
        ymax1 = np.max(y)
	ymax2 = np.max(Hx)
	try:
            mask = self.y.control[:,self.y_meta.control=='mask'].flatten()
        except:
            mask = np.ones(yshape[0])
	if len(mask) < yshape[0]:
	    mask = np.ones(yshape[0])
	# a spectral plot
	for i in xrange(yshape[1]):
	    axisNum += 1
	    ymax1 = np.max(y[:,i])
	    ymax2 = np.max(Hx[:,i])
	    ax = pylab.subplot(nn1,nn2, axisNum)
	    #ax.set_ylim(0.,np.max([ymax1,ymax2]))
            try:
                ax.set_xlim(np.min(location),np.max(location))
            except:
                pass

            ax2 = ax.twinx()
            yprops = dict(rotation=90,\
                        horizontalalignment='right',\
                        verticalalignment='center')
            ax2.set_ylabel(self.y_meta.state[i],**yprops)

            # some transform bug in plotting error bars 
            # leave for now
            if isgrid or True or y.shape[0] > 100:
                try:
                    if ploterr:
                        ax.fill_between(location,y1=dy2[:,i],y2=dy1[:,i],facecolor='0.8')
                    ax.plot(location,Hx[:,i],'r')
                    ax.plot(location,y[:,i],'g,')
                except:
                    xx = np.arange(yshape[:-1]).flatten()
                    if ploterr:
                        ax.fill_between(xx,y1=dy2[:,i],y2=dy1[:,i],facecolor='0.8')
                    ax.plot(xx,Hx[:,i],'r')
                    ax.plot(xx,y[:,i],'g.')
            else:
                if ploterr:
                    dy1 = dy1 - y
                    dy2 = y - dy2
                try:
                    if ploterr:
                        ax.errorbar(location, y[:,i], yerr=[dy2[:,i],dy1[:,i]], fmt='ro')
                    ax.plot(location,Hx[:,i],'r')
                    ax.plot(location,y[:,i],'g.')
                except:
                    xx = np.arange(yshape[:-1]).flatten()
                    if ploterr:
                        ax.errorbar(xx,y[:,i], yerr=[dy2[:,i],dy1[:,i]], fmt='ro')
                    ax.plot(xx,Hx[:,i],'r')
                    ax.plot(xx,y[:,i],'g.')
            #ax2 = ax.twinx()
            # set nice ticks
            #ax2.set_yticks(ax.get_yticks())
            ax.yaxis.set_major_locator(pylab.MaxNLocator(3))
            #ax2.yaxis.set_major_locator(pylab.MaxNLocator(2))
            #ax.set_yticklabels([])
            pylab.xticks()
            ax.xaxis.set_major_locator(pylab.MaxNLocator(9))
            if i == xshape[-1]-1:
                ax.set_xlabel(label)
            else:
                ax.set_xticklabels([])
            #ax.set_ylim(ax2.get_ylim())
            for lab in ax.get_xticklabels():
                lab.set_rotation(0)
	if self.showPlot:
            pylab.show() 
	else:
	    fig.savefig(self.plot_filename + '.y2.png')
            if self.plotMovie:
                fig.savefig(self.plot_filename + '.%08d.y2.png'%self.plotMovieFrame)
        

        nn1 = int(np.sqrt(yshape[0]))
        nn2 = yshape[0]/nn1 + 1
        fig = pylab.figure(2)
        pylab.clf()
        axisNum = 0
        ymax1 = np.max(y)
        ymax2 = np.max(Hx)

	for i in xrange(yshape[0]):
                axisNum += 1
                ax = pylab.subplot(nn1,nn2, axisNum)
                ax.set_ylim(0.,np.max([ymax1,ymax2]))
                try:
                    wl = self.y_meta.spectral.nlw[self.y_meta.spectral.median_bands]
                except:
                    wl = np.arange(yshape[1])
                pylab.plot(wl,y[i])
                pylab.plot(wl,Hx[i],'g^')
                pylab.yticks()
                ax.yaxis.set_major_locator(pylab.MaxNLocator(3))
                if not (axisNum-1)%nn2 == 0:
                    ax.set_yticklabels([])
                pylab.xticks()
                ax.xaxis.set_major_locator(pylab.MaxNLocator(3))
                if (axisNum-1)/nn2 == nn1-1:
                    for label in ax.get_xticklabels():
                        label.set_rotation(90)
                else:
                    ax.set_xticklabels([])

	if self.showPlot:
	    pylab.show()
        else:
            fig.savefig(self.plot_filename + '.y.png')
            if self.plotMovie:
                fig.savefig(self.plot_filename + '.%08d.y.png'%self.plotMovieFrame)

            if self.plotMovie:
                self.logger.info("... written plots to " + \
                        self.plot_filename + ".%08d.y.png"%self.plotMovieFrame + " and " +\
                        self.plot_filename + ".%08d.y2.png"%self.plotMovieFrame + " and " +\
                        self.plot_filename + ".%08d.x.png"%self.plotMovieFrame)
            self.logger.info("... written plots to " + \
                        self.plot_filename + ".y.png" + " and " +\
                        self.plot_filename + ".y2.png" + " and " +\
                        self.plot_filename + ".x.png")

	if self.plotMovie:
	     self.plotMovieFrame += 1 
 
    def hessian(self):
        '''
            If the Operator is a super operator
            it may contain other operators in self.operators
            in which case, cost() returns J_prime_prime
            summed for all sub operators
            
            If not, then just calculate the J_prime_prime in this operator.
            Note that in the case of a super operator, we do not
            count self cost.

        '''
        self.x_state.logger.info('Calculating Hessian')
        if not 'operators' in self.dict():
            op = self
            op.loader(self)
            J,J_prime_tmp,J_prime_prime_tmp = \
                self.J_prime_prime ()
            J_prime = op.unloader(J_prime_tmp,self.x.state,sum=True)
        
            J_prime_prime = op.unloader(J_prime_prime_tmp,\
                                        self.x.state,sum=True,M=True)
            op.x_state.logger.info("J = %f"%Jtmp)
        #op.x_state.logger.info("J' = %s"%str(\
        #        J_prime[tuple(np.array(J_prime_tmp.shape[:-1])*0)]))
        
        for i,op in enumerate(self.operators):
            # load from self.x.state, the *full* representation
            # into that required by this operator
            op.loader(self)
            Jtmp,J_prime_tmp,J_prime_prime_tmp = \
                op.J_prime_prime ()
            J_prime_tmp = op.unloader(J_prime_tmp,self.x.state,sum=True)
            J_prime_prime_tmp = op.unloader(J_prime_prime_tmp,\
                            self.x.state,sum=True,M=True)
            #J_prime_prime_tmp = J_prime_prime_tmp
            op.x_state.logger.info("     J   = %f"%Jtmp)
            if i == 0:
                J = Jtmp
                J_prime = J_prime_tmp
                J_prime_prime = J_prime_prime_tmp
            else:
                J += Jtmp
                J_prime += J_prime_tmp
                J_prime_prime += J_prime_prime_tmp
        #self.memory()
        return J,J_prime.flatten(),J_prime_prime

    def J ( self ):
        """
            The operator contribution to the cost function:
            
            J = 0.5 (y - H(x))^T (C_y^-1 + C_H(x)^-1) (y - H(x))
            
            This is a single value, J
            """
        x,Cx1,xshape,y,Cy1,yshape = self.getxy()
        self.Hx = self.H(x).flatten()
        # NB, Hx should be same shape as y
        # or y is a constant
        d1 = np.array((y - self.Hx))
        self.linear.d1 = np.array((y - self.Hx))
        d = (Cy1) * d1
        result = (0.5 * d1 * d).flatten()
        w = np.where(result>0)
        J = result[w].sum()
        return J

    def J_prime ( self ):
        """
            The operator contribution to the cost function:
            
            J' = dJ/dx = (y - H(x))^T (C_y^-1) H'(x)
            
            This should be of dimensions n_
            """
	J = self.J()
	#if self.J_prime_approx != None:
	#    return self.J_prime_approx()
        x,Cx1,xshape,y,Cy1,yshape = self.getxy()
        H_prime_x = self.H_prime(x)
        try:
            Jprime = -np.array(self.linear.d1 * Cy1 * H_prime_x)
        except:
            Jprime = -np.array(self.linear.d1 * Cy1 * np.matrix(H_prime_x).T)[0]
	# Jprime should be the same shape as self.x.state
	if Jprime.size == self.x.state.size **2:
	    Jprime = Jprime.diagonal()
        return J,Jprime
        
            #return J,

    def J_prime_prime ( self ):
        """
            The operator contribution to the cost function:
            
            J'' = d^2J/(dx_idx_j) 
            
            Here, this is simply Cy^-1


	    NB: if you inherit this class you *must*
	   define a new J_prime_prime if you want to
	   calculate uncertainties
        
	   or sel J_prime_prime to return J_prime_prime_approx
	"""
        self.x_state.logger.info(' ... Calculating Hessian')
        x,Cx1,xshape,y,Cy1,yshape = self.getxy()
	# the operator here is I()
	# so we return C-1
	J,J_prime_0 = self.J_prime()
	return J,J_prime_0 , np.eye(Cy1)


    def J_prime_prime_approx ( self ):
        """
            The operator contribution to the cost function:
            
            J'' = d^2J/(dx_idx_j) 
            
	numerically.

	This is costly and you can normally find a better way to do it.
	"""
	## an attempt at doint it numerically 
	## revisit this some time
        # run the base J_prime
        J,J_prime_0 = self.J_prime()
        J_prime_0 = J_prime_0.copy()
        x0 = x.copy()
        xshape2 = np.array(xshape)
	x0 = x0.flatten()
        out = np.zeros((x0.size,x0.size))
	xmax = self.xmax.flatten()
        xmin = self.xmin.flatten()
        delta = self.delta.flatten()
	x1 = x0.copy()
	xnparam = xshape2.prod()
        for i in xrange(xnparam):
	    self.x_state.logger.info('...%d/%d'%(i+1,xnparam))
            if delta[i] != 0:
                x1[i] += delta[i]
                if x1[i] > xmax[i]: 
		    x1[i] = xmax[i] - delta[i]
                if x1[i] < xmin[i]: 
                    x1[i] = xmin[i]+delta[i]
	        d = x1[i] - x0[i]
		self.x.state = x1.reshape(xshape).copy()
		J1,J_prime_1 = self.J_prime()
		out[:,i] = ((J_prime_1-J_prime_0)/d).flatten()
		x1[i] = x0[i]
        self.x.state = x0.reshape(xshape).copy()
        return J,J_prime_0,out
    
    def JJ(self,x):
        '''
        A call to self.J that also loads x
            
        Required by self.J_prime_approx_3()
            
        '''
        self.x.state = x.reshape(self.x.state.shape)
        return self.J({})

    def J_prime_approx_1(self):
        '''
        A discrete approximation to J_prime
       
	The method assumes that J_prime is independent for
	each sample in the last column of the state vector.

	This will be appropriate e.g. for observation operators
	where the derivative of one observation does
	does not depend on any other samples. This means that we can
	take finite difference steps only over this last dimension
	and not over all samples. 
	
	see also:
	
		J_prime_approx_3
		J_prime_approx_2 (not yet implemented)
        
        '''
	# can probably reuse this code to write a better version of H_prime
        # J at x
        J0 = self.J()
        # make a copy of x.state
        xstate = self.x.state.copy()
        # now loop over each parameter
        # this greatly speeds up the calculation of an approximate J_prime
        # as we know that each observation is independent (i.e the J_prime
        # of each observation depends only on the state vactor elements for
        # that observation)
        for i in np.where(self.xranger>0)[0]:
            fx0 = xstate.copy()
            delta = self.delta[:,i]
            xmax = self.xmax[:,i]
            xmin = self.xmin[:,i]
            x0 = xstate[:,i]
            x1 = x0 + delta
            ww = np.where(x1 > xmax[i])
            x1[ww] = x0[ww] -delta[ww]
            ww = np.where(x1 < xmin)
            x1[ww] = xmin[ww]
            delta = x1 - x0
            ww = np.where(delta != 0)
            fx0[ww,i] = x1[ww]
            self.x.state = fx0
            J1 = self.J()
            dJ = np.zeros_like(delta)
            dJ[ww] = (J1-J0)/delta[ww]
            self.linear.J_prime[...,i] = dJ
        # reload x.state
        self.x.state = xstate.copy()
        return J0,self.linear.J_prime

    def J_prime_approx_3(self):
        '''
        A discrete approximation to J_prime
       
	This method is the 'backup' and default approximation method
	for J_prime as it treats all samples independently
	and so has to go ober x.size finite steps for J.

	Generally, J_prime_approx_2 or J_prime_approx_1 will
	be significantly faster than this, but there may be occasions
	when this method is appropriate.

	The method makes use of and requires DerApproximator
	so will only work if this python library is available.

	It might be adviseable at some point to write a backup method in case
	that is not installed, but thats not a very good use of time really ...

        '''
        try:
            from DerApproximator import get_d1
        except:
            raise Exception(\
                "Cannot import DerApproximator for derivative approx"\
		+ " in J_prime_approx_3 ... check it is installed or"\
		+ " avoid calling this method")
            J,J_prime = self.J_prime()
            self.J_prime_approx = J_prime.flatten()
            return self.J_prime_approx
            
        self.J_prime_approx = get_d1(self.JJ,np.array(self.x.state).flatten())
        return self.J_prime_approx


def tester(plot=True):
    '''
        Derivative test for individual J_primes
        
    '''
    from eoldas_Solver import eoldas_Solver
    from eoldas_ConfFile import ConfFile

    logdir = 'logs'
    logfile = 'log.dat'
    thisname = 'eoldas_Test1'
    conffile = ['semid_default.conf'] # ,'Obs1.conf']
    datadir = ['.']
    confs = ConfFile(conffile,logger=None,\
                     logdir=logdir,\
                     logfile=logfile,\
                     datadir=datadir)
    solver = eoldas_Solver(confs,thisname=thisname,\
                           logger=confs.logger,\
                           datadir=datadir)
    print "See logfile logs/log.dat for results of test"
    
    Expectation = '''
    - eoldas_Test1-obs-x - INFO - operator eoldas_Test1-obs-x
    - eoldas_Test1-obs-x - INFO - Calculating J_prime
    - eoldas_Test1-obs-x - INFO - Calculating approximate J_prime
    - eoldas_Test1-obs-x - INFO - J_prime        Range: [-78928.2236:83512.5508]
    - eoldas_Test1-obs-x - INFO - J_prime_approx Range: [-78928.2216:83512.5483]
    - eoldas_Test1-obs-x - INFO - Mean Diff -0.000050
    - eoldas_Test1-obs-x - INFO - Mean Abs Diff 0.001528
    - eoldas_Test1-obs-x - INFO - RMSE 0.001904
    - eoldas_Test1-modelt-x - INFO - operator eoldas_Test1-modelt-x
    - eoldas_Test1-modelt-x - INFO - Calculating J_prime
    - eoldas_Test1-modelt-x - INFO - Calculating approximate J_prime
    - eoldas_Test1-modelt-x - INFO - J_prime        Range: [-1.621250:1.696041]
    - eoldas_Test1-modelt-x - INFO - J_prime_approx Range: [-1.621250:1.696041]
    - eoldas_Test1-modelt-x - INFO - Mean Diff 0.000000
    - eoldas_Test1-modelt-x - INFO - Mean Abs Diff 0.000000
    - eoldas_Test1-modelt-x - INFO - RMSE 0.000000    
    '''
    print 'Expectation:'
    print Expectation
    for i in xrange(len(solver.confs.infos)):
        solver.prep(i)
        xopt = np.zeros(solver.nmask1+solver.nmask2)
        # load xopt into solver.root.x.state
        solver.unloader(xopt,solver.root.x.state)
        # randomise, so we get a good signal to look at
        # Make the xstate random here, just as a good test
        
            #solver.root.x.state = np.random.rand(solver.root.x.state.size).\
            #    reshape(solver.root.x.state.shape)
            #solver.root.x.state = (np.arange(solver.root.x.state.size)\
            #        / float(solver.root.x.state.size)).\
            #        reshape(solver.root.x.state.shape)
        
        #solver.root.x.state[...,0] = 1
        #solver.root.x.state[...,0]*10 + \
        # np.random.rand(solver.root.x.state[...,0].size)
        Jall = 0
        J_primeall = solver.root.x.state * 0.
        J_prime_approxall = solver.root.x.state * 0.
        for i,op in enumerate(solver.root.operators):
            
            op.logger.info('operator %s'%op.options.thisname)
            op.loader(solver.root)
            op.logger.info('Calculating J_prime')
            J,J_prime = op.J_prime({})
            
            # unload into the *full* representation
            J_prime_tmp = op.unloader(J_prime,solver.root.x.state,sum=True)
            J_primeall += J_prime_tmp
            J_prime = J_prime.flatten()
    
            op.logger.info('Calculating approximate J_prime')
            J_prime_approx = op.J_prime_approx_slow()
            J_prime_tmp = op.unloader(J_prime_approx,solver.root.x.state,sum=True)
            J_prime_approxall += J_prime_tmp
            J_prime_approx = J_prime_approx.flatten()
            
            d = (J_prime - J_prime_approx)
            n = float(len(d))
            op.logger.info('J_prime        Range: [%.6f:%.6f]'\
                           %(np.min(J_prime),np.max(J_prime)))
            op.logger.info('J_prime_approx Range: [%.6f:%.6f]'\
                           %(np.min(J_prime_approx),np.max(J_prime_approx)))
            op.logger.info('Mean Diff %f'%((d).sum()/n))
            op.logger.info('Mean Abs Diff %f'%(np.abs(d).sum()/n))
            op.logger.info('RMSE %f'%np.sqrt((d*d).sum()/n))            
            
            if plot:
                try:
                    import pylab
                    max = np.max([np.max(J_prime),np.max(J_prime_approx)])
                    min = np.min([np.min(J_prime),np.min(J_prime_approx)])
                    pylab.plot(min,max,'b-')
                    pylab.plot(J_prime,J_prime_approx,'o')
                    pylab.show()
                except:
                    pass
        if plot:
            # ideally all points should overlie each other
            # on a line
            J_prime = J_primeall.flatten()
            J_prime_approx = J_prime_approxall.flatten()

            J = solver.cost(xopt)
            J_prime2 = solver.root.x.state*0.
            J_prime2tmp = solver.cost_df(xopt)
            solver.loader(J_prime2tmp,J_prime2)
            J_prime2 =J_prime2.flatten()
            
            J_prime_approx2 = solver.root.x.state*0.
            J_prime_approx2tmp = solver.approx_cost_df(xopt)
            solver.loader(J_prime_approx2tmp,J_prime_approx2)
            J_prime_approx2 =J_prime_approx2.flatten()
            
            try:
                import pylab
                max = np.max([np.max(J_prime),np.max(J_prime_approx)])
                min = np.min([np.min(J_prime),np.min(J_prime_approx)])
                pylab.plot(min,max,'b-')
                pylab.plot(J_prime,J_prime_approx,'o')
                pylab.plot(J_prime,J_prime2,'v')
                # for some reason J_prime_approx2 is different
                pylab.plot(J_prime,J_prime_approx2,'^')
                pylab.show()
            except:
                pass

def demonstration(plot=False):
    tester(plot=plot)

if __name__ == "__main__":
    demonstration()




