#!/usr/bin/env python
import pdb
import numpy as np
#from eoldas_model import subset_vector_matrix
from eoldas_State import State
from eoldas_ParamStorage import ParamStorage
from eoldas_Operator import *

class DModel_Operator ( Operator ):
    
    def preload_prepare(self):
        '''
        Here , we use preload_prepare to make sure 
        the x & any y data are gridded for this
        operator. This greatly simplifies the 
        application of the differential operator.
            
        This method is called before any data are loaded, 
        so ensures they are loaded as a grid.
        '''
	from eoldas_Lib import sortopt
        for i in np.array(self.options.datatypes).flatten():
            # mimic setting the apply_grid flag in options
            if  self.dict().has_key('%s_state'%i):
                self['%s_state'%i].options[i].apply_grid = True
        self.novar = sortopt(self,'novar',False)
        self.gamma_col = sortopt(self,'gamma_col',None)
	self.beenHere =False

    def postload_prepare(self):
        '''
        This is called on initialisation, after data have been read in
           
	Here, we load parameters specifically associated with the
	model H(x).

	In the case of this differential operator, there are:

		model_order	: order of the differential operator
				  (integer)
		wraparound	: edge conditions
				  Can be:
					periodic
					none
					reflexive
		lag		: The (time/space) lag at which
				  the finite difference is calculated
				  in the differential operator here.
				  If this is 1, then we take the difference
				  between each sample point and its neighbour.
				  This is what we normally use. The main
				  purpose of this mechanism is to allow
				  differences at multiple lags to be 
			          calculated (fuller autocorrelation function
				  constraints as in kriging)

				  Multiple lags can be specified (which you could
				  use to perform kriging), in which case lag
				  weight should also be specified.

		lag_weight	: The weight associated with each lag. This will
				  generally be decreasing with increasing lag for
				  a 'usual' autocorrelation function. There is no point
				  specifying this if only a single lag is specified
				  as the function is normalised.
				
	If the conditions are specified as periodic
	the period of the function can also be specified, e.g. 
	for time varying data, you could specify 365 for the 
	periodic period.

	These are specified in the configuration file as

		operator.modelt.rt_model.model_order
		operator.modelt.rt_model.wraparound
		operator.modelt.rt_model.lag
		operator.modelt.rt_model.lag_weight

	The default values (set here) are 1, 'none', 1 and 1 respectively.

	To specify the period for `periodic` specify e.g.:

	[operator.modelt.rt_model]
	wraparound=periodic,365

	The default period is set to 0, which implies that it is periodic
	on whatever the data extent is.

	Or for multiple lags:

	[operator.modelt.rt_model]
	lag=1,2,3,4,5
	lag_weight=1,0.7,0.5,0.35,0.2

	NB this lag mechanism has not yet been fully tested
        and should be used with caution. It is intended more
	as a placeholder for future developments.

	Finally, we can also decide to work with 
	inverse gamma (i.e. an uncertainty-based measure)

	This is achieved by setting the flag

		operator.modelt.rt_model.inverse_gamma=True

	This flag should be set if you intend to estimate gamma in the
	Data Assimilation. Again, the is experimental and should be used
	with caution.
 
        '''
        from eoldas_Lib import sortopt
        self.rt_model = sortopt(self.options,'rt_model',ParamStorage())
	self.rt_model.lag = sortopt(self.rt_model,'lag',1)
	self.rt_model.inverse_gamma= \
		sortopt(self.rt_model,'inverse_gamma',False)
        self.rt_model.model_order = \
			sortopt(self.rt_model,'model_order',1)
	self.rt_model.wraparound = \
			sortopt(self.rt_model,'wraparound','none')
        self.rt_model.wraparound_mod = 0
	if np.array(self.rt_model.wraparound).size == 2 and \
		np.array(self.rt_model.wraparound)[0] == 'periodic':
	    self.rt_model.wraparound_mod = \
			np.array(self.rt_model.wraparound)[1]
	    self.rt_model.wraparound = \
                        np.array(self.rt_model.wraparound)[0]
        self.rt_model.lag = \
			sortopt(self.rt_model,'lag',[1])
	self.rt_model.lag = np.array(self.rt_model.lag).flatten()

	self.rt_model.lag_weight = \
			sortopt(self.rt_model,'lag_weight',[1.]*\
			self.rt_model.lag.size)
	self.rt_model.lag_weight = np.array(\
			self.rt_model.lag_weight).flatten().astype(float)

	if self.rt_model.lag_weight.sum() == 0:
	    self.rt_model.lag_weight[:] = np.ones(self.rt_model.lag_weight.size)
 
	self.rt_model.lag_weight = self.rt_model.lag_weight\
			/ self.rt_model.lag_weight.sum()
    
    def setH(self):
        '''
            This method sets up the matrices required for the model.
            
            This operator is written so that it can apply smoothing
            in different dimensions. This is controlled by the model
            state vector. 
            
            The names of the states are stored in self.x_meta.location
            and the associated location information in self.x_meta.location.
            So, we look through these looking for matches, e.g. 'row' in 
            location and 'gamma_row' in names would mean that we want
            to apply the model over the row dimension. There should be only
            one gamma term in the state vectors for this operator. If you
            give more than one, only the last one will be used.
            
            NOT YET IMPLEMENTED: 
            The model can be applied to multiple dimensions by specifying
            e.g. gamma_time_row. If you want separate gammas for e.g. 
            time and row, then you should use separate operators. If
            gamma_roccol is specified, then the model applies to
            Euclidean distance in row/col space.
            
            Formally, the problem can be stated most simply as a matrix
            D so that gamma D x is the rate of change of x with respect 
            to the target location variable (time, row, col etc). 
            The job of this method then is to form and store D.
            
            The main complication to this is we have to split up x into
            those terms that we will apply D to (x2 here) and separately
            pull out the gamma terms. The resultant matrix D then needs to
            be re-formed so as to apply to the whole vector x, rather than 
            just x2. We do this with masks.
            
            On input, x is a 1D vector.

        '''
	x,Cx1,xshape,y,Cy1,yshape = self.getxy()
        # the names of the variables in x
        names = np.array(self.x_meta.state)
        # the names of the location information (e.g. time, row, col)
        location = self.x_meta.location
        self.logger.info('Setting up model matrices...')

        if self.x_meta.is_grid:
            try:
		self.x.location = self.x_state.ungridder.location
		self.x.qlocation = self.x_state.ungridder.qlocation
            except:
                raise Exception("You are trying to ungrid a dataset that wasn't gridded using State.regrid()" +\
                    " so the ungridder information is not available. Either load the data using State.grid " +\
                    " or set it up some other way or avoid calling this method with this type of data")
        
        # first, reshape x from its 1-D form to
        # have the same shape as self.x.state. We store
        # this shape as xshape.
        xshape = self.x.state.shape
        
        # we can't change the tuple directly, so need a 
        # vector representation that we can manipulate
        # This is xshaper
        xshaper = np.array(xshape)
        
        # the data are assumed loaded into x
        
        # At this point, x2 is just a copy of the full input vector x
        # mask then is a mask of the same size as self.x_meta.state
        # by deafult, this mask is True. We will modify it to
        # take out bits we dont want later.
        x2 = x.reshape(xshape)
        mask = np.ones_like(x2).astype(bool)
        
        # We now need to recognise any gamma terms that might be in
        # the state vector. Candidates are 'gamma_%s'%(location)
        # e.g. gamma_time.
        
        # The number of dimensions of x can vary, depending on how many
        # loaction terms are used, so its a little tricky to 
        # pull the information out.
        # We loop over the locations, indexed as i
        self.linear.datamask = np.ones(xshape[-1]).astype(bool)
        for i in xrange(len(location)):
            # and form the name of the candidate term in the variable 'this'
            this = 'gamma_%s'%location[i]
            ww = np.where(this == names)[0]
            # Then we see if it appears in the names of the state variables
            if len(ww):
                # form a mask so we dont apply the operator to gamma
                # terms. Note that *all* gamma terms are masked
                # even though we only actually use the last one we
                # come across. 
                # we use [...,ww[0]] because the identifier for the
                # state is always in the final dimension.
                mask[...,ww[0]] = False
                # We store ww[0] as it will alllow us to access gamma
                # in subsequent calls in this same way. This is
                # self.linear.gamma_col
                self.linear.gamma_col = ww[0]
                # and is used as ...
                gammas = x2[...,self.linear.gamma_col]
                self.linear.datamask[self.linear.gamma_col] = False
                # We want to store an index into which of the 
                # location vector terms we are dealing with here.
                # This is 
                self.linear.gamma_loc = i
                # Once we apply the mask to get rid of the gamma columns
                # we need to keep track of the new shape for x2
                # This will be x2shape
                xshaper[-1] -= 1
        
        self.linear.x2shape = tuple(xshaper)
        self.linear.x2mask = mask.flatten()
        # so, apply the mask to take out the gamma columns
        x2 = x[self.linear.x2mask].reshape(self.linear.x2shape)
    
        # We next need access to the location information
        # for the selected dimension self.linear.gamma_loc.
        # If the data are gridded, we need to form the relevant information
        # Ungridded data we can access location directly as it is explicitly
        # stored. We store the location vector as 'locations'
        try:
            locshape = gammas.shape
        except:
            # If no gamma term is given, it is implicit that it is 
            # the first dimension of location, but we have no data to mask
            self.linear.gamma_col = None
            self.linear.gamma_loc = 0
            locshape = (0)
            gammas = x2[...,0]*0.+1.0
        #if self.x_meta.is_grid:
            # the locational variable of interest is self.linear.gamma_loc
            # the grid is dimensioned e.g. [t,r,c,p]
            # so we need e.g. locations which is of dimension
            # e.g. [t,r,c]
        #    locations = self.x.location
        # access the ungridded location data
        lim = self.x_meta.qlocation[self.linear.gamma_loc]
        nloc = lim[1] - lim[0] + 1
        locations = self.x.location[...,self.linear.gamma_loc]
	locshape = tuple(np.array(self.x.location.shape)[:-1])

	for (i,lag) in enumerate(self.rt_model.lag):
	    wt = self.rt_model.lag_weight[i]
	    
            slocations = wt*(np.roll(locations,lag,\
				axis=self.linear.gamma_loc) - locations).astype(float)
            slocations2 = (locations - np.roll(locations,-lag,\
				axis=self.linear.gamma_loc)).astype(float)
                
            # If there is no variation, it is a waste of time to calculate
            # the derivative
            if i == 0 and np.abs(slocations).sum() + np.abs(slocations2).sum() == 0:
                # there is no variation here    
                self.novar = True
                return 0
            self.novar = False
            ww = np.where(slocations > 0)
            # error found in wraparound when lim[-1] is not 1: fixed by normalising by lim[-1]
            # Lewis 22 June
	    mod = int(self.rt_model.wraparound_mod)/float(lim[-1]) or slocations.shape[self.linear.gamma_loc]
            if self.rt_model.wraparound == 'reflexive':
                slocations[ww] = 0.
                #slocations[ww] = -np.fmod(mod - slocations[ww],mod)
	    elif self.rt_model.wraparound == 'periodic':
                if self.rt_model.wraparound_mod == 0:
	            slocations[ww] = slocations2[ww]
	        else:
		    slocations[ww] = -np.fmod( mod - slocations[ww]/lim[-1],mod)
	    else:   # none
	        slocations[ww] = 0.
            ww = np.where(slocations != 0)
            slocations[ww] = 1./slocations[ww]

	    if i == 0:
        	# Form the D matrix. This is of the size required to 
        	# process the x2 data, and this is the most convenient
       	 	# form to use it in
        	m = np.zeros(slocations.shape * 2)
            ww = np.where(slocations != 0)
	    ww2 = np.array(ww).copy()
	    ww2[self.linear.gamma_loc] = ww2[self.linear.gamma_loc] - lag
	    ww2 = tuple(ww2)
	    m[ww*2] = m[ww*2] - slocations[ww]
   	    if False and self.rt_model.wraparound == 'reflexive':
		ww2 = np.abs(ww-lag)	
	        # this is looped as there might be multiple elements with the	
	        # same index for the reflecxive case
		if m.ndim > 2:
		    raise Exception("Not yet implemented: Can't use reflexive mode for multi-dimensions yet")
                for (c,j) in enumerate(ww2):
                    m[j,ww[c]] = m[j,ww[c]] + slocations[ww[c]]
            else:
		ww = tuple(ww)
		ww2 = tuple(ww2)
	   	m[ww2+ww] = m[ww2+ww] + slocations[ww]
        # fix for edge conditions
        dd = m.copy()
        dd = dd.reshape(tuple([np.array(self.linear.x2shape[:-1]).prod()])*2)
        ddw = np.where(dd.diagonal() == 0)[0]
        for d in (ddw):
            ds = -dd[d,:].sum()
            dd[d,:] += dd[d,:]
            dd[d,d] = ds
        m = dd.reshape(m.shape)
        self.logger.info('Caching model matrices...')
        # 
        if np.array(xshape).prod() == Cy1.size:
            self.linear.C1 = Cy1.reshape(xshape)[mask]\
                    .reshape( self.linear.x2shape )
	elif xshape[1] == Cy1.size:
            self.linear.C1 = np.tile(Cy1,xshape[0])[mask.flatten()].reshape( self.linear.x2shape )
        else:
            raise Exception("Can't deal with full covar matrix in DModel yet")
        nn = slocations.flatten().size
	m = m.reshape(nn,nn) 
        self.linear.D1 = np.matrix(m).T    
	for i in xrange(1,self.rt_model.model_order):
	    m = np.matrix(self.linear.D1).T * m  
        self.linear.D1 = m     
        self.logger.info('... Done')
        return True

    def J(self):
        '''
        A slightly modified J as its efficient to 
        precalculate things for this model
       
	J = 0.5 * x.T D1.T gamma^2 D1 x
 
        '''
        x,Cx1,xshape,y,Cy1,yshape = self.getxy()
        self.Hsetup()
        if self.novar:
            return 0
        xshape = self.x.state.shape
	try:
            if self.linear.gamma_col != None:
                gamma = x.reshape(self.x.state.shape)\
                    [...,self.linear.gamma_col].flatten()
            else:
                # no gamma variable, so use 1.0 
                gamma = x.reshape(self.x.state.shape)\
                    [...,0].flatten()*0.+1.
        except:
	    self.logger.error('gamma_col not set ... recovering and assuming no variation here')
	    self.linear.gamma_col = None
	    gamma = x.reshape(self.x.state.shape)[...,0].flatten()*0.+1.
	    self.novar = True
	    self.Hsetup()
	    return 0
	
        x2 = x[self.linear.x2mask].reshape(self.linear.x2shape)
        J = 0.
        i = 0
	if self.rt_model.inverse_gamma:
	    tgamma = 1./gamma
	else:
	    tgamma = gamma    
        for count in xrange(self.x.state.shape[-1]):
            if count != self.linear.gamma_col:
                C1 = np.diag(self.linear.C1[...,i].\
                             reshape(self.linear.D1.shape[0]))
                x2a = x2[...,i].reshape(self.linear.D1.shape[0])
                xg = np.matrix(x2a*tgamma).T
                dxg = self.linear.D1.T * xg
                
                J += np.array(0.5 * dxg.T * C1 * dxg)[0][0]
                i += 1
    #print x[0],J
        return np.array(J).flatten()[0]
       
    def J_prime_prime(self):
	'''
	Calculation of J''

	We already have the differntial operator
	self.linear.D1 and self.gamma
	after we call self.J_prime()

	Here, J'' = D1.T gamma^2 D1

	J' is of shape (nobs,nstates)
	which is the same as the shape of x
	
	D1 is of shape (nobs,nobs)
	which needs to be expanded to 
	(nobs,nstates,nobs,nstates)

	'''
	x,Cx1,xshape,y,Cy1,yshape = self.getxy()
	J,J_prime = self.J_prime() 
	xshape = self.x.state.shape

	if not 'linear' in self.dict():
	    self.linear = ParamStorage()
	if not 'J_prime_prime' in self.linear.dict():
	    self.linear.J_prime_prime = \
		np.zeros(xshape*2) 
	else:
	    self.linear.J_prime_prime[:] = 0 
	# we need an indexing system in case of multiple
	# nobs columns
        x2a = np.diag(np.ones(self.linear.x2shape[:-1]).flatten())
	try:
	    gamma = self.linear.gamma.flatten()
	except:
            if self.linear.gamma_col != None:
                gamma = x.reshape(self.x.state.shape)\
                    [...,self.linear.gamma_col].flatten()
            else:
                # no gamma variable, so use 1.0 
                gamma = x.reshape(self.x.state.shape)\
                    [...,0].flatten()*0.+1.
            gamma = self.linear.gamma.flatten()
        if self.rt_model.inverse_gamma:
            tgamma = 1./gamma
	    dg = 2./(gamma*gamma*gamma)
        else:
            tgamma = gamma
	    dg = 1.0
        nshape = tuple([np.array(self.linear.x2shape[:-1]).prod()])
        D1 = np.matrix(self.linear.D1.reshape(nshape*2))
	i = 0
        # so, e.g. we have xshape as (50, 100, 2)
        # because one of those columns refers to the gamma value
        # self.linear.gamma_col will typically be 0
	for count in xrange(xshape[-1]):
            if count != self.linear.gamma_col:
                # we only want to process the non gamma col
                C1 = np.diag(self.linear.C1[...,i].\
                             reshape(self.linear.D1.shape[0]))
	        xg = np.matrix(x2a*tgamma*tgamma)
	        dxg = D1 * xg
	        deriv = np.array(dxg.T * C1 * D1)
                # so we have gamma^2 D^2 which is the Hessian
                # we just have to put it in the right place now
                # the technical issue is indexing an array of eg 
                # (50, 100, 2, 50, 100, 2)
                # but it might have more or fewer dimensions
                nd = len(np.array(xshape)[:-1])
                nshape = tuple(np.array(xshape)[:-1])
                if nd == 1:
                    self.linear.J_prime_prime[:,count,:,count] = deriv.reshape(nshape*2)
                elif nd == 2:
                    self.linear.J_prime_prime[:,:,count,:,:,count] = deriv.reshape(nshape*2)
                elif nd == 3:
                    self.linear.J_prime_prime[:,:,:,count,:,:,:,count] = deriv.reshape(nshape*2)
                else:
                    self.logger.error("Can't calculate Hessian for %d dimensions ... I can only do up to 3"%nd)
 
	        #ww = np.where(deriv)
	        #ww2 = tuple([ww[0]]) + tuple([ww[0]*0+count]) \
		#		+ tuple([ww[1]] )+ tuple([ww[0]*0+count])
                #x1 = deriv.shape[0]
                #x2 = self.linear.J_prime_prime.shape[-1]
                #xx = self.linear.J_prime_prime.copy()
                #xx = xx.reshape(x1,x2,x1,x2)
                #xx[ww2] = deriv[ww]
	        #self.linear.J_prime_prime = xx.reshape(self.linear.J_prime_prime.shape)
		i += 1
        if self.linear.gamma_col != None:
            c = self.linear.gamma_col
            nd = len(np.array(xshape)[:-1])
            nshape = tuple(np.array(xshape)[:-1])
            deriv = np.diag(dg*2*J/(tgamma*tgamma)).reshape(nshape*2)
            if nd == 1:
                self.linear.J_prime_prime[:,c,:,c] = deriv
            elif nd == 2:
                self.linear.J_prime_prime[:,:,c,:,:,c] = deriv
            elif nd == 3:
                self.linear.J_prime_prime[:,:,:,c,:,:,:,c] = deriv
            else:
                self.logger.error("Can't calculate Hessian for %d dimensions ... I can only do up to 3"%nd)

	    #dd = np.arange(nshape[0])
            #x1 = dd.shape[0]
            #x2 = self.linear.J_prime_prime.shape[-1]
            #xx = self.linear.J_prime_prime.copy()
            #xx = xx.reshape(x1,x2,x1,x2)
            #xx[dd,dd*0+self.linear.gamma_col,\
            #	dd,dd*0+self.linear.gamma_col] = dg*2*J/(tgamma*tgamma)
            #self.linear.J_prime_prime = xx.reshape(self.linear.J_prime_prime.shape)
	n = np.array(xshape).prod()
	return J,J_prime,self.linear.J_prime_prime.reshape(n,n)

    def J_prime(self):
        '''
            A slightly modified J as its efficient to 
            precalculate things for this model
           
	   J' = D.T gamma^2 D x 
            
        '''
        J = self.J()
        if self.novar:
            return 0,self.nowt
        x,Cx1,xshape,y,Cy1,yshape = self.getxy()
        x2 = x[self.linear.x2mask].reshape(self.linear.x2shape)
        if self.linear.gamma_col != None:
            gamma = x.reshape(self.x.state.shape)\
                [...,self.linear.gamma_col].flatten()
        else:
            # no gamma variable, so use 1.0 
            gamma = x.reshape(self.x.state.shape)\
                [...,0].flatten()*0.+1.
        #gamma = self.linear.gamma.flatten()

        if self.rt_model.inverse_gamma:
            tgamma = 1./gamma
            dg = -1./(gamma*gamma)
        else:
            tgamma = gamma
            dg = 1.0

        g2 = tgamma * tgamma
        xshape = self.x.state.shape
        J_prime = np.zeros((x.shape[0]/xshape[-1],xshape[-1]))
        D2x_sum = 0.
        # loop over the non gamma variables
        i = 0
	# store gamma in case needed elsewhere
	self.linear.gamma = gamma
        for count in xrange(self.x.state.shape[-1]):
            if count != self.linear.gamma_col:
                C1 = np.diag(self.linear.C1[...,i].\
                             reshape(self.linear.D1.shape[0]))
                x2a = x2[...,i].reshape(self.linear.D1.shape[0])
                xg = np.matrix(x2a*tgamma).T
                dxg = self.linear.D1 * xg
                deriv = np.array(dxg.T * C1 * self.linear.D1)[0]
                J_prime[...,count] = deriv * tgamma
                #if self.linear.gamma_col != None:
                #        J_prime_gamma = deriv * x2a
                #        D2x_sum = D2x_sum + J_prime_gamma
                i += 1
        if self.linear.gamma_col != None:    
            J_prime[...,self.linear.gamma_col] = dg*2*J/tgamma
        
        return J,J_prime

    def Hsetup(self):
        '''
        setup for the differential operator H(x) 
            
        '''
        if not self.beenHere and not 'H_prime' in self.linear.dict():
            self.logger.info('Setting up storage for efficient model operator')
            if 'y' in self.dict():
                self.linear.H = np.zeros(self.y.state.shape)
                self.linear.H_prime = np.zeros(self.y.state.shape*2)
            else:
                self.linear.H = np.zeros(self.x.state.shape)
                self.linear.H_prime = np.zeros(self.x.state.shape*2)
            self.setH()
            if self.novar:
                self.nowt = 0. * self.x.state
                #del self.linear.H_prime, self.linear.H
            self.beenHere = True


