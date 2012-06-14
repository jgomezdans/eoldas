#!/usr/bin/env python
import pdb
import numpy as np
from eoldas_State import State
from eoldas_ParamStorage import ParamStorage
from eoldas_Operator import *
from eoldas_Lib import sort_non_spectral_model,sortlog

class eoldas_Solver(ParamStorage):
    '''
       Eoldas solver class
        
    '''
    def __init__(self,confs,logger=None,logfile=None,thisname=None,name=None,datadir=None,logdir=None):
        '''
        Initialise the solver.
            
        This does the following:
            1. Read configuration file(s)
            2. Load operators
            3. Test the call to the cost function
            
        There can be multiple groups of configuration files, so
        self.confs, that holds the core information setup here
        can contain multiple configurations.
        
        The number of configurations is len(confs.infos) and
        the ith configuration is conf = self.confs.infos[i].
            
        Various loggers are available throughout the classes
        used, but the top level logger is self.confs.logger,
        so you can log with e.g.
            
        self.confs.logger.info('this is some info')
            
        The root operator is stored in self.confs.root[i]
        for the ith configuration, so the basic call to the cost
        function is:
            
        J,J_prime = self.confs[i].parameter.cost(None)
            
        '''
        from eoldas_ConfFile import ConfFile
        from eoldas_Lib import sortopt
        name = name or thisname
        if name == None:
           import time
           thistime = str(time.time())
           name = type(self).__name__
           name =  "%s.%s" % (name,thistime)
        self.thisname = name
                
        self.confs = confs
        self.top = sortopt(self,'top',ParamStorage())
        self.top.general = sortopt(self.top,'general',ParamStorage())
        thisname = sortopt(self.top.general,'name',thisname or self.thisname)
        logfile = sortopt(self.top.general,'logfile',logfile or 'log.dat')
        logdir = sortopt(self.top.general,'logdir',logfile or 'logs')
        datadir = sortopt(self.top.general,'datadir',datadir or ['.'])
        self.logger = sortlog(self,logfile,logger or self.confs.logger,\
			name=self.thisname,logdir=logdir,debug=True)
        n_configs = len(self.confs.infos)
        self.confs.root = []
        self.have_unc = False
        try:
            logdir = logdir
        except:
            logdir = self.top.general.logdir

        # first set up parameter
        conf = confs.infos[0]
           
        general = conf.general
        op = conf.parameter    
        if not 'parameter' in conf.dict():
            raise Exception('No parameter field found in %s item %d'%\
                                (conf.__doc__,0))
        general.is_spectral = sortopt(general,'is_spectral',True)
        if not general.is_spectral:
            sort_non_spectral_model(op,conf.operator,logger=confs.logger)
        general.init_test = sortopt(general,'init_test',False)    
        confs.logger.info('loading parameter state')
        conf.parameter.name = 'Operator'
        parameter = eval(op.name)(op,general,\
                                      parameter=None,\
                                      logger=confs.logger,\
                                      name=name+".parameter",\
                                      datatype=list(conf.parameter.datatypes),\
                                      logdir=logdir,\
                                      logfile=logfile,\
                                      datadir=datadir)
        try:
            parameter.transform = parameter.options.x.transform
            parameter.invtransform = parameter.options.x.invtransform
        except:
            parameter.transform = parameter.options.x.names
            parameter.invtransform = parameter.options.x.names
            
        # we now have access to parameter.x.state, parameter.x.sd etc
        # and possibly parameter.y.state etc.
        operators = []
        for (opname,op) in conf.operator.dict().iteritems():
            if opname != 'helper':
	        #pdb.set_trace()
                exec('from eoldas_%s import %s'%(op.name,op.name))
                # make sure the data limits and x bounds are the same 
                # ... inherit from parameter
                op.limits = parameter.options.limits
                if not 'datatypes' in op.dict():
                    op.datatypes = 'x'
                thisop = eval(op.name)(op,general,parameter=parameter,\
                                           logger=confs.logger,\
                                           name=name+".%s-%s"%(thisname,opname),\
                                           datatype=list(op.datatypes),\
                                           logdir=logdir,\
                                           logfile=logfile,\
                                           datadir=datadir)
                # load from parameter
                thisop.loader(parameter)
                operators.append(thisop)
                try:
                    thisop.transform = parameter.options.x.transform
                    thisop.invtransform = parameter.options.x.invtransform
                except:
                    thisop.transform = parameter.options.x.names
                    thisop.invtransform = parameter.options.x.names
		thisop.ploaderMask = np.in1d(parameter.x_meta.state,thisop.x_meta.state)
		try:
		    thisop.invtransform = np.array(thisop.invtransform[thisop.ploaderMask])
		    thisop.transform = np.array(thisop.transform[thisop.ploaderMask])
		except:
		    ww = thisop.ploaderMask
		    thisop.invtransform = np.array(thisop.transform)[ww]
		    thisop.transform = np.array(thisop.transform)[ww]
        # sort the loaders
        parameter.operators = operators
        self.confs.root.append(parameter)
        # Now we have set up the operators
        # try out the cost function
	if general.init_test:    
            self.logger.info('testing cost function calls')
            J,J_prime = self.confs.root[0].cost()
            self.logger.info('done')
        self.confs.root = np.array(self.confs.root)

    def cost(self,xopt):
        '''
        Load xopt into the full state vector into root.x.state
            
        and calculate the cost J and J_prime
            
        The J_prime stored in self.J_prime is of dimension
        of the number of state variables that are targeted
        for estimation.
            
        '''
        if not xopt == None:
            self.loader(xopt,self.root.x.state)
        else:
            if not 'nmask' in self.dict():
                for i in xrange(len(self.confs.infos)):
                    self.prep(i)
            xopt = np.zeros(self.nmask1+self.nmask2)
        J,J_prime = self.root.cost()
        self.J_prime = xopt*0.        
        self.unloader(self.J_prime,J_prime.reshape(self.root.x.state.shape))
        return np.array(J).flatten()[0]
    
    cost_df = lambda self,x:self.J_prime
    cost_df.__name__ = 'cost_df'
    cost_df.__doc__ = '''
            This method returns the cost function derivative
            J_prime, assuming that self.cost(xopt) has
            already been called.
    '''
    
    def approx_cost_df(self,xopt):
        '''
            A discrete approximation to the cost fn and its derivative
            
            Mainly useful for testing as it can be done faster
            
            If DerApproximator is not available, the 'full'
            cost function is returned, which doesn't allow a test.
            Check log file or try:
            
                from DerApproximator import get_d1
            
            if you are concerned about that.
            
            '''
        try:
            from DerApproximator import get_d1
        except:
            self.confs.logger.error(\
                                    "Cannot import DerApproximator for derivative approx'")
            J = self.cost(xopt)
            J_prime = self.cost_df(xopt)
            self.J_prime_approx = J_prime.flatten()
            return self.J_prime_approx
        
        self.J_prime_approx = get_d1(self.cost,xopt)
        return self.J_prime_approx

    def prep(self,thisconf):
        '''
        A method to prepare the solver
            
        '''
        self.root = self.confs.root[thisconf]
        root = self.confs.root[thisconf]

        self.sortMask()

        self.op = sortopt(root.general,'optimisation',ParamStorage())
        self.op.plot = sortopt(self.op,'plot',0)
        self.op.name = sortopt(self.op,'name','solver')
        self.op.maxfunevals = sortopt(self.op,'maxfunevals',2e4)
        self.op.maxiter = sortopt(self.op,'maxiter',1e4)
        self.op.gtol = sortopt(self.op,'gtol',1e-3)
        self.op.iprint = sortopt(self.op,'iprint',1)
        self.op.solverfn = sortopt(self.op,'solverfn','scipy_lbfgsb')
        self.op.randomise = sortopt(self.op,'randomise',False)
        self.op.no_df = sortopt(self.op,'no_df',False)

        self.result = sortopt(root.options,'result',ParamStorage())
        self.result.filename = sortopt(root.options.result,'filename',\
                                       'results.pkl')
        self.result.fmt = sortopt(root.options.result,'format','pickle')
	try:
            self.transform = self.root.options.x.transform
            self.invtransform = self.root.options.x.transform
        except:
	    self.transform = None
	    self.invtransform = None

        # descend into the operators and identify any observation ops
        # we then want to be able to write out files (or plot data)
        # that are H(x).
        # We only do this for Operators that have both 'x' and 'y'
        # terms as we store the filename under 'y.result'
        self.Hx_ops = []
        for i,op in enumerate(root.operators):
            op.loader(root)
            if 'y_meta' in op.dict() and op.options.y.datatype == 'y':
                # There is a potential observation
                op.y_state.options.y.result = \
                    sortopt(op.y_state.options.y,'result',ParamStorage())
                op.y_state.options.y.result.filename = \
                    sortopt(op.y_state.options.y.result,\
                        'filename',op.y_state.thisname)
                op.y_state.options.y.result.format = \
                    sortopt(op.y_state.options.y.result,\
                        'format','PARAMETERS')
                state = op.y_state._state
                this = { \
                    'filename':op.y_state.options.y.result.filename,\
                    'format':op.y_state.options.y.result.format,\
                    'state':state,\
		    'y':op.y_state,\
                    'op':op,\
		    'transform':self.transform,\
		    'invtransform':self.invtransform,\
                    'Hx':op.linear.H}
		op.Hx_info = this
                self.Hx_ops.append(this)
	    else:
		this = { \
		    'transform':self.transform,\
		    'invtransform':self.invtransform,\
		}
		op.Hx_info = this

            
    def solver(self,thisconf=0):
        '''
        The solver. Use this method to run the optimisation
        code to minimse the cost function.
            
        Options:
        --------
        thisconf    :   Index of which configuration set to use.
                        By default, this is the first of the set.
                         
            
        '''
        from eoldas_Lib import sortopt
        self.logger.info('importing optimisation modules')
        try:
            from openopt import NLP
            self.isNLP = True
            self.confs.logger.info("NLP imported from openopt")             
            import scipy.optimize
            self.confs.logger.info("scipy.optimize imported")
        except:
            self.isScipy = False
            self.confs.logger.error("scipy.optimize NOT imported")
            self.confs.logger.error("Maybe you don't have scipy ...")
            self.confs.logger.error("***************************")
            self.confs.logger.error("Failed to find an optimizer")
            self.confs.logger.error(\
                        "... get a proper python installation")
            self.confs.logger.error("with scipy and/or openopt")
            self.confs.logger.error("NLP NOT imported from openopt")
            self.confs.logger.error("Maybe you don't have openopt ...")
            self.isNLP = False   
                
        root = self.root
        self.logger.info('done') 
        self.bounds = root.x_meta.bounds
        self.names  = root.x_meta.state
        self.xorig     = root.x.state.copy()

        self.lb = np.array(([ self.bounds[i][0] \
                for i in xrange(len(self.bounds))])\
                    *(self.xorig.size/len(self.bounds)))\
                    .reshape(self.xorig.shape)
        self.ub = np.array(([ self.bounds[i][1] \
                for i in xrange(len(self.bounds))])\
                    *(self.xorig.size/len(self.bounds)))\
                    .reshape(self.xorig.shape)
         
        xopt = np.zeros(self.nmask1+self.nmask2)
        lb   = np.zeros(self.nmask1+self.nmask2)
        ub   = np.zeros(self.nmask1+self.nmask2)

        self.unloader(xopt,root.x.state)
        self.unloader(lb,self.lb)
        self.unloader(ub,self.ub)
        if self.op.randomise:
            xopt = np.random.rand(xopt.size)*(ub-lb)+lb
            self.confs.logger.info('Randomising initial x')
            self.loader(xopt,root.x.state)
        # now we have to subset self.xorig, self.lb, self.ub
        # for the solver here
        if self.op.no_df:
            pp = NLP(self.cost, xopt.flatten(), iprint=self.op.iprint, \
                    goal='min', name=self.op.name, show=0,\
                    lb = lb, ub = ub, \
                    plot = int(self.op.plot) ,\
                    maxFunEvals = int(self.op.maxFunEvals), \
                    maxIter = int(self.op.maxIter), gtol = self.op.gtol)
        else:
            pp = NLP(self.cost, xopt.flatten(), iprint=self.op.iprint, \
                     goal='min', name=self.op.name, show=0,\
                     lb = lb, ub = ub, df=self.cost_df,\
                     plot = int(self.op.plot) ,\
                     maxFunEvals = int(self.op.maxfunevals), \
                     maxIter = int(self.op.maxiter), gtol = self.op.gtol)                
        # df=self.cost_df
        r = pp.solve(self.op.solverfn)
        self.loader(r.xf,root.x.state)
        self.min_cost = r.ff   
        self.confs.logger.info('Min cost = %s'%str(self.min_cost))
        

    def sortMask(self):
        '''
        Sort masks for loading and unloading data
        '''
        self.root.options.solve = \
                    sortopt(self.root.options,\
                        'solve',np.array([1]*len(self.root.x_meta.state)))
        self.solve = np.array(self.root.options.solve).copy()
        # set up masks for loading and unloading
        self.mask1 = np.zeros_like(self.root.x.state).astype(bool)
        ww = np.where(self.solve == 1)[0]
        self.mask1[...,ww] = True
        self.mask2 = np.zeros_like(self.root.x.state[0]).astype(bool)
        ww = np.where(self.solve == 2)[0]
        self.mask2[ww] = True
        self.nmask1 = self.mask1.sum()
        self.nmask2 = self.mask2.sum()
        self.wmask2 = np.where(self.mask2)[0]

            
    def loader(self,xopt,xstate,M=False):
        '''
        From xopt, that being optimized, load the full xstate
            
        '''
        if M:
            mask1 = self.mask1.flatten()
	    count = 0
            for i in np.where(mask1)[0]:
	        xstate[i,mask1] = np.array(xopt[count]).flatten()
                count += 1
            return

        xstate[self.mask1] = xopt[:self.nmask1].flatten()
        for i in self.wmask2:
            xstate[...,i] = xopt[self.nmask1+i]
            
    def unloader(self,xopt,xstate,M=False):
        '''
        From xstate, load xopt, that being optimized
        '''
        if M:
	    n = self.nmask1
	    mask1 = np.matrix(self.mask1.flatten())
	    MM = mask1.T * mask1
	    out = xstate[MM].reshape(n,n)
            return out
                
        xopt[:self.nmask1] = xstate[self.mask1].flatten()
        for i in self.wmask2:
            xopt[self.nmask1+i] = xstate[...,i][0] 
    
    def writeHx(self,filename=None,format=None,fwd=True):
        '''
            Writer function for observations ('y' states)
            
            Assumes filename in self.result.filename
            and format in self.result.format
            
            These can be specified in the config file as:
            
            parameter.result.filename
            parameter.result.format
            
            These can of course be over-ridden using the options
            filename and format.
            
            '''
        for this in self.Hx_ops:
            filename = this['filename']
            format = this['format']
            state = this['state']
            op = this['op']
	    # ensure Hx is up to date
	    J = op.J()
	    Hx = op.Hx
	    filenamey = filename+ '_orig'
            self.logger.info('Writing H(x) data to %s'%this['filename'])
            self.logger.info('Writing y data to %s'%filenamey)
            try:
		# dont stop just because it messes up plotting
                op.plot(ploterr=self.have_unc)
            except:
                pass
	    # write the observation data
	    state.name.fmt = format
	    state.write(filenamey,format)

            # to write Hx data we have to mimic
            # the y data 
            # first make a copy of the dataset
            self.datacopy = state.data.state.copy()
            self.sdcopy = state.data.sd.copy()
            self.C1copy = state.data.C1.copy()

	    # try fwdSd
            try:
                state.data.sd = op.fwdSd
                state.data.C1 = op.fwdUncert
            except:
                state.data.sd = self.sdcopy*0.
		state.data.C1 = self.C1copy*0

	    # insert the Hx data
	    # into state
            state.data.state = Hx.reshape(state.data.state.shape)

	    # write the file
            state.write(filename,format)
	    #then copy back the original data to tidy up
            state.data.state = self.datacopy.copy()
            state.data.sd = self.sdcopy.copy()
            state.data.C1 = self.C1copy.copy()
            #state.name.fmt = self.fmtcopy

    def uncertainty(self):
        '''
        Calculate the inverse Hessian of all operators
            
        '''
        from sys import stderr
        J,J_prime,J_prime_prime = self.root.hessian()
        # reduce the rank 
        Hsmall = self.unloader(None,J_prime_prime,M=True)
	self.have_unc = False
        try:
            IHsmall = np.matrix(Hsmall).I
	    self.have_unc = True
	    #U, ss, V = np.linalg.svd(Hsmall)
	    #ww = np.where(ss>ss[0]*0.01)
	    #sss = ss*0
	    #sss[ww] = 1./ss[ww]
	    #IHsmall = np.dot(U, np.dot(np.diag(sss), V))
        except:
            IHsmall = np.matrix(Hsmall)
	J_prime_prime = J_prime_prime*0
        self.loader(IHsmall,J_prime_prime,M=True)
        self.Ihessian = J_prime_prime
	self.IHsmall = IHsmall
        dd = self.IHsmall.diagonal()
  
	#print >> stderr, "WARNING: ... something "
	try:
            self.root.x.sd = np.sqrt(np.array(self.Ihessian.diagonal()).flatten())
	    self.root.Ihessian = self.Ihessian
	    nfwd  = self.root.fwdError()
	    # if this works you should get uncertainty if fwd modelling in 
	    # op.fwdUncert for each operator self.operators
	except:
	    # then you have -ve values
	    self.logger.error("WARNING: ill-conditioned system with unstable estimates of uncertainty")
        for i,op in enumerate(self.root.operators):
            # propagate the sd data down
	    op.x.sd = self.root.x.sd.reshape(op.loaderMask.shape)[op.loaderMask]

    def write(self,filename=None,format=None):
        '''
        Writer function for the state variable
            
        Assumes filename in self.result.filename
        and format in self.result.format
            
        These can be specified in the config file as:
            
        parameter.result.filename
        parameter.result.format
            
        These can of course be over-ridden using the options
        filename and format.
            
        '''
        filename = filename or self.result.filename
        format = format or self.result.format
        self.logger.info('writing results to %s'%filename)
        try:
	    np.savez(filename.replace('.dat','.npz'),Ihessian=self.Ihessian,IHsmall=self.IHsmall)
        except:
            pass
        try:
            self.root.plot(noy=True,ploterr=self.have_unc)
        except:
            pass
        self.root.x_state.write(filename,None,fmt=format)

def tester():
    '''
    Derivative test for total J_prime
        
    It should plot a scatterplot of derivatives calculated
    by independent methods.
        
    They should lie on a 1:1 line, or if not, there
    is a problem with the derivative calculation implemented.
        
    In this case, you should check the individual operator
    derivatives carefully, using e.g. tester() in 
    eoldas_Operator.py
    '''
    solver = eoldas_Solver()
    print "See logfile for results of test"
    
    for i in xrange(len(solver.confs.infos)):
        solver.prep(i)
        xopt = np.zeros(solver.nmask1+solver.nmask2)
        
        # randomise, so we get a good signal to look at
        # Make the xstate random here, just as a good test
        #xopt = np.random.rand(solver.nmask1+solver.nmask2)
        
        solver.loader(xopt,solver.root.x.state)
        J = solver.cost(xopt)
        J_prime = solver.cost_df(xopt)
        J_prime_approx = solver.approx_cost_df(xopt)
        #ww = np.where(J_prime>0)
        #J_prime = J_prime[ww]
        #J_prime_approx = J_prime_approx[ww]
        try:
            import pylab
            max = np.max([np.max(J_prime),np.max(J_prime_approx)])
            min = np.min([np.min(J_prime),np.min(J_prime_approx)])
            pylab.plot(min,max,'b-')
            pylab.plot(J_prime,J_prime_approx,'o')
            pylab.show()
        except:
            pass
    
def demonstration():
    '''
    An example of running EOLDAS
    '''
    from eoldas_ConfFile import ConfFile
    print "Testing ConfFile class with conf file eoldas_Test1"
    logdir = 'test/eoldas_Test/log'
    logfile = 'log.dat'
    thisname = 'eoldas_Test1'
    conffile = ['semid_default.conf']
 
	#['Obs1.conf']
    datadir = ['.']
    confs = ConfFile(conffile,\
                             logdir=logdir,\
                             logfile=logfile,\
                             datadir=datadir)

    solver = eoldas_Solver(confs,thisname=thisname,\
                           logdir=logdir,\
                           logfile=logfile,\
                           datadir=datadir)
    
    for i in xrange(len(solver.confs.infos)):
        solver.prep(i)
        # try an initial calculation
        J = solver.cost()
        J_prime = solver.cost_df(None)
        # run the solver
        solver.solver()
        # Hessian
        solver.uncertainty()
        # write out the state
        solver.write()
        # write out any fwd modelling of observations
        solver.writeHx()

if __name__ == "__main__":
    from eoldas_Solver import *
    help(eoldas_Solver)
    help(tester)
    help(demonstration)
    demonstration()
