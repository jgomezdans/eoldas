#!/usr/bin/env python

from eoldas_ParamStorage import *
from eoldas_Files import *
from eoldas_Lib import *
from eoldas_Parser import *
from eoldas_ConfFile import *
from eoldas_SpecialVariable import *
from eoldas_State import *
from eoldas_Operator import *
from eoldas_Spectral import *
from eoldas_DModel_Operator import *
from eoldas_Observation_Operator import *
from eoldas_Kernel_Operator import *
from eoldas_Solver import *
class eoldas(Parser):
    from eoldas_ParamStorage import ParamStorage
    from eoldas_Parser import Parser
    from eoldas_Solver import eoldas_Solver as Solver
    
    '''
    The Earth Observation Land Data Assimilation System: EOLDAS 
        
    This tool is designed primarily to be used from a command
    line prompt.
        
    The operation of the EOLDAS is controlled by these main mechanisms:
        
        1. The command line
        ==========================
            A parser is invoked. This has a set of default options 
            but can be extended through the configuration file.
            Type:
        
                eoldas.py --help
        
            to see this, or:
        
                eoldas.py --conf=default.conf --help
        
            to see how new command line options have been added.
            You should notice that you can now specify e.g.:
        
                --parameter.limits=PARAMETER.LIMITS
                --parameter.solve=PARAMETER.SOLVE
        
            This allows the user a good deal of flexibility in
            setting up EOLDAS experiments, as it shouldn't involve
            much writing code, just setting things up in a configuration
            file or files.
        
        2. The configuration file
        ==========================
            This is the main way of controlling an EOLDAS experiment.
            
        
        3. Calling the eoldas class
        ==========================
            e.g. from python:
            
            this = eoldas(args)
        
            where args is a list or a string containing the 
            equivalent of command line arguments
        
            e.g.
        
            this = eoldas('eoldas --help')
            
        
    '''
    def __init__(self,argv,name='eoldas',logger=None):
        from eoldas.eoldas_Lib import sortopt, sortlog

        argv = argv or sys.argv
        here = os.getcwd()
        self.thisname = name
        Parser.__init__(self,argv,name=self.thisname,logger=logger,\
            general=None,outdir=".",getopdir=False,parse=True)
        os.chdir(here)
        if not hasattr(self,'configs'):
            self.logger.error('No configration file specfied')
            help(eoldas)
            return

        self.thisname = name
        solver = Solver(self,logger=self.logger,name=self.thisname+'.solver')
        self.general = sortopt(self.root[0],'general',ParamStorage())
        self.general.write_results = sortopt(self.general,'write_results',True)
        self.general.calc_posterior_unc = sortopt(self.general,'calc_posterior_unc',False)
        self.general.passer = sortopt(self.general,'passer',False)
        self.solver = solver
        self.logger.info('testing full cost functions')
        for i in xrange(len(solver.confs.infos)):
        self.logger.info('%d/%d ...'%(i+1,len(solver.confs.infos)))
            # try an initial solver.prep(i)
            J = solver.cost(None)
            J_prime = solver.cost_df(None)
        self.logger.info('done')

        # give the user some info on where the log file is
        # in case theyve forgotten
        print 'logging to',self.general.logfile
        def solve(self,unc=None,write=None):
        '''
        Run the solver
            
        Options:
            
        unc     :   Set to True to calculate posterior uncertainty
        write   :   Set to True to write out datafiles
            
        '''
        if unc == None:
        unc = self.general.calc_posterior_unc
    if write == None:
        write = self.general.write_results
        solver = self.solver
        for i in xrange(len(solver.confs.infos)):
            solver.prep(i)
            J = solver.cost(None)
            J_prime = solver.cost_df(None)
            # run the solver
        if not self.general.passer:
                solver.solver()
            # Hessian
            if unc:
                solver.uncertainty()
            # write out the state
            if write:
                solver.write()
                # write out any fwd modelling of observations
                solver.writeHx()

    def uncertainty(self,write=None):
        '''
            Calculate uncertainty
            
            Options:
            
            write   :   Set to True to write out datafiles
            
        '''
        solver = self.solver
        if write == None:
            write = self.general.write_results

        for i in xrange(len(solver.confs.infos)):
            solver.prep(i)
            J = solver.cost(None)
            J_prime = solver.cost_df(None)
            # run the solver
            # write out the state
            if write:
                solver.write()
                # write out any fwd modelling of observations
                solver.writeHx()

    def write(self):
        '''
            Write out datafiles
            
            '''
        solver = self.solver
        for i in xrange(len(solver.confs.infos)):
            solver.prep(i)
            J = solver.cost(None)
            J_prime = solver.cost_df(None)
            # write out the state
            solver.write()
            # write out any fwd modelling of observations
            solver.writeHx()

