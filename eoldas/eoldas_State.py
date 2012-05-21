#!/usr/bin/env python
from eoldas_ParamStorage import ParamStorage
from eoldas_SpecialVariable import SpecialVariable
import pdb
import numpy as np
from eoldas_Files import writer
from eoldas_Lib import sortlog

class State(ParamStorage):
    '''
    A data class to represent and manipulate state vector data
        
    '''

    def __init__(self,options,limits=None,bounds=None,\
                 datatype='x',names=None,logdir=None,writers={},\
                 control=None,location=None,env=None,debug=None,\
                 grid=True,logger=None,\
                 datadir=None,logfile=None,name=None,info=[],readers=[]):
        '''
        Initialise State class
            
        Inputs:
            options     :   A ParamStorage data type. Most terms can be 
                            over-ridden via Options, but it should normally 
                            contain:
            
                options.thisname    :   same as name below
                options.names       :   same as names below
                options.datadir     :   same as datadir
                options.location    :   same as location
                options.limits      :   same as limits
                options.control     :   same as control
                options.env         :   an environment variable containing
                                        directory names to search for input
                                        files.
                options.bounds      :   same as bounds
            
        
        Options:
            limits      :   List of limits to apply when reading data.
                            Limits atre also used to quantize locational data.
                            Should be of the same length as `location`
                            with each sub-list of the form [min,max,step].
                            The default is [0,None,1].
            bounds      :   List of bounds to be applied to the data, or 
                            the same length as datatypes, with each sub-list of
                            the form [min,max]
            datatype    :   The state variable data types. This will 
                            normally be `x` or `y`but may for instance be
                            `x1` or `y2`
            names       :   List of state vector names 
                            e.g. [`lai`, `chlorophyl`]
            control     :   List of strings describing control variables
                            e.g. [`mask`,`vza`,`sza`,`vaa`,saa`]
            location    :   List of strings describing location, e.g.
                            [`time`,`row`,`col`]
            datadir     :   List of ditrectories to search for data in.
            logfile     :   File name to be used for logging
            logdir      :   Directory to put logfile if logfile is not
                            an absolute pathname.
            name        :   Name to be used in logging. The default is None
                            so there is no logging. If set to True, logs 
                            to stdout. Otherwise logs to logfile if set.
            header      :   Over ride the default header string used
                            for pickle files.
            info        :   A list that is passed through to file readers etc
            readers     :   A list of file readers that is pre-prended to 
                            existing ones.
            writers     :   A list of file readers that is pre-prended to 
                            existing ones.
            grid        :   Flag to interpret the data on a grid
                            This is done by interpolation, but a mask
                            is set to 2 for such values.
        '''
        self.reinit(options,limits=limits,bounds=bounds,datatype=datatype,\
                    control=control,names=names,location=location,env=env,\
                    logdir=logdir,writers=writers,logger=logger,\
                    datadir=datadir,logfile=logfile,name=name,info=info,\
                    readers=readers,debug=debug,grid=grid)

    def reinit(self,options,names=None,datatype=None,limits=None,\
            bounds=None,control=None,location=None,env=None,header=None,\
               logdir=None,writers={},grid=False,logger=None,\
            datadir=None,logfile=None,name=None,info=[],readers=[],debug=None):
        '''
        Method to re-initialise the class instance
            
        The setup is on the whole controlled by the datatype which
            contains e.g. 'x'. This is used to set up the members
            self.x and self.y as SpecialVariables (see SpecialVariable
            in eoldas_SpecialVariable.py). There are some special attributes
            for datatypes starting with 'y'. These are assumed to be 
            observational data, which means that when they are read, the 
            data names associated with them are not limited to those in 
            self.names but rather set to whatever is read in in the data. This 
            is because the data names for observational data may be terms such 
            as waveband names etc that need special interpretation. Also,
            the default output format for observational data is 
            different to that of other data.  
            
        The elements self.state is a SpecialVariables which means 
            that they can be assigned various data types (see SpecialVariables)
            and loaded accordingly (e.g. if a filename is specified, this is 
            read in to the data structure. The SpecialVariables contain 'hidden'
            datasets, which here are mainly the 'control' and 'location'
            information. A SpecialVariable has two internal structures: `data`
            and `name`. The former is used to store data values (e.g. the 
            state values) and the latter to store associated metadata.
            For example, `control` is passed here e.g. as [`mask`,`vza`] and this
            gives the metadata that are stored in `name`. The actual values
            of the control data are stored in the `data` section. For
            location, we might be passed [`time`,`row`,`col`], so this is set
            in names.location, and the data.location contains the values
            of the location at each of these elements. For the actual state
            dataset, this is stored according to its name, so for `x` the values
            are stored in data.x and the associated data names in name.x.
        
        State datasets must represent at least the mean and standard deviation
            of a state for them to be of value in EOLDAS. TThe mean is accessed
            as e.g. self.state for the state dataset. 
            The sd is accessed can be accessed as self._state.sd if it 
            has been set. This reference can also be used to directly set 
            data associated with a SpecialVariable, e.g.
            self.Data.control = np.zeros([2,3])
            
            to represent 2 samples with 3 control variables. You can access name
            information similarly with
            
            print self.Name.control
            
            but this will generate a KeyError if the term has not been set. You
            can check it exists with:
            
            key = 'control'
            if key in self.Name:
                this = (self.Data[key],self.Name[key])
            
            To get e.g. a dictionary representation of a SpecialVariable
            you can use eg:
            
                self.Name.to_dict()
            
            to get the name dictionary, or
            
                thisdict = self._state.to_dict()
            
            to get the full representation, which then contains 'data' and
            'name' as well as some other information stored in the 
            SpecialVariable.
            
            You can similarly load them using e.g. 
            
                self.Data.update(
                        ParamStorage().from_dict(thisdict['data'])
                        combine=True)
        
        '''

        # set up a fakes dictionary from the data types
        self.set('datatype',datatype)
        self.set('fakes', {'state':'_state'})
        
        # first check that options is sensible
        self.__check_type(options,ParamStorage,fatal=True)
        self.options = options
       
        from eoldas_Lib import set_default_limits,\
                    check_limits_valid,quantize_location, sortopt
   
        nSpecial = 1
        if name == None:
           import time
           thistime = str(time.time())
           name = type(self).__name__
           name =  "%s.%s" % (name,thistime)
        self.thisname = name
        self.options.thisname = str(name).replace(' ','_')
        

        log_terms = {\
            'logfile':logfile or sortopt(self.options,'logfile',None),\
            'logdir':logdir or sortopt(self.options,'logdir',None),\
            'debug' : debug or sortopt(self.options,'debug',True)}
        
        self.datadir = datadir or sortopt(self.options,'datadir',["."])
        self.header = header or "EOLDAS pickle V1.0 - plewis"
        env = env or sortopt(self.options,'env',None)
        names = names or sortopt(self.options,'names',None)
        location = location or sortopt(self.options,'location',['time'])
        control = control or sortopt(self.options,'control',[])
        limits = limits or sortopt(self.options,'limits',\
                        set_default_limits(np.array(location)))
        limits = limits or self.options.limits
        limits = np.array(check_limits_valid(limits))
        bounds = bounds or sortopt(self.options,'bounds',\
                    [[None,None]] * xlen(names))
        self.options.bounds = bounds
       
        self.headers = {'PARAMETERS-V2':"PARAMETERS-V2", \
            'PARAMETERS':"PARAMETERS", \
            'BRDF-UCL':'BRDF-UCL',\
            'BRDF': 'BRDF'}
        self.headers_2 = {'BRDF-UCL':'location'}

        #  The ones pre-loaded are
        # self.read_functions = [self.read_pickle,self.read_numpy_fromfile]
        self._state = SpecialVariable(info=info,name=self.thisname,\
                                  readers=readers,datadir=self.datadir,\
                                  env=env,writers=writers,\
                                  header=self.header,\
                                  logger=logger,log_terms=log_terms,\
                                  simple=False)
        # self._state is where data are read into
        # but self.Data and self.Name are where we access them from
        self.grid=grid
        # this is so we can access this object from
        # inside a SpecialVariable
        self.state = np.array([0.])
        # a default data fmt output     
        if datatype[0] == 'y':
            self.Name.fmt = 'BRDF'
            self.Name.state = np.array(['dummy'])
        else:
            self.Name.fmt = 'PARAMETERS'
            n_params = xlen(names)
            if not n_params:
                error_msg = \
                    "The field 'names' must be defined in options or"+ \
                    "passed directly to this method if you have the data type x"
                raise Exception(error_msg)
        self.Name.state = np.array(names)   
        self.Name.location = np.array(location)
        self.Name.control = np.array(control)
        self.Name.header = self.header
        self.Name.bounds = np.array(bounds)
        self.Name.qlocation = np.array(limits)
        self.Name.datadir = datadir
        #
        # sort this object's name 
        # sort logging
        self.logger = sortlog(self,log_terms['logfile'],logger,name=self.thisname,
			logdir=log_terms['logdir'],debug=log_terms['debug'])
        self.logger.info('Initialising %s' % type(self).__name__)


    get = lambda self,this :ParamStorage.__getattr__(self,this)
    get.__name__ = 'get'
    get.__doc__ = '''
        An alternative interface to get the value of a class member
        that by-passes any more complex mechanisms. This returns the 'true'
        value of a class member, as opposed to an interpreted value.
        '''
            
    set = lambda self,this,that :ParamStorage.__setattr__(self,this,that)
    set.__name__ = 'set'
    set.__doc__ = '''
        An alternative interface to set the value of a class member
        that by-passes any more complex mechanisms. This sets the 'true'
        value of a class member, as opposed to an interpreted value.
        '''
           
    def __set_if_unset(self,name,value):
        '''
            A utility to check if the requested attribute
            is not currently set, and to set it if so.
            '''
        if name in self.fakes.keys():
            fname = self.fakes[name]
            if not fname in self.__dict__:
                ParamStorage.__setattr__(self,fname,value)
                return True
        else:
            if not name in self.__dict__:  
                ParamStorage.__setattr__(self,name,value)
                return True
        return False
    
    
    def __getattr__(self,name):
        '''
            get attribute, e.g. return self.state
            '''
        return self.__getitem__(name)
    
    def __setattr__(self,name,value):
        '''
            set attribute, e.g. self.state = 3
            '''
        if not self.__set_if_unset(name,value):
            self.__setitem__(name,value,nocheck=True)
    
    def __getitem__(self,name):
        '''
            get item for class, e.g. x = self['state']
            '''
        if name in ['Data','Name']:
            return self._state[name.lower()]
        elif name in ['Control','Location']:
            return self._state[name.lower()]
        elif name == 'state':
            return SpecialVariable.__getattr__(self._state,name)
            #return super( State, self ).__getattr__(name )
        else:
            return self.__dict__.__getitem__ ( name )
    
    def __setitem__(self,name,value,nocheck=False):
        '''
            set item for class e.g. self['state'] = 3
            '''
        if nocheck or not self.__set_if_unset(name,value):
            if name in ['Data','Name']:
                self._state[name.lower()] = value
            elif name in ['Control','Location']:
                self._state[name.lower()] = value
            elif name == 'state':
                # NB 'self' during the __setitem__ call
                # will be self._state as far as we are 
                # concerned here
                try:
                    self._state.name.datatype = self.datatype
                except:
                    pass
                SpecialVariable.__setattr__(self._state,name,value)
                #super( State, self ).__setattr__(name,value)
                # apply any overrides from options
                self.apply_defaults(self._state,self.options)
                if self.grid:
                    if 'location' in self.Name.dict().keys():
                        self.regrid()
            else:
                ParamStorage.__setattr__(self,name,value)
    
    def apply_defaults(self,this,options):
        datatypes = np.array(options.datatypes).flatten()  
        for i in datatypes:
            try:
                this.data.sd = np.array([float(v) for v in options.sd])
                n_samples = this.data.state.shape[0]
                if (np.array(options.sd)).size == 1:
                    this.data.sd = this.data.state*0. + np.array(options.sd)[0]
                elif this.data.sd.size != this.data.state.size:
                    this.data.sd = np.tile(np.array(options.sd),n_samples).\
                        reshape(this.data.state.shape)
            except:
                pass
            try:
                default = options[i].default
                options[i].sddefault = np.array(options.sd).flatten()
                for jj in xrange(this.data.state.shape[1]):
                    ww = np.where(np.array([np.isnan(i) for \
                                   i in this.data.state[:,jj].astype(float)]))
                    this.data.state[ww,jj] = default[jj]
            except:
                pass  
              
    def ungrid(self,state,sd):
	'''
	Utility to take a gridded dataset that has been gridded using 
	self.regrid() and ungrid it.

	The ungridding is applied to self.x.state and self.x.sd

	Locational information is formed in
	 	self.Data.ungridder
		self.Names.ungridder

	returns:
		(locations,qlocations,state,sd)	

	where:
		state 	: state array
		sd    	: sd array
	        location: location
                qlocation:quantised location data 
		
	'''
	try:
	    qlocations = self.ungridder.qlocation
            locations = self.ungridder.location
            nloc = self.ungridder.nloc
        except:
            raise Exception("You are trying to ungrid a dataset that wasn't gridded using State.regrid()" +\
                " so the ungridder information is not available. Either load the data using State.grid " +\
                " or set it up some other way or avoid calling this method with this type of data")
        h0 = [qlocations[...,0].flatten()]
        for i in xrange(1,nloc):
            h1 = qlocations[...,i].flatten()
            h0.append(h1)
        qlocations = np.array(h0).T
        h0 = [locations[...,0].flatten()]
        for i in xrange(1,nloc):
            h1 = locations[...,i].flatten()
            h0.append(h1)
        locations = np.array(h0).T
        h0 = [state[...,0].flatten()]
        for i in xrange(1,state.shape[-1]):
            h1 = state[...,i].flatten()
            h0.append(h1)
        state = np.array(h0).T
        h0 = [sd[...,0].flatten()]
        for i in xrange(1,sd.shape[-1]):
            h1 = sd[...,i].flatten()
            h0.append(h1)
        sd  = np.array(h0).T
        return locations,qlocations,state,sd

 
    def regrid(self):
        '''
        Utility to regrid non-gridded state (& associated) data 
            
        If no data are specified (we can see this because
        self.Data.qlocation doesnt exist) then a default grid
        is generated.
            
        Outputs:
            self.Name.gridder
            self.Data.gridder
            
        containing:
            grid       : state vector grid (offset in name)
            sdgrid     : state vector sd grid (offset in name)
            ngrid      : n samples from ip data for grids
            wheregrid  : where grid points are data points
            
        '''

        try: 
            qlocation = self.Data.qlocation
            has_data = True
            self.logger.info("Looking at loading a grid")
        except:
            nd = len(self.Name.qlocation)
            qlocation = np.zeros((2,nd))
            for i in xrange(nd):
                qlocation[:,i] = self.Name.qlocation[0][0:2]
            # no data defined, so set up a default grid
            has_data = False
            self.logger.info("No input data given: using default grid")
        if type(self.Data.state) == str:
	    # somehow the magic reader hasnt worked      
 	    raise Exception("Datafile %s hasn't been interpreted as state data"\
							%self.Data.state)  
        if self.Data.state.size == 1 and self.Data.state == np.array(None):
            has_data = False
                
        datatype = self.options.datatype
        if datatype == 'y':
            self.logger.info("Not loading grid as datatype is y")
            return
        try:
            if not self.options[datatype].apply_grid:
                self.logger.info("Not loading grid as apply_grid not set")
                return
        except:
            return 
        try:
            default = self.options[datatype].default
        except:
            default = np.zeros_like(self.Name.state).astype(float)
        try:
            sddefault = self.options[datatype].sddefault
        except:
            sddefault = list(np.array(default)*0.)

        self.ungridder = ParamStorage()


        nd = qlocation.shape[1]
        limits = self.Name.qlocation
        nparams = len(default)
        x = []
        minx = []
        for i in xrange(nd):
            xmin = qlocation[:,i].min()
            xmax = qlocation[:,i].max()
            lim = limits[i]
            if lim[0] != None:
                xmin = lim[0]
            if lim[1] != None:
                xmax = lim[1] 
            x.append(xmax-xmin+1)
            minx.append(xmin)
        self.Name.qlocation_min = minx   
        # x contains the number of desired samples 
        # in each dimension
        x.append(nparams)
        minx.append(0)
        # now loop over all observations and place in grid
        ntot = np.array(x).prod()
        grid = np.zeros(ntot).reshape(tuple(x))
        sdgrid = np.zeros(ntot).reshape(tuple(x))
        ngrid = np.zeros(ntot/x[-1],dtype=int).reshape(tuple(x[:nd]))
        # now fill the grid 
        all = ':,'
        for i in xrange(1,nd):
            all = '%s:,'%all
        for i in xrange(nparams):
            if i >= len(default):
                self.logger.error("Incorrect length for default")
            if i >= len(sddefault):
                self.logger.error("Incorrect length for sd")
            exec('grid[%s%d] = default[%d]'%(all,i,i))
            exec('sdgrid[%s%d] = sddefault[%d]'%(all,i,i))
        if has_data:
            for i in xrange(qlocation.shape[0]):
                loc = tuple(qlocation[i,:]) #-minx[:nd])
                ngrid[loc] += 1
                thisdata = self.Data.state[i]
                if ngrid[loc] == 1:
                    grid[loc][:] = thisdata
                else:
                    grid[loc][:] += thisdata
        wheregrid = np.where(ngrid>0)
        self.Name.gridder = ParamStorage()
        self.Name.gridder.nd = len(self.Name.qlocation)
        self.Data.gridder = ParamStorage()
        self.Data.gridder.grid = grid
        self.Data.gridder.ngrid = ngrid
        self.Data.gridder.sdgrid = sdgrid
        self.Data.gridder.wheregrid = wheregrid
        self.Name.gridder.grid = minx
        self.Name.gridder.sdgrid = minx
        self.Name.gridder.wheregrid = wheregrid[0].size
        self.Data.gridder.ngrid = x

	# now form the information needed for ungridding
	state = grid
        nloc = len(state.shape)-1
        ss = np.array(state.shape)
        ss[-1] = nloc
        ss = tuple(ss)
        qlocation_min = minx

        locations = np.zeros(ss)
        qlocations = np.zeros(ss,dtype=int)
        for i in xrange(nloc):
            aa = np.zeros(nloc,dtype=object)
            for jj in xrange(ss[i]):
                if i == 0:
                    qlocations[jj,...,i] = jj
                    locations[jj,...,i] = jj + qlocation_min[i]
                elif i == 1:
                    qlocations[:,jj,...,i] = jj
                    locations[:,jj,...,i] = jj  + qlocation_min[i]
                elif i == 2:
                    qlocations[:,:,jj,...,i] = jj
                    locations[:,:,jj,...,i] = jj + qlocation_min[i]
                elif i == 3:
                    qlocations[:,:,:,jj,...,i] = jj
                    locations[:,:,:,jj,...,i] = jj + qlocation_min[i]
                elif i == 4:
                    qlocations[:,:,:,:,jj,...,i] = jj
                    locations[:,:,:,:,jj,...,i] = jj + qlocation_min[i]
                else:
                    raise Exception('How many dimensions in your dataset ??? > 4 ??? thats ridiculous'+\
                                " ... I can't write that ")
	self.ungridder.qlocation = qlocations
	self.ungridder.location = locations
	self.ungridder.nloc = nloc
	
	                    
    
    def tester(self):
        '''
            Run a test using  all methods to check competence
        '''
        print '== Data type:',self.datatype
        print '==============='
        print "  name: state"
        print "  fmt: %s:"%self.Name.fmt
        state = self.state
        print "  n_samples: %d"%xlen(state)
        print "data:"
        if xlen(state):
            print state
        else:
            print 'not set'

        # look for items that appear in both name and data
        name = self.Name
        data = self.Data
        for i in name.dict():
            if i != 'state' and i in data.dict():
                print "Sub Dataset: %s"%i
                print "----------------"
                print name[i]
                print "----------------"
                print data[i]
          
    def get_dimension_span(self):
        '''
        Return the number of unique samples in each dimension
        '''
        retval = {}
        locations = self.Data.location
        for i in xrange(self.Name.location):
            n = xlen(np.unique(data[i]))
            if n == 1:
                n = 0
            retval[i] = n
        return retval

    def __check_type(self,this,thistype,fatal=False,where=None):
        if type(thistype) != type:
            thistype = type(thistype)
        if type(this) != thistype:
            self.error = True
            self.error_msg = "Unexpected type %s to variable " % \
                type(this).__name__
            if where != None:
                self.error_msg = self.error_msg + " " + str(where)
            self.error_msg = self.error_msg +  ": should be type %s" \
                % thistype
            if self.logger:
                self.logger.error("Error %s: "%str(self.error),self.error_msg)
            if fatal:
                raise Exception(self.error_msg)
            return not self.error
        self.error = False
        return not self.error

    def __name_guess_suffix(self):
        '''
        If no name is assigned, we get the default name from
        self.getname() and add this string on the back of it.

        Here, it is defined as a time string for now.
        '''
        import time
        # need to make one up
        return str(time.time())

    def __logger(self,x):
        if self.log:
            print x

    def getname(self):
        '''
        Return the name associated with this class ("eoldas") or 
        self.options.thisname
        '''
        # must be better way of getting this ... leave for later
        if not self.__pcheck(self.options,"thisname"):
            return type(self).__name__
        else:
            return self.options.thisname

    __pcheck = lambda self,this,name : this.dict().has_key(name)
   
    def startlog(self,log_terms,name=None):
        '''
        Start the logger.

        This is called on initialisation and you shouldn't 
            normally need to access it.
        '''
        import logging
        from eoldas_Lib import set_up_logfile
        try:
           self.logger.shutdown()
        except:
            self.logger = ParamStorage ()
        logfile = log_terms['logfile'] or self.options.logfile
        logdir = log_terms['logdir'] or self.options.logdir
        name = name or self.options.thisname
        self.logger = set_up_logfile(logfile,\
                                     name=name,logdir=logdir)
         
    def write(self,filename,dataset,fmt='pickle'):
        '''
        A state data write method
        '''
        writer(self,filename,dataset,fmt=fmt)   

def xlen(x):
    t = type(x)
    if t == list:
        return len(x)
    elif t == np.ndarray:
        this=x.shape
        return len(this)== 0 or x.shape[0]
    else:
        return 1

def demonstration():
    from eoldas_State import State
    from eoldas_ParamStorage import ParamStorage
    import numpy as np
    
    # a basic set up for State, setting names & bounds etc
    options = ParamStorage()
    options.logfile = 'test/data_type/logs/log.dat'
    options.names = \
       'gamma xlai xhc rpl xkab scen xkw xkm xleafn xs1 xs2 xs3 xs4 lad'.split()
    options.bounds = [[0.01,None],\
                      [0.01,0.99],\
                      [0.01,10.0],\
                      [0.001,0.10],\
                      [0.1,0.99],\
                      [0.0,1.0],\
                      [0.01,0.99],\
                      [0.3,0.9],\
                      [0.9,2.5],\
                      [0.0, 4.],\
                      [0.0, 5.],\
                      [None, None],\
                      [None, None],\
                      [None, None]]
    options.default = -1.0*np.ones(len(options.names))
    options.location = 'time'.split()
    options.control = 'mask vza vaa sza saa'.split()
    options.datadir = ['.','test/data_type']

    name = "eoldas_data_type test 0"
    options.limits = [[170,365,1]]

    self = State(options,datatype='y',name=name,datadir=\
                             options.datadir,env=None,logfile=options.logfile)
    self.tester()
            
    # Now we set some state data        
    this = ParamStorage()
    # how many state vector elements should there be?
    n_states = len(self.Name.state)       
    self.state  = np.ones([100,n_states])
    self.Data.sd = np.ones([1,n_states])
    self.Name.sd = self.Name.state
  
    print '******************'
    print (self.Data.sd,self.Name.sd)

    
    this.data = ParamStorage() 
    this.name = ParamStorage() 
    this.data.state = self.state *2
    controls = self.Name.control
    n_controls = len(controls)
    this.data.control = np.ones([100,n_controls])
    this.data.location = np.ones([100,n_controls])

    # we can load x_state from a ParamStorage
    self.state = this
    # now we should see the control data etc.
    self.tester()

    # change a dataset name to see if that works:
    # should load everything as numpy arrays
           
    self.Name.control = np.array(['vza','sza'])

    # change a dataset to see if that works .. deliberately load a bad one
    self.Data.control = 0

    # now try a direct state data load
    self.state = np.zeros([100,100])+5.
    # which will load into self.Data.state

    self['state'] = np.zeros([100,100])+6.
    #now try accessing it:
    print self.state

    # change the control info
    self.Name.control = np.array(['vza'])
    print self.Name.control

    # reset it 
    self.state = self.state * 2
    print self.state
    # now try reading a file into state
    self.state = 'test/data_type/input/test.brf'
    print '========'
    print 'data from a BRDF file'
    self.tester()
    print '========'
    # written as a pickle
    self.write('test/data_type/output/test.pickle',None,fmt='pickle')
    self.logger.info("...DONE...") 
    name = "eoldas_data_type test 1"
    del self
    self1 = State(options,datatype='y',name=name,datadir=\
             options.datadir,env=None,logfile=options.logfile,grid=True)
    # read from pickle
    self1.state = 'test/data_type/output/test.pickle'
    print '========'
    print 'data from a pickle file'
    self1.tester()
    print '========'
    # try to load an npx file
    del self1
    options.location = 'time row col'.split()
    options.limits = [[170,365,1],[0,500,1],[200,200,1]]

    self2 = State(options,datatype='y',name=name,datadir=\
                  options.datadir,env=None,logfile=options.logfile)
    self2.state = 'test/data_type/input/interpolated_data.npz'
    print '========'
    print 'data from a npz file'
    self2.tester()
    print '========'
    # write as BRDF-UCL
    self2.write('test/data_type/output/test.brf',None,fmt='BRDF-UCL')
    del self2
    self3 = State(options,datatype='y',name=name,datadir=\
             options.datadir,env=None,logfile=options.logfile)
    # then test the reader
    self3.state = 0.
    self3.state = 'test/data_type/output/test.brf'
    print '========'
    print 'data from a BRDF-UCL file'
    print '========'
    self3.tester()
    print '========'
    # then write as a PARAMETERS file
    self3.write('test/data_type/output/test.param',None,fmt='PARAMETERS')
    del self3
    options.location = 'time row col'.split()
    options.limits = [[170,365,1],[0,500,1],[200,200,1]]
    options.control = np.array(['mask','vza','vaa','sza','saa']) 
    self4 = State(options,datatype='y',name=name,datadir=\
             options.datadir,env=None,logfile=options.logfile)
    # then test the reader
    self4.state = 0.
    self4.state = 'test/data_type/output/test.param'
    print '========'
    print 'data from a PARAMETERS file'
    print '========'
    self4.tester()
    print '========'



if __name__ == "__main__":
    from eoldas_State import State
    help(State)
    demonstration()
