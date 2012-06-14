#!/usr/bin/env python
from eoldas_ParamStorage import ParamStorage

import pdb
import numpy as np

def init_read_write(self,header,readers,writers):
    self.header = header or "EOLDAS -- plewis -- UCL -- V0.1"
    self.write_functions = writers

    # a set of asimple ASCII formats that are quite similar
    # and can be done with the same writers and readers

    self.headers = {'PARAMETERS-V2':"PARAMETERS-V2", \
        'PARAMETERS':"PARAMETERS", \
        'BRDF-UCL':'BRDF-UCL',\
        'BRDF': 'BRDF'}
    self.headers_2 = {'BRDF-UCL':'location'}

    for i in self.headers:
        self.write_functions[i] = 'write_output_file'

    self.write_functions['pickle'] = 'write_pickle'

    self.read_functions = readers
    self.read_functions.append('read_pickle')
    self.read_functions.append('read_input_file')
    self.read_functions.append('read_numpy')

    # get only unique ones
    self.read_functions = list(np.unique(np.array(self.read_functions)))
    self.write_functions = list(np.unique(np.array(self.write_functions)))
    self.read_functions.append('read_numpy_fromfile')

def writer(self,filename,dataset,fmt='pickle',info=[]):
    '''
    Write state data to a file

    fmt controls the putput format
    '''
    import os,errno
    from os import makedirs
    from os.path import dirname
    from eoldas_Files import write_pickle, write_output_file
    # just do a pickle dump
    try:
        if not 'write_functions' in self.dict():
            self.write_functions = self._state.write_functions
    except:
        try:
            self.logger.error('Error in calling writer: no write_functions in self')
            return
        except:
            print 'Error in calling writer: no write_functions in self'
            return
    # first confirm the path is writeable
    dd = os.path.dirname(filename)
    try:
	if dd != '':
            os.makedirs(os.path.dirname(filename))
    except OSError as exc:
	if exc.errno == errno.EEXIST:
	    pass
	else:
	    raise
        
    if fmt in self.write_functions[0]:
        eval('%s'%self.write_functions[0][fmt])(self,filename,dataset,info=info)
    else:
        write_pickle(self,filename,dataset,info=None)


def reader(self,filename,dataset,fmt=None,info=[]):
    '''
    An interface to a set of utilities for reading files
    for EOLDAS.

    Inputs:
        self     : ParamStorage that may contain information
                   where the data are to be read into
        filename : the name of a file to read
        dataset  : a tag for a particular dataset to read into
        
    Options:
        fmt 		: string
        info		: list

    Output:
        stuff, is_error


    The interface to all reading functions is:

    result,(error,error_msg)  = reader_function(self,filename,dataset,info=[])

    If fmt is specified and in the format dictionar, an attempt 
    is made to read a specific format.

    If fmt is not set, an attempt to make a guess at the file format 
    is made, but other than that, the reader just goes through all
    available formats until a suitable one is found. 

    where:

        self:
	    is a dictionary or ParamStorage or derived class instance 
            into which the data will be loaded.
        filename:
            is a string containing the name of a file to read. 
            All readers are designed to be quite tolerant if there
            are some issues, but the file should at least exist and ideally
            contain information in an interprable format.
        dataset:
	    is a string containing the name of a particular dataset (within `data`)
            to target for writing to the file. This may be ignored by some formats that
            write comprehensive information (such as the pickle dump).

        info:
	    a list, containing any other information the user might want to pass
            through to a reader.
               
    Readers available here include:

       fmt == 'pickle'	:	read_pickle()   
			
           Read an EOLDAS pickle dump of a state.
           A very general format maintaining full information about the state.

       fmt == 'NpzFile'	:	read_numpy_fromfile():
           Read an EOLDAS binary data format for spatio-temporal datasets (e.g. MODIS)
           Used for largish BRDF datasets. This could be more general, but provides
           a link through to e.g. MODIS data at present.

       fmt == 'BRDF' or 'BRDF-UCL' or 'PARAMETERS' or 'PARAMETERS-V2'
			:	read_input_file()
           An interface to a set of ASCII formats. They all have an ASCII header
           with some metadata and columns of datasets. The formats PARAMETERS
           and PARAMETERS-V2 are the most general  

       read_numpy()
           a generic 'read' utility from numpy. A last resort attemt at reading if no 
           other format works. 

    '''
    stuff = None
    if 'logger' in self or 'logger' in self.dict():
        self.logger.debug('try to read file %s'%filename)
    # loop over read functions
    # have to use try in case of major
    # failure in readers
    if fmt != None:
        stuff,is_error = try_read(self,this,filename,dataset,info=info)
        if not is_error[0]: 
            return stuff,(False,"")
    # if not, then try othr formats
    for this in self.read_functions:
	self.logger.warning("Unable to read %s with standard readers"%filename)
	self.logger.warning("Will try to interpret as flat ASCII file")
        stuff,is_error = try_read(self,this,filename,dataset,info=info)
	# check teh sanity of the read
        err = False
        try:
	    nstates = len(np.atleast_1d(self.name.state))
        except:
            nstates = 1
            err = True
        try:
	    if self.data.state.shape[1] != nstates:
                err = True
        except:
            err = True
        if err:
            try: 
	        self.logger.warning("*****************************************")
	        self.logger.warning("MAJOR WARNING")
                self.logger.warning("*****************************************")
    	        self.logger.warning("Inconsistency found in data read")
	        self.logger.warning("You requested %s"%str(self.name.state))
	        self.logger.warning("but we find %d columns of data"%self.data.state.shape[1])
	        self.logger.warning("We will guess that last columns are the one wanted")
	        self.logger.warning("but this may not work")
	        self.logger.warning("You should check the state names in the config file")
	        self.logger.warning("and compare that with what you have in the datafile")
	        self.data.state = self.data.state[:,-nstates]
            except:
                self.logger.warning("error logging warning message")
        if not is_error[0]:
            return stuff,(False,"")
    return stuff,(True,"Exhausted formats for file %s"%(filename))

def try_read(self,this,filename,dataset,info=[]):
    '''
    Attempt a file read using read function this
    '''
    exec('from eoldas_Files import %s'%this)
    if 'logger' in self or 'logger' in self.dict():
        self.logger.debug('format: %s'%this)
    is_error = (True,"Something went awry")
    try:
        stuff,is_error = eval(this)(self,filename,dataset,info=info)
    except:
        pass
    if not is_error[0]:
        # read ok
        stuff_type = type(stuff)
        if stuff_type != str:
            self.__setattr__(dataset,stuff)
            return stuff,(False,"")
    return None,(True,"") 
 
def read_pickle(self,filename,dataset,info=None):
    '''
    Read pickle file 'filename' into dataset 'dataset'
        
    '''
    error = False
    error_msg = ""
    if 'logger' in self or 'logger' in self.dict():
        self.logger.info('Attempting to read pickle file %s'%filename)
    this = self.from_pickle(filename,header=self.header)
    if this == None:
        error = True
        error_msg = "Failure reading pickle file %s"% filename
    return this,(error,error_msg)

def write_pickle(self,filename,dataset,info=None):
    '''
    Write pickle file into 'filename' from dataset 'dataset'
    '''
    error = False
    error_msg = ""
    if 'logger' in self or 'logger' in self.dict():
        self.logger.info('Attempting to write pickle file %s'%filename)
    this = self.to_pickle(filename,header=self.header)
    if this == None:
        error = True
        error_msg = "Failure writing pickle file %s"% filename
    return this,(error,error_msg)       

def read_numpy_fromfile(self,goodfile,dataset,info=None):
    '''
    Utility to try to read a file goodfile

    This simple version of the utility just tries a numpy.fromfile

    It is not generally recommended to use this first
    as it will skip headers etc that may contain
    information

    However, this is a good illustration of the required interface format
    for file readers.


    Inputs:
        goodfile  : a filename (ideally one that exists)
        dataset   : dataset name (e.g. x_state)
        info      : a list of other information

    Outputs:
        retval    : ParamStorage
        error     : tuple

    where:
         retval contains dataset 'this' in data['this']
         and other information (e.g. locations etc) in
         retval.data and retval.names

    '''
    #self.store_header = goodfile.open().readline().close()
    this = np.genfromtxt(goodfile,skip_header=1)
    l = len(np.atleast_1d(this).shape)
    if this.size > 0:
        retval = ParamStorage()
        retval.name = ParamStorage()
        retval.data = ParamStorage()
        retval.data[dataset] = this
        retval.name.filename = goodfile
        retval.name.fmt = 'np.genfromtxt'
        return retval,(False,"")
    error = True
    error_msg = "Failed numpy read of %s"%goodfile
    return 0,(error,error_msg)
 
def read_numpy(self,filename,name,info=[]):
    '''
        Try to read the file as as a NpzFile file
        '''
    from eoldas_Lib import set_default_limits,check_limits_valid,\
        quantize_location,dequantize_location
    
    # none of these ciritical to functioning
    try:
        info = self._state.info
    except:
        info = []
    try:
        names = self.name.state
    except:
        try:
            names = self.Name.state
        except:
            names = None
    try:
        control = self.Name.control
    except:
        try:
            control = self.name.control
        except:
            control = None
    try:
        location = self.name.location
    except:
        try:
            location = self.Name.location
        except:
            location = ['time','row','col']
    try:
        limits = self.name.qlocation
    except:
        try:
            limits = self.Name.qlocation
        except:
            limits = set_default_limits(location)
    # refl_check=False,names=None,\
    # control=['mask','vza','vaa','sza','saa'],\
    # location=['time','row','col'],limits=None
    
    # location specifies the dimesions and names of the
    # problem, e.g., & typically [time,row,col]
    limits = np.array(check_limits_valid(limits))
    
    try:
        f = np.load(filename)
        if not type(f).__name__ == 'NpzFile':
            f.close()
            self.error_msg="%s is not a NpzFile"%filename
            self.error=True
            if 'logger' in self or 'logger' in self.dict():
                self.logger.info(self.error_msg)
            return 0,(self.error,self.error_msg)
    except:
        self.error_msg="a problem opening %s as a NpzFile"%filename
        self.error=True
        if 'logger' in self or 'logger' in self.dict():
            self.logger.info(self.error_msg)
        return 0,(self.error,self.error_msg)
    # ok so far then
    # lets have a look inside
    
    ncontents = np.array(f.files)  
    contents = np.array(f.files)
    # translation table for default names
    def_names = 'b1 b2 b3 b4 b5 b6 b7'.split()
    if names == None:
        # assume MODIS
        names = def_names
    def_alt_names = \
        '645.5 856.5 465.6 553.6 1241.6 1629.1 2114.1'.split()
    # look for any of names in contents
    datasets = []
    alt_datasets = []
    alt_names = names
    for i in xrange(len(np.atleast_1d(contents))):
        if contents[i] in names:
            datasets.append(i)
    
    if not len(np.atleast_1d(datasets)):
        if 'logger' in self or 'logger' in self.dict():
            self.logger.error(\
                          "None of requested datasets %s found in %s ..." \
                          %(str(names),filename) + \
                          " trying default MODIS names: only %s"\
                          %(str(contents)))
        names = def_names
        alt_names = def_alt_names
        for i in xrange(len(np.atleast_1d(contents))):
            if contents[i] in names:
                datasets.append(i)
        if not len(np.atleast_1d(datasets)):
            self.error_msg = "None of requested datasets %s found in %s"\
                %(str(names),filename) + ' ' + \
                "... trying default MODIS names: only %s"\
                %(str(contents))
            self.error = True
            if 'logger' in self or 'logger' in self.dict():
                self.logger.error(self.error_msg)
            return 0,(self.error,self.error_msg)
    trans_names = {}
    for (i,j) in enumerate(alt_names):
        trans_names[names[i]] = j
#trans_names = {names[i]:j for (i,j) in enumerate(alt_names)}
    alt_name = []
    this_name = []
    for i in datasets:
        this_name.append(contents[i])
        alt_name.append(trans_names[contents[i]])
    
    # Translate  some old stylies...
    trans = {'raa':'vaa','doys':'time'}
    for i in trans:
        if i in contents:
            ncontents[np.where(contents==i)[0]]=trans[i]
    # as a minimum, there needs to be some definition of one of
    # the terms in location

    # check how many dimensions this has
    # now find a dataset
    try:
        # This could be more general, but this will do for now as its useful
        # for spatial datasets
        QA_OK = np.array(\
                         [8, 72, 136, 200, 1032, 1288, 2056,2120, 2184, 2248])
        doy = f['doys'] - 2004000
        qa = f['qa']
        vza = f['vza']
        sza = f['sza']
        raa = f['raa']
        y = []
        for i in this_name:
            y.append(f[i])
        #mask = np.logical_or.reduce([qa==x for x in QA_OK ])
        if 'logger' in self or 'logger' in self.dict():
            self.logger.info(\
                         "sucessfully interpreted NpzFile dataset from %s"\
                             %filename)
            self.logger.info("sub-setting ...")
        controls = []
        locations = []
        grid = []
        qlocations = []
        thisshape = vza.shape
        starter = {'time':np.min(doy),'row':0,'col':0}
        delta = {'time':1,'row':1,'col':1}
        if len(np.atleast_1d(limits)) <3:
            from eoldas_Lib import set_default_limits
            old_loc = location
            location = np.array(['time','row','col'])
            lim2 = set_default_limits(location)
            for i in xrange(len(np.atleast_1d(limits))):
                ww = np.where(old_loc[i] == location)[0]
                lim2[ww] = list(limits[i])
            limits = lim2
        for i in xrange(len(np.atleast_1d(limits))):
            if limits[i][0] == None:
                limits[i][0] = starter[location[i]]
            if limits[i][1] == None:
                limits[i][1] = (thisshape[i]-1) + starter[location[i]]
            if limits[i][2] == None:
                limits[i][2]= delta[location[i]]
        limits = np.array(limits)
        start_doy = limits[0][0]
        end_doy =   limits[0][1]
        step_doy =  limits[0][2]
        start_row = limits[1][0]
        end_row =   limits[1][1]
        step_row =  limits[1][2]
        start_col = limits[2][0]
        end_col =   limits[2][1]
        step_col =  limits[2][2]
        gooddays = np.logical_and.reduce(np.concatenate(\
                            ([doy >= start_doy],[doy  <=end_doy])))
        qa = qa[gooddays,start_row:end_row+1,start_col:end_col+1]
        vza = vza[gooddays,start_row:end_row+1,start_col:end_col+1]*0.01
        sza = sza[gooddays,start_row:end_row+1,start_col:end_col+1]*0.01
        raa = raa[gooddays,start_row:end_row+1,start_col:end_col+1]*0.01
        yy = []
        for i in xrange(len(np.atleast_1d(this_name))):
            this = y[i]
            yy.append(this[gooddays,start_row:end_row+1,\
                           start_col:end_col+1]*0.0001)
        doy = doy[gooddays]
        # now do QA
        mask = np.zeros_like(qa).astype(bool)
        # loop over qa
        for j in xrange(len(np.atleast_1d(QA_OK))):
            ww = np.where(qa==QA_OK[j])
            mask[ww] = True
        # better look over data to check valid
        for j in xrange(len(np.atleast_1d(yy))):
            ww = np.where(yy[j] < 0)
            mask[ww] = False
        ww = np.where(mask)
        if 'logger' in self or 'logger' in self.dict():
            self.logger.debug('parsing dataset: %d samples look ok'\
                          %np.array(ww).shape[1]) 
        vza = vza[ww]
        sza = sza[ww]
        raa = raa[ww]
        doy= doy[ww[0]]
        row = ww[1]+start_row
        col = ww[2]+start_col
	locations  = np.array([doy,row,col])
	nnn = len(np.atleast_1d(locations[0]))
        orig = np.repeat(np.array([start_doy,start_row,start_col]),locations.shape[1]).reshape(locations.shape).T
        div = np.repeat(np.array([step_doy,step_row,step_col]),locations.shape[1]).reshape(locations.shape).T
        qlocations = ((locations.T - orig)/div.astype(float)).astype(int).T
        controls = np.array([np.ones_like(doy).astype(bool),\
                             vza,raa,sza,0*doy])
        y = []
        for i in xrange(len(np.atleast_1d(this_name))):
            this = yy[i]
            y.append(this[ww])
        grid = np.array(y)   
        fmt = 'BRDF-UCL'
        control = ['mask','vza','vaa','sza','saa']
        bands = alt_name
        if not np.array(grid).size:
            if 'logger' in self or 'logger' in self.dict():
                self.logger.error(\
                              "Warning: returning a zero-sized dataset ... "+\
                              " I wouldn;t try to do anything with it")
        # in case we dont have data for all bands
        mask =  np.logical_or.reduce([[this_name[i]==x for x in names] \
                                      for i in xrange(len(np.atleast_1d(this_name)))])
        sd = np.array('0.004 0.015 0.003 0.004 0.013 0.01 0.006'\
                      .split())[mask]
        sd = np.array([float(i) for i in sd.flatten()])\
            .reshape(sd.shape)
        nsamps = grid.shape[1]
        sd = sd.repeat(nsamps).reshape(grid.shape).T
        datasets = ParamStorage()
        datasets.data  = ParamStorage()
        datasets.name  = ParamStorage()
        datasets.name.fmt = fmt
        grid = grid.T
        datasets.data[name] = np.zeros([grid.shape[0],len(np.atleast_1d(names))])\
                                                        .astype(object)
        datasets.data[name][:,:] = None
        for i in xrange(len(np.atleast_1d(this_name))):
            ww = np.where(names == this_name[i])[0][0]
            datasets.data[name][:,ww] = grid[:,i]
        datasets.data.location = np.array(locations).T
        datasets.data.control = np.array(controls).T
        datasets.data.qlocation = np.array(qlocations).T
        datasets.name[name] = np.array(names)
        datasets.name.location = np.array(['time','row','col'])
        datasets.name.control = np.array(control)
        datasets.name.qlocation = limits
        datasets.name.bands = np.array(bands)
        datasets.data.sd = np.zeros([grid.shape[0],len(np.atleast_1d(names))])\
                                                        .astype(object)
        # for i in xrange(grid.shape[0]):
        # datasets.data.sd[i,:] = self.options.sd
        datasets.data.sd[:,:] = None
        for i in xrange(len(np.atleast_1d(this_name))):
            ww = np.where(names == this_name[i])[0][0]
            datasets.data.sd[:,ww] = sd[:,i]
        datasets.name.sd = np.array(names)
        if 'logger' in self or 'logger' in self.dict():
            self.logger.debug('finished parsing dataset')
    except:
        self.error_msg=\
            "a problem processing information from  %s as a NpzFile"\
                %filename
        self.error=True
        if 'logger' in self or 'logger' in self.dict():
            self.logger.info(self.error_msg)
        return 0,(self.error,self.error_msg)
    f.close()
    if 'logger' in self or 'logger' in self.dict():
        self.logger.info('... done')
    self.error=False
    self.error_msg=""
    return datasets,(self.error,self.error_msg)


def read_input_file(self,filename,name,info=[]):
    '''

        Read state data from ASCII filename

        Returns:

        thisdata,is_error

        where:

        is_error = (error,error_msg)

        Load data from a file (e.g. BRDF data or parameters data)

        The file format is flat ASCII with a header, and needs to
        be one of the formats appearing in self.headers

        '''
    from eoldas_Lib import set_default_limits,\
        check_limits_valid,quantize_location, sortopt
    try:
        f = open(filename,'r')
    except:
        return 0,(True,'Failed to open load file %s with call to %s' % \
                  (filename,str('read_input_file')))
    try:
        if f.errors != None:
            error_msg = str(f.errors)
            return 0,(True,error_msg)
    except:
        pass
    # try to read a PARAMETERS file
    find_col = lambda name :np.where(np.array(params) == name)

    # read the first line
    header = f.readline().replace('#','').split()
    MAGIC = header[0]
    found = False
    nl = 0
    for (k,v) in self.headers.iteritems():
        if MAGIC == v:
            found = True
            nl = 1
            basic = header[1:]
            if k in self.headers_2:
                header2 = f.readline().replace('#','').split()
                if header2[0] != self.headers_2[k]:
                    found = False
                else:
                    nl = 2
                    extras = header2[1:]
        if found:
            fmt = k
            break

    if nl == 0:
        f.close()
        return 0,(True,'File %s not recognised by %s'\
                  % (filename,str('read_input_file')))
    if 'logger' in self or 'logger' in self.dict():
        self.logger.info("Interpreted format of %s as %s"%(filename,k))
    f.close()
    f = open(filename,'r')
    [f.readline() for i in xrange(nl)]


    # the limits info is used to only read observations
    # within these limits
    # The size depends on location and should have 3 numbers
    # for each location entry
    try:
        location = self.Name.location
    except:
        try:
	    location = self.name.location
        except:
            if fmt == 'BRDF':
                location = ['time']
            else:
                location = 'time row col'.split()
    location = np.array([i.replace('[','').replace(']','') for i in location])
    try:
        limits = self.name.qlocation
    except:
        limits = set_default_limits(location)
    try:
        names = np.array(self._state.name.state)
    except:
	try:
	    names = np.array(self.name.state)
	except:
	    names = ['default']

    limits = np.array(check_limits_valid(limits))

    sd_params = []
    names = np.atleast_1d(names)
    try:
        for i in xrange(len(names)):
            sd_params.append("sd-%s"%names[i])
    except:
        pass
    sd_params = np.array(sd_params)
    if (fmt == 'BRDF' or fmt == 'BRDF-UCL'):
        # unpack the header
        nbands = int(basic[1])
        bands = basic[2:nbands+2]
        try:
            if self.name.datatype == 'y':
                names = bands
        except:
            names = bands
        sd_params = []
        for i in xrange(len(np.atleast_1d(names))):
            sd_params.append("sd-%s"%names[i])
        sd_params = np.array(sd_params)
        sd = np.zeros(sd_params.shape[0])
        for i in xrange(len(np.atleast_1d(names))):
            this = np.where(np.array(bands) == names[i])[0]
            if this.size:
                sd[i] = float(basic[2+nbands+this[0]])
        #sd = np.array([float(i) for i in basic[2+nbands:]])
        if fmt == 'BRDF-UCL':
            params = extras
        #location = extras
        else:
            params = ['time']
        nlocation = len(np.atleast_1d(params))
        params.extend("mask vza vaa sza saa".split())
        params.extend(bands)
        if fmt == 'BRDF-UCL':
            params.extend(sd_params)
        params = np.array(params)
            #names = bands
    else:
        params = basic
        sd = np.zeros_like(names).astype(float)

    # check to see if any location information given
    # loop over self._state.name.location and see which
    # columns appear in params
    loccols = []
    for i in xrange(len(np.atleast_1d(location))):
        ccc = find_col(location[i])
        if len(np.atleast_1d(ccc)):
            loccols.append(ccc[0])
        else:
            loccols.append(0)
    # now do the same for control
    controlcols = []
    try:
        control=self.name.control
    except:
        try:
            control=self.Name.control
        except:
            control = 'mask vza vaa sza saa'.split()
    try:
        if len(np.atleast_1d(control)) == 0:
            control = np.array("mask".split())
    except:
        if control.size == 0:
            control = np.array("mask".split())
    control = control.reshape(control.size)
    #strip out superflous brackets
    control = np.array([i.replace('[','').replace(']','') for i in control])
    for i in xrange(control.size):
        ccc = find_col(control[i])
        if len(np.atleast_1d(ccc)):
            controlcols.append(ccc[0])
    # if the datatype is y, then we get the names from the file
    # which we suppose by default to be anything
    # other than options & control
    # but first we see if we can find anything defined in names
    # now for the bands
    wnames = [find_col(i) for i in names]
    # and sd
    wsdnames = [find_col(i) for i in sd_params]
    have_names = False
    # check to see if any names data found
    nnames = np.array([np.array(i).size for i in wnames]).sum()
    if nnames ==0 and (self.datatype == None or \
                                 self.datatype[0] == 'y'):
        # we found no names so check datatype is None or y & guess the 
        # names from the params fields that arent used as control or location
        names = []
        sd_params = []
        p_orig = params
        wnames = []
        wsdnames = []
        for i in xrange(len(np.atleast_1d(p_orig))):
            taken = False
            params = control 
            taken = taken or \
                    bool(np.array(find_col(p_orig[i])).flatten().shape[0])
            params = location
            taken = taken or \
                    bool(np.array(find_col(p_orig[i])).flatten().shape[0])
            params = names
            taken = taken or \
                    bool(np.array(find_col(p_orig[i])).flatten().shape[0])
            params = sd_params
            taken = taken or \
                    bool(np.array(find_col(p_orig[i])).flatten().shape[0])
            if not taken:
                names.append(p_orig[i])
                sd_params.append("sd-%s"%p_orig[i])
                params = p_orig
                wnames.append(find_col(names[-1]))
                wsdnames.append(find_col(sd_params[-1]))
        params = p_orig
    data = f.readlines()
    f.close()
    # check to see if there is a mask column
    is_mask = 'mask' in params
    want_mask = True or  'mask' in control

    # so we need a grid to stroe the data
    # [p,t,r,c ...] or similar
    # the total datasize will be len(data) * (len(names) + len(location))
    # set to nan ... but we'll return a mask later
    grid = []
    locations = []
    qlocations = []
    controls = []
    sd2 = []
    maxi = [(limits[i,1]-limits[i,0]*1.)/limits[i,2] for i in xrange(len(np.atleast_1d(limits)))]
    for i in xrange(len(np.atleast_1d(data))):
        ok = True
        liner = data[i].split()
        get_col = lambda index,liner : float(len(np.atleast_1d(index)) and liner[index])
        ldata = []
        for c in xrange(len(np.atleast_1d(location))):
            ldata.append(get_col(loccols[c],liner))
        qldata = quantize_location(ldata,limits)
        if (np.array(qldata) < 0).all() or (maxi - np.array(qldata) < 0).all():
	    ok = False
        cdata = []
        for c in xrange(len(np.atleast_1d(controlcols))):
            if want_mask and not is_mask and \
			(control[c] == 'mask' or control[c] == '[mask]'):
                cdata.append(1)
            else:
                cdata.append(get_col(controlcols[c],liner))
	# check the mask value
        try:
	    if not (want_mask and not is_mask):
	        c = np.where(control=='mask')[0]
	        if c.size == 0: c = np.where(control=='[mask]')[0]
                if c.size == 0:
	 	    ok = True
	        elif int(cdata[c]) != 1:
	            ok = False  
        except:
	    ok = True
            cdata.append(1)  
        if ok:
 	    this = np.zeros(len(np.atleast_1d(names)))
            this[:] = None
            # this will set unread fields to nan
            for (j,k) in enumerate(wnames):
                if np.array(k).size >0:
                    this[j] = float(liner[k[0]])
            that = np.zeros(len(np.atleast_1d(names)))
            that[:] = None
            # this will set unread fields to nan
            for (j,k) in enumerate(wsdnames):
                if np.array(k[0]).shape[0] > 0:
                    that[j] = float(liner[k[0]])
            locations.extend(ldata)
            controls.append(cdata)
            qlocations.append(qldata)
            grid.append(this)
            sd2.append(that)
    # check to see if the sd data are any good
    sd2a = np.array(sd2)
    if sd2a.flatten().sum() > 0:
        sd = sd2a
    n_samples = len(np.atleast_1d(data))
    data = {}
    name = {}
    data['state'] = np.array(grid)
    nsamples = data['state'].shape[0]
    if not 'datatype' in self.name.dict() or self.name.datatype == None or \
                                            self.name.datatype[0] == 'y':
        # its a y or its a bit broken
        name['state'] = np.array(names)
    # note, the state list can be updated
    # by what it finds, but only for y states
    name['fmt'] = fmt
    name['location'] = np.array(location)
    nlocations = name['location'].shape[0]
    data['location'] = np.array(locations).reshape(nsamples,nlocations)
    name['location'] = np.array(location)
    name['qlocation'] = np.array(limits)
    #orig = np.repeat(np.array(name['qlocation'][:,0]),nsamples).reshape(nlocations,nsamples).T
    data['qlocation'] = np.array(qlocations).reshape(nsamples,nlocations) #+ orig
    name['qlocation'] = np.array(limits)
    name['control'] =  np.array(control)
    ncontrol = np.max((1,name['control'].shape[0]))
    if name['control'].shape[0]:
        data['control'] =  np.array(controls).reshape(nsamples,ncontrol)
    else:
        data['control'] =  np.array(controls)
    # only return sd if its > 0
    if sd.size != data['state'].size:
	try:
            sd = np.tile(np.array(sd),nsamples).reshape(data['state'].shape)
	except:
	    self.logger.info("can't tile sd data: %s"%str(sd))
	sd = np.array([0.])
    if sd.flatten().sum() > 0:
        name['sd'] = np.array(names)
        data['sd'] = sd
    datasets = {'data':data,'name':name}
    return datasets,(False,'Data read from %s with %s fmt %s'% \
                     (filename,str('read_input_file'),fmt))

def write_output_file(self,filename,name,info=[]):
    '''

        Attempt to write state data as ASCII to filename

        have a standard interface.

        Returns:
        is_error

        where:
        is_error = (error,error_msg)

        The file format is flat ASCII with a header

        '''
    #First check the data is as intended
    from eoldas_Lib import sortopt
    try:
        f = open(filename,'w')
        if self.Data.sd != None:
            sd = self.Data.sd.reshape(self.Data.state.shape)
        else:
            # set to unity
            sd = 0.*self.Data.state + 1.0
    except IOError:
	raise Exception('IOError for %s'%filename)
    except:
	try:
	    if 'sd' in self.data.dict():
                sd = self.data.sd.reshape(self.data.state.shape)
            else:
                # set to unity
                sd = 0.*self.data.state + 1.0

	except:
	        return (True,'Failed to open file %s for writing with call to %s'%\
        	        (filename,str('write_output_file')))

    if 'f' in self.__dict__ and f.errors != None:
        error_msg = str(f.errors)
        return (True,error_msg)
    # make a guess at what the format is
    try:
        fmt = self.options.result.fmt
    except:
        try:
            fmt = self._state.name.fmt
        except:
            try:
	        fmt = self.name.fmt
            except:
                fmt = None

    if fmt == None:
        fmt = 'PARAMETER'
    try:
        state = self.Data.state
    except:
        state = self.data.state
    try:
        name = self._state.name
    except:
        name = self.name
    gridded = False
    try:
        locations = np.array(self.Data.location)
    except:
        try:
            locations = np.array(self.data.location)
        except:
            # gridded
            gridded = True
    try:
        controls = np.array(self.Data.control)
    except:
        if not gridded:
            controls = np.array(sortopt(self.data,'control',[]))

    if gridded and len(np.atleast_1d(state).shape) == 2:
        # so no location info
        try:
            locations = np.arange(state.shape[0])*name.qlocation[0][2]\
							+name.qlocation[0][0]
            self.data.location = locations
        except:
            try:
   	        self.Data.location = locations
            except:
                # need more complex way to sort grid
	        error_msg = "I can't write this format for this data configuration yet ..."
	        self.logger.error(error_msg)
	        return (True,error_msg)
    elif gridded:
	try:
	    (locations,qlocations,state,sd) = self.ungrid(state,sd)
	except:
	    raise Exception("You are trying to ungrid a dataset that wasn't gridded using State.regrid()" +\
		" so the ungridder information is not available. Either load the data using State.grid " +\
	        " or set it up some other way or avoid calling this method with this type of data")
    n_samples = np.prod(state.shape)/state.shape[-1]
    location = np.array(name.location)
    control = np.array(name.control)
    try:
        locations = locations.reshape((n_samples,location.shape[0]))
    except:
	# really its gridded then ...
	# it just thinks it isnt
	try:
	    locations = self.ungridder.location
            locations = locations.reshape((n_samples,location.shape[0]))
	except:
	    raise Exception('Inconsistency in data shape ... locations is not consistent with state')
    if len(np.atleast_1d(control)):
	try:
            controls = controls.reshape((n_samples,control.shape[0]))
        except:
            if (control == np.array(['mask'])).all():
                controls = np.ones(n_samples).reshape((n_samples,control.shape[0]))
            else:
                controls = None
    else:
        controls = None

    bands = np.array(name.state)
    nbands = len(np.atleast_1d(bands))
    names = np.atleast_1d(np.array(name.state))

    grid = state
    if fmt == 'BRDF':
        '''
            The 'old' style UCL BRDF format

            A header of :
            BRDF n_samples nbands BANDS SD

            where:
            BANDS   wavebands (names or intervals or wavelength, nm)
            SD      standard deviation (assumed same for all observations)

            Data columns:
            time    days
            mask    1 == Good data
            VZA     degrees
            VAA       ""
            SZA       ""
            SAA       ""
            R1
            R2 ...
            '''
        header = "#%s %d %d"%(self.headers[fmt],n_samples,nbands)
        for i in xrange(nbands):
            header = header + ' ' + str(names[i])
        #  for i in xrange(nbands):
        #    header = header + ' ' + str(sd[0,i])
        header = header + '\n'

        ww = np.where(location == 'time')[0]
        if not len(np.atleast_1d(ww)):
            location = np.zeros([n_samples,1])
            locations = ['time']
        ww = np.where(location == 'time')[0]
        timer = locations[:,ww]
        try:
            ww = np.where(control == 'mask')[0]
        except:
            ww = [] 
        if not len(np.atleast_1d(ww)):
            mask = (0*timer+1).astype(int)
        elif len(np.atleast_1d(control)):
            mask = controls[:,ww].astype(int)
        else:
            mask = (0*timer+1).astype(int)
        try:
            ww = np.where(control== 'vza')[0]
	except:
            ww = []
        if not len(np.atleast_1d(ww)):
            vza = timer*0
        else:
            vza = controls[:,ww]
        try:
            ww = np.where(control== 'sza')[0]
        except:
            ww = []
        if not len(np.atleast_1d(ww)):
            sza = timer*0
        else:
            sza = controls[:,ww]
	try:
            ww = np.where(control== 'raa')[0]
	except:
            ww = []
        if not len(np.atleast_1d(ww)):
	    try:
                ww1 = np.where(control== 'saa')[0]
	    except:
	        ww1 = []
            if not len(np.atleast_1d(ww1)):
                saa = timer*0
            else:
                saa = controls[:,ww1]
	    try:
                ww1 = np.where(control== 'vaa')[0]
	    except:
	        ww1 = []
            if not len(np.atleast_1d(ww1)):
                vaa = timer*0
            else:
                vaa = controls[:,ww1]
        else:
            vaa = controls[:,ww]
            saa = timer*0
        todo = np.array([timer,mask,vza,vaa,sza,saa]).reshape(6,n_samples).T
        todo = np.hstack((todo,grid))

    elif fmt == 'BRDF-UCL':
        '''
            a slighly modified BRDF output format

            A header of :
            #BRDF-UCL n_samples nbands BANDS SD

            where:
            BANDS   wavebands (names or intervals or wavelength, nm)
            SD      standard deviation
            A 2nd header line of:

            #location time row col

            or similar n_location location fields.

            Data columns:
            LOCATION
            mask    1 == Good data
            VZA     degrees
            VAA       ""
            SZA       ""
            SAA       ""
            R1
            R2 ...
            S1
            S2 ...
            where LOCATION is n_location columns corresponding to
            the information in the #location line.
            R refers ro reflectance/radiance samples and S to SD

            '''
        header = "#%s %d %d"%(self.headers[fmt],n_samples,nbands)
        header1 = "#%s "%self.headers_2[fmt]
        for i in xrange(nbands):
            header = header + ' ' + str(names[i])
        #for i in xrange(nbands):
        #    header = header + ' ' + str(sd[0,i])
        header = header + '\n'
        for i in xrange(len(np.atleast_1d(location))):
            header1 = header1 + ' ' + location[i]
        header1 = header1 + '\n'
        header = header + header1

        #ww = np.where(location == 'time')[0]
        #if not len(ww):
        #location = np.zeros([n_samples,1])
        #locations = ['time']
        #ww = np.where(location == 'time')[0]
        #timer = locations[:,ww]
        ww = np.where(control == 'mask')[0]
        if not len(np.atleast_1d(ww)):
            mask = (0*timer+1).astype(int)
        else:
            mask = controls[:,ww].astype(int)
        ww = np.where(control== 'vza')[0]
        if not len(np.atleast_1d(ww)):
            vza = timer*0
        else:
            vza = controls[:,ww]
        ww = np.where(control== 'sza')[0]
        if not len(np.atleast_1d(ww)):
            sza = timer*0
        else:
            sza = controls[:,ww]
        ww = np.where(control== 'raa')[0]
        if not len(np.atleast_1d(ww)):
            ww1 = np.where(control== 'saa')[0]
            if not len(np.atleast_1d(ww1)):
                saa = timer*0
            else:
                saa = controls[:,ww1]
            ww1 = np.where(control== 'vaa')[0]
            if not len(np.atleast_1d(ww1)):
                vaa = timer*0
            else:
                vaa = controls[:,ww1]
        else:
            vaa = controls[:,ww]
            saa = timer*0
        todo = np.array([mask,vza,vaa,sza,saa]).reshape(5,n_samples).T
        todo = np.hstack((locations,todo))
        todo = np.hstack((todo,grid))
        todo = np.hstack((todo,sd))
    elif fmt == 'PARAMETERS':
        ndim = len(np.atleast_1d(location))
        header = '#%s'%self.headers[fmt]
        for i in xrange(ndim):
            header = header + ' ' + location[i]
        ndim = len(np.atleast_1d(control))
        for i in xrange(ndim):
            header = header + ' ' + control[i]
        for i in xrange(names.size):
            header = header + ' ' + names[i]
        for i in xrange(names.size):
            header = header + ' ' + 'sd-%s'%names[i]
        header = header + '\n'
        if len(np.atleast_1d(control)):
            todo = np.hstack((locations,controls))
        else:
            todo = locations
        grid = grid.reshape(np.array(grid.shape).prod()/grid.shape[-1],grid.shape[-1])
        sd = sd.reshape(np.array(grid.shape).prod()/grid.shape[-1],grid.shape[-1])
        todo = np.hstack((todo,grid))
        todo = np.hstack((todo,sd))
    elif fmt == 'PARAMETERS-V2':
        # dummy
        ndim = len(np.atleast_1d(location))
        header = '#%s'%self.headers[fmt]
        for i in xrange(ndim):
            header = header + ' ' + location[i]
        for i in xrange(names.size):
            header = header + ' ' + names[i]
        header = header + '\n'
        todo = np.hstack((locations,grid))
    else:
        f.close()
        error_msg = "Unrecognised format for data in write_output_file: %s"\
            %fmt
        return (True,error_msg)
    # bit complex, but done because cant dump header data
    # write the header to f (one or two lines)
    import tempfile
    f.write(header)
    # open a tmp file
    ff2 = tempfile.NamedTemporaryFile()
    # save the data to the tmp file
    np.savetxt(ff2.name,todo,fmt=('%11.6f'))
    # read in the formatted data as lines & write straight after the header
    fa = open(ff2.name)
    ff2.close()
    f.writelines(fa.readlines())
    # close both files ... I dont know if you can close the tmp file earlier
    #  to ensure flushed output ...?
    f.close()
    return (False,'Data written to %s with %s'% (filename,str \
                                                 ('write_output_file')))

def demonstration():
    pass

if __name__ == "__main__":
    from eoldas_Files import init_read_write,reader,writer
    help (init_read_write)
    help(reader)
    help (writer)
    demonstration()

