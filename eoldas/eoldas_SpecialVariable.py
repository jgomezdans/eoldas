#!/usr/bin/env python
from eoldas_ParamStorage import ParamStorage
import pdb
import numpy as np
from eoldas_Files import writer,reader,init_read_write
from eoldas_Lib import sortlog

class SpecialVariable(ParamStorage):
    '''
    A class that can deal with the datatypes needed for eoldas

    It allows a variable to be set to various data types
    and interprets these into the data structure.
        
    The data structure imposed here is:
        
        self.data.this
        self.name.this 
        
    to store some item called 'this'. Other information can be stored as well
    but part of the idea here is to have some imposed constraints on the
    data structure so that we can sensibly load up odfferent datasets.
        
    The idea is that data will be stored in self.data and associated 
    metadata in self.name. If items are given the same name in both
    sub-structures, we can easily keep track of them. There is no actual
    requirement that this is adhered to, but it is certainy permitted and
    encouraged for the indended use of this class.
        
    Probably the most important thing then about this class is that is
    a SpecialVariable is assigned different data types, then it can do sensible
    things with them in the context of the EOLDAS (and wider applications).
        
    When an assignment takes place (either of ther form 
        
        self.state = foo
        
        or
        
        self['state'] = foo
        
    then what is actually stored depends on the type and nature of foo. The 
    main features as follows:
        
    If foo is a string:
        A guess is made that it is a filename, and an attempt is made to
        read the file. All directories in the list self.dirnames are 
        searched for the filename, and any readable files found are 
        considered candidates, Each of these is read in turn. A set of
        potential data formats, specified by the readers in readers 
        (self.reader_functions) is considered, as if a sucessful interpretation
        takes place the data is returned and stiored in the derired variable.
        
        So, for example, if we have self.state = foo as above and foo is a valid, 
        readable file in the list of directories specified, and it is 
        interprable with one of the formats defined, then the main dataset
        is loaded into:
        
        self.data.state 
        
        (alternatively known as self.data['state']).
        
    If foo is a ParamStorage, it should have the same structure as that here 
        (i.e. self.data and self.name) and these structures are then loaded.
        
        
    If foo is a dictionary (type dict)
        It is first converted to a ParamStorage and then loaded as above.
        
    If foo is any other datatype, it is left pretty much as it, except that
        an attempt to convert to a np.array is made.
        
        Depending on the format, there might be data other than the main
        dataset (e.g. locational information) and these are loaded by the 
        loaders into relevant oarts of self.data and self.name.
        
        For classes that use this class for EOLDAS, we will typically use:
        
        self.data.state     : state variable data
        self.data.sd        : uncertainty information as sd
                              (or a similar fuller representation)
        self.data.control   : control information (e.g. view angles)
        self.data.location  : location information
        
        with associated descriptor data in the relevant parts of self.name.
        
        The idea for simple use of the data structure then is for all of these
        datasets represented as 2D datasets, where the number of rows in
        each of the self.data.state etc field will be the same, but
        the number of columns will tend to vary (e.g different numbers of state
        variables). 
        
        The reason for considering such a 2D 'flat'(ish) representation is
        that it is easy to tabulate and understand. In fact the data will be of
        quite high dimension. E.g. is the data vary by time, x and y, then
        we would have 3 columns for self.data.location, with descriptors for
        the columns in self.name.location, and corresponding state data in
        self.data.state (with the number of state variables determining the
        number of columns in that table).
        
        As mentioned, at the pooint of this class, there is no strict 
        requirement for any such structure ti the data loaded or used, but 
        that is the plan for EOLDAS use, so worth documenting at this point.

    '''

    def __init__(self,info=[],name=None,thisname=None,readers=[],log_terms={},\
                                                datadir=["."],env=None,\
                                                header=None,writers={},
                                                simple=False,logger=None):
        '''
        Class SpecialVariable initialisation.
            
        Sets up the class as a ParamStorage and calls self.init()
            
        See init() fort a fuller descripotion of the options.
            
        '''
        ParamStorage.__init__(self,logger=logger)
        name = name or thisname
        if name == None:
           import time
           thistime = str(time.time())
           name = type(self).__name__
           name =  "%s.%s" % (name,thistime)
        self.thisname = name

        self.init(info=[],name=self.thisname,readers=readers,log_terms={},\
                                                datadir=datadir,env=env,\
                                                header=header,writers=writers,\
                                                simple=False,logger=logger)
        

    
    def init(self,info=[],name=None,readers=[],log_terms={},\
                                                datadir=None,env=None,\
                                                header=None,writers={},\
                                                simple=False,logger=None):
        '''
        Initialise information in a SpecialVariable instance.
            
        Options:
            info        :   Information tthat can be passed through to reader
                            methods (a list).
            thisname    :   a name to use to identify this instance in any
                            logging. By default this is None. If thisname is set
                            to True, then logging is to stdout.
            readers     :   A list of reader methods that are pre-pended to 
                            those already contained in the class.
            log_terms   :   A dictionary of log options. By default
                            {'logfile':None,'logdir':'log','debug':True}
                            If thisname is set, and logfile specified, then 
                            logs are logged to that file. If thisname is set
                            to True, then logging is to stdout.
            datadir     :   A list of directories to search for data files
                            to interpret if the SpecialVariable is set to a 
                            string.
            env         :   An environment variable that can be used to extend
                            the datadir variable.
            header      :   A header string to use to identify pickle files.
                            By default, this is set to
                            "EOLDAS -- plewis -- UCL -- V0.1"
            simple      :   A flag to swicth off the 'complicated' 
                            interpretation methods, i.e. just set and return 
                            variables literally, do not try to interpret them.
            
        
        '''
        self.set('simple',True)
        if name == None:
           import time
           thistime = str(time.time())
           name = type(self).__name__
           name =  "%s.%s" % (name,thistime)
        self.thisname = name

        # this is where we will put any data     
        self.data = ParamStorage()
        self.name = ParamStorage()

        self.info = info

        self.datadir = datadir or ['.']
        self.env = env     

        init_read_write(self,header,readers,writers)
        
        # sort logging and log if thisname != None
        self.log_terms = {'logfile':None,'logdir':'log','debug':True}
        # override logging info
        for (key,value) in log_terms.iteritems():
            self.log_terms[key] = value
        self.logger= sortlog(self,self.log_terms['logfile'],logger,name=self.thisname,\
				logdir=self.log_terms['logdir'],\
				debug=self.log_terms['debug'])
        self.simple = simple

    set = lambda self,this,value :ParamStorage.__setattr__(self,this,value)
    set.__name__ = 'set'
    set.__doc__ = """
        A method to set the literal value of this, rather than attempt
        an interpretation (e.g. used when self.simple is True)
    """
    get = lambda self,this :ParamStorage.__getattr__(self,this)
    get.__name__ = 'get'
    get.__doc__ = """
        A method to get the literal value of this, rather than attempt
        an interpretation (e.g. used when self.simple is True)
        """

    def __setitem__(self,this,value):
        '''
        Variable setting method for style self['this']. 
            
        Interpreted the same as via __setattr__.
        
        '''
        # always set the item
        self.__setattr__(this,value)
    
    def __setattr__(self,this,value):
        '''
        Variable setting method for style self.this
            
        Varies what it does depending on the type of value.
            
        The method interprets and sets the SpecialVariable value:
            
            1.  ParamStorage or SpecialVariable. The data are directly loaded.
                This is one of the most flexible formats for input. It expects
                fields 'data' and/or 'name', which are loaded into self.
                There will normally be a field data.this, where this is 
                the variable name passed here.
            2.  A dictionary, same format as the ParamStorage.
            3.  A tuple, interpreted as (data,name) and loaded accordingly.
            4.  *string* as filename (various formats). An attempt to read the
                string as a file (of a set of formats) is made. If none pass 
                then it it maintained as a string.
            5.  A numpy array (np.array) that is loaded into self.data.this.
            6.  Anything else. Loaded into self.data.this as a numpy array.
            
        '''
        if self.simple:
            self.set(this,value)    
            return
        t = type(value)
        try:
            if t == ParamStorage or t == SpecialVariable:
                # update the whole structure
                #self.__set_if_unset('data',ParamStorage())
                #self.__set_if_unset('name',ParamStorage()) 
                self.data.update(value.data,combine=True)
                self.name.update(value.name,combine=True)
            elif t == dict:
                n_value = ParamStorage().from_dict(value)
                self.__setattr__(this,n_value)
            elif t == tuple or t == list:
                # assumed to be (data,name) or [data,name]
                #self.__set_if_unset('data',ParamStorage())
                #self.__set_if_unset('name',ParamStorage())
                #ParamStorage.__setattr__(self['data'],this,value[0]) 
                #ParamStorage.__setattr__(self['name'],this,value[1])
                ParamStorage.__setattr__(self['data'],this,np.array(value))
            elif t == str:
                # set the term
                #self.__set_if_unset('data',ParamStorage())
                #self.__set_if_unset('name',ParamStorage())
                ParamStorage.__setattr__(self['data'],this,value)
                # interpret as a file read if possible
                self.process_data_string(this,info=self.info)
            elif t == np.ndarray:
                #self.__set_if_unset('data',ParamStorage())
                #self.__set_if_unset('name',ParamStorage())
                ParamStorage.__setattr__(self['data'],this,value)
            else:
                ParamStorage.__setattr__(self['data'],this,\
                                         np.array(value))
        except:
            if self.logger:
                self.logger.info("Failed to set SpecialVariable %s from %s %s"\
                             %(this,t.__name__,value))
            return
        if self.logger:
            self.logger.info("Set variable %s from type %s"%(this,t.__name__))

    def __getattr__(self,name):
        '''
        Variable getting method for style self.this 
            
        If the field 'data' exists in self.__dict__ and 'name'
        is in the dictionary, then the field self.data.this is returned.
        
        Otherwise, if the field 'name' is in self.__dict__, self.name
        is returned.
            
        Otherwise return None.

        '''
        if 'data' in self.__dict__ and name in self.data.__dict__:
            return self.data.__dict__.__getitem__ ( name )
        elif name in self.__dict__:
            return self.__dict__.__getitem__ ( name )
        else:
            return None

    def __getitem__(self,name):
        '''
        Variable getting method for style self['this']. 
            
        Interpreted the same as via __getattr__.
            
        '''
        # first look in data
        if 'data' in self.__dict__ and name in self.data.__dict__:
            return self.data.__dict__.__getitem__ ( name )
        elif name in self.__dict__:
            return self.__dict__.__getitem__ ( name )
        else:
            return None
    

    def process_data_string(self,name,info=[],fmt=None):
        '''
        Attempt to load data from a string, assuming the string is a filename.
            
        The array self.datadir is searched for readable files with the 
        string 'name' (also self.env), and a list of potential files
        considered for reading.
            
        Each readable file is passed to self.read, and if it is interpretable,
        it is loaded according to the read method.
           
        Note tha the format can be specified. If not, then all formats
        are attempted until a sucessful read is made.
 
        '''
        from eoldas_Lib import get_filename

        orig = self.data[name]
        if self.logger:
            self.logger.debug('%s is a string ... see if its a readable file ...' \
                          % name)
        # find a list of potential files
        goodfiles, is_error = get_filename(orig,datadir=self.datadir,\
                                          env=self.env,multiple=True)
        if is_error[0] and self.logger:
            self.logger.debug(str(is_error[1]))
            return 
        if self.logger:
            self.logger.debug("*** looking at potential files %s"%str(goodfiles))
        # loop over all files that it might be
        for goodfile in goodfiles:
            stuff,is_error = reader(self,goodfile,name,fmt=fmt,info=info)
            if not is_error[0] and self.logger:
                self.logger.info("Read file %s "%goodfile)
                return 
        if self.logger:
            self.logger.debug(self.error_msg)
        return 


    write = lambda self,filename,fmt : writer(self,filename,None,fmt=fmt)    
    read  = lambda self,filename,fmt : reader(self,filename,None,fmt=fmt,info=[])  

class DemoClass(ParamStorage):
    ''' 
        A demonstration class using SpecialVariable
        
        The behaviour we desire is that a SpecialVariable
        acts like a ParamStorage (i.e. we can get or set 
        by attribute or item 
        
        e.g. 
        
        x.state = 3 and
        x['state'] = 3
        
        give the same result, and
        
        print x['state'] and
        print x.state 
        
        give the same result. This is easy enough to achieve 
        for all cases other than getting from x.state. It turns out
        that __getattr__ does not override the default method
        for state *if* state is set in the class instance.
        
        To get around that, we have to use a fake name (fakes here)
        and instead of storing state, we store _state. This makes daling with
        with all of the conditions a little more complicated and a little 
        slower, but it allows a much more consistent interface.
        
        At any time, a SpecialVariable can simply be over-written by using 
        assigning to its fake name.
        
        e.g.
        
        instance the class
        
            x = demonstration()
        
        set a non-special value 'cheese'
        
            x.foo = 'bar'
        
        we can use this as x.foo or x['foo']
        
            print x.foo,x['foo']
        
        which should give bar bar
        
        now use the SpecialVariable. There are many way to load this up, 
        but an easy one is via a dictionary.
        
            data = {'state':np.ones(2)*5.  ,'foo':np.ones(10)}
            name = {'state':'of the nation','foo':'bar'}
            this = {'data':data,'name':name}
            x.state = this
        
            print x.state,x['state']
        
        which gives [ 5.  5.] [ 5.  5.], so we get the same from either
        approach. Note that what is returned from the SpecialVariable is 
        only what is in this['data']['state'], and that is fully the intention
        of the SpecialVariable class. It can be loaded with rich information
        from a range of sources, but if you want a quick interpretation of
        the data (i.e. x.state) you only get what is in x.state, or more fully,
        
            x._state.data.state
        
        The other data that we passed to the SpecialVariable is as it was
        when read in, but relative to x._state, i.e. we have:
        
            x._state.name.foo
        
        which is bar.
        
        If you want to directly access the SpecialVariable, you can use:
        
            x.get(x.fakes['state'])
        
        which is the same as
        
            x._state or x[x.fakes['state']]
        
        It is not adviseable to directly use the underscore access as the fakes 
        lookup dictionary can be changed. It is best to always use 
        x.fakes['state']. Indeed, if you want to override the 'special' nature
        of a term such as 'state', you can simply remove their entry from the 
        table:
        
            old_dict = x.fakes.copy()
            del x.fakes['state']
        
        Now, if you type:
        
            print x.state
        
        You get a KeyError for state, so it would have been better to:
        
            x.fakes = old_dict.copy()
            del x.fakes['state']
            x['state'] = x[old_dict['state']]
            print x.state
        
        which should give [ 5.  5.], but the type of x.state will have
        changed from SpecialVariable to np.ndarray.
        
        If you want to convert the SpecialVariable back to a dictionary
        you can do:
        
            print x[x.fakes['state']].to_dict()
        
        or a little less verbosely:
        
            print x._state.to_dict()
        
        
        '''
    

    def __init__(self,info=[],thisname=None,readers=[],\
                 datadir=["."],\
                 env=None,\
                 header=None,\
                 logger=None,
                 log_terms={},simple=False):
        '''
        Class initialisation.
            
        Set up self.state and self.other as SpecialVariables
        and initialise them to None.
            
        '''
        
        self.set('fakes',{'state':'_state','other':'_other'})
        
        nSpecial = len(self.get('fakes'))
        for i in self.fakes:
            thatname = thisname and "%s.%s"%(thisname,i)
                
            self[i] = SpecialVariable(logger=logger,info=info,thisname=thatname,\
                                     readers=readers,datadir=datadir,\
                                     env=env,\
                                     header=header,\
                                     log_terms=log_terms,\
                                     simple=False)
            self[i] = None
    

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

    var = lambda self,this : self[self['fakes'][this]]
    var.__name__='var'
    var.__doc__ = '''
        Return the data associated with SpecialVariable this, 
        rather than an interpretation of it
        '''
    
    
    def __set_if_unset(self,name,value):
        '''
        A utility to check if the requested attribute
        is not currently set, and to set it if so.
        '''
        if name in self.fakes:
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
        elif name in self.fakes:
            this = self.get(self.fakes[name])
            return SpecialVariable.__getitem__(this,name)
        else:
            this = self.get(name)
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
            elif name in self.fakes:
                this = self.get(self.fakes[name])
                SpecialVariable.__setattr__(this,name,value)
            else:
                this = self.get(name)
                ParamStorage.__setattr__(self,name,value)


def demonstration():
    # set state to a filename
    # and it will be loaded with the data
    x = DemoClass()
    
    data = {'state':np.ones(2)*5.  ,'foo':np.ones(10)}
    name = {'state':'of the nation','foo':'bar'}
    this = {'data':data,'name':name}
    x.state = this
    print 1,x.state,x['state']

    
    x.oats = 'beans and barley-o'
    # nothing set so far
    print 2,x['state']
    # should return the same
    print 3,x.state
    x.state = 'test/data_type/input/test.brf'
    print 4,x.state
    print 5,x.Name.fmt

    # set state to a dict and
    # it will load from that
    data = {'state':np.zeros(10)}
    name = {'state':'foo'}
    x.state = {'data':data,'name':name}
    print 6,x.state
 
    # set from a ParamStorage
    # and it will be loaded
    this = ParamStorage()
    this.data = ParamStorage()
    this.name = ParamStorage()
    this.data.state = np.ones(10)
    this.name.state = 'bar'
    this.data.sd = np.ones(10)*2.
    this.name.sd = 'sd info'
    # assign the data
    x.state = this
    # access the data
    print 7,x.state
    # access another member
    # Data, Name == implicitly .state
    print 8,x.Data.sd
    print 9,x.Name.sd
    # set directly
    x.Name.sd = 'bar'
    print 10,x.Name.sd

    # set from a tuple (data,name)
    # or a list [data,name]
    data = 'foo'
    name = 'bar'
    x.state = (data,name)
    print 11,x.state
    x.state = [name,data]
    print 12,x.state

    # set from a numpy array
    x.state = np.array(np.arange(10))
    print 13,x.state

    # set from another state
    y = DemoClass()
    y.state = x.state
    x.state = x.state * 2
    print 'x state',x.state
    print 'y state',y.state

    # set from a float
    x.state = 100.
    print 14,x.state

    # another interesting feature
    # we have 2 special terms in demonstration
    # state and other
    # if we set up some strcture for data 
    # for other
    this = ParamStorage()
    this.data = ParamStorage()
    this.name = ParamStorage()
    this.data.other = np.ones(10)
    this.name.other = 'bar'
    # and the assign it to state
    x.state = this
    print 15,'state',x.state
    # we see state is unchanged
    # but other is also not set.
    print 16,'other',x.other
    # we load into other using:
    x.other = this
    print 'other',x.other
    # but if you look at the information contained
    print 17,x._other.to_dict()
    print 18,x._state.to_dict()
    
    # or better writtem as:
    print 19,x.var('state').to_dict()
    
    # you will see that state contains the other data that was loaded

    # a simple way to write out the data is to a pickle
    # x.write_pickle('xstate','x_state.pkl')
    # but try to avoid using the underscores
    print 20,"x state in pickle:",x.state
    SpecialVariable.write(x._state,'x_state.pkl',fmt='pickle')
    
    # which we can reload:  
    z = DemoClass()
    z.state = 'x_state.pkl'
    print 21,"z state read from pickle",z.state

    # which is the same as a forced read ...
    zz = DemoClass()
    zz.state = 'x_state.pkl'
    print 22,zz.state

    # read a brf file
    zz.Name.qlocation = [[170,365,1],[0,500,1],[200,200,1]]
    zz.state = 'test/data_type/input/interpolated_data.npz'
    print zz.state
    SpecialVariable.write(zz._state,'test/data_type/output/interpolated_data.pkl',fmt='pickle')
    zz.state = 'test/data_type/input/test.brf'
    print zz.state

    # so we can convenirntly use pickle format as an interchange 


if __name__ == "__main__":
    demonstration()
    help(SpecialVariable)
    help(DemoClass)



