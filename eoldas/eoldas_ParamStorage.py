#!/usr/bin/env python
import pdb

class ParamStorage( dict):
    """
    A class to store parameters as a dictionary, but to being able to retrieve
    them using point notation.
    
    """
    def __init__(self,name=None,doc=None,logger=None):
        '''
        Initialise ParamStorage instance
            
        (just a call to dict initialisation for self.__dict__)
            
        '''
        if logger:
            self.logger = logger
        #from eoldas_Lib import sortlog
        #sortlog(self,None,logger,debug=True)
        dict.__init__(self)
        # needed for pickling
        if name:
            self.__name__ = str(name)
        if doc:
            self.__doc__ = str(doc)
    #def __getstate__(self):
    #    return self.__dict__
    #def __setstate__(self):
    #    return self.__dict__


  
    def dict(self):
        '''
        Return the dictionary (self.__dict__)
        '''
        return self.__dict__

    def update ( self, other_dict,combine=False,copy=False ):
        '''
        Hierarchically update the ParamStorage self with
        another ParamStorage other_dict.
            
        If combine is True, then a full Hierarchical copy is made.
        If not true, then any sub-members in other_dict overwrite those
        in self.
        
        If copy is True, then a .copy() is applied to all elements when
        updating.
            
        '''
        for ( k, v ) in other_dict.iteritems():
            vtype = type(v) != dict and type(v) != ParamStorage
            if k in self.dict():
                # k exists in self as well as other_dict
                if combine:
                    if type(self[k]) == ParamStorage:
                        self[k].update(other_dict[k],copy=copy,combine=combine)
                    else:
                        self[k] = other_dict[k]
                else: 
                    if copy and vtype:
                        try:
                            vv = v.copy()
                        except:
                            vv = v
                        self[ k ] = vv
                        self.__dict__.__setitem__ ( k, v)
                    else:
                        self[ k ] = v
                        self.__dict__.__setitem__ ( k, v)
            else:
                if copy and vtype:
                    try:
                        vv = v.copy()
                    except:
                        vv = v
                    super( ParamStorage, self ).__setattr__( k, vv)
                else:
                    super( ParamStorage, self ).__setattr__( k, v )


    def log(self,this=None,logger=None,name=None,supername='',logfile="log.dat",n=0,logdir="logs",debug=True,print_helper=False):
        '''
        Dump the whole contents to the a log

        Useful for inspecting data

        If self.logger is set, this is the log that is used, otherwise
        one is set up from logfile (and possibly logdir).
        In this case, name is used as the identifier in the log. You
        have to set name to enable log initialisation.

        If print_helper is True, then print_helper fields are logged (default False)
        
        '''
        from eoldas_Lib import sortlog
        self.logger = sortlog(self,logfile,logger,name=name,logdir=logdir,debug=debug)
        if this == None:
            this = self
            try:
                self.logger.info("**"*20)
                self.logger.info("logging parameters ...")
                self.logger.info("**"*20)
            except:
                pass
        for ( k, v ) in this.iteritems():
            if supername != '':
                thisname = "%s.%s" % (supername,str(k))
            else:
                thisname = str(k)
            strk = str(k)
            doit = True
            if strk.replace('__','') != strk or str(k) == 'helper' or str(k)[:5] == "help_":
                doit = False
            if doit  or print_helper:
                try:
                    self.logger.info("%s = %s" % (thisname,str(v)))
                except:
                    pass
            if type(v) == ParamStorage and k != 'logger':
                self.log(this=this[k],name=None,n=n+2,supername=thisname,print_helper=print_helper)

    def __getitem__(self, key):
        val = self.__dict__.__getitem__ ( key )
        return val

    def __setitem__(self, key, val):
        self.__dict__.__setitem__ ( key, val )

    def __getattr__( self, name ):
        #return self[name]
        #if name in self.__dict__:
        return self.__dict__.__getitem__ ( name )
        #return None

    def __setattr__(self, name, value):
        if name in self:
            self[ name ] = value
            self.__dict__.__setitem__ ( name, value )
        else:
            super( ParamStorage, self ).__setattr__( name, value )
    
    def iteritems(self):
        for k in self:
            yield (k, self[k])
    def __iter__(self):
        for k in self.__dict__.keys():
            yield k

    def to_dict(self,no_instance=False):
        '''
        Convert the contents of a ParamStorage to a dict type
        using hierarchical interpretation of ParamStorage elements.
        You can access the dictionary directly as self.__dict__ but this
        might contain other ParamStorage elements.

        Returns the dictionary.

        This is useful for portability, pickling etc.

        If no_instance is set, instancemethod types
        are treated as strings.
        '''
        this = {}
        for ( k, v ) in self.iteritems():
            t = type(v)
            tt = str(t)[:6]
            if k != 'logger' and k != 'self':
                if type(v) == ParamStorage:
                    this[k] = v.to_dict(no_instance=no_instance)
                # a hck-y bit to not dump methods & classes
                elif not (no_instance and t.__name__ == 'instancemethod') \
                                and tt != '<class' and tt != '<bound' and\
                                str(v).find('<class') < 0 and \
                                str(v).find('<bound method') < 0:
                    this[k] = v
                else:
                    self[k] = str(v)
        return this

    def from_dict(self,thisdict,no_instance=False):
        '''
        Convert the contents of a dict to a ParamStorage type.

        returns the ParamStorage

        This is useful for portability, pickling etc.

        If no_instance is set, instancemethod types
        are treated as strings.

        '''
        for ( k, v ) in thisdict.iteritems():
            t = type(v)
            tt = str(t)[:6]
            if k != 'logger':
                if t == dict: 
                    self[k] = ParamStorage()
                    self[k] = self[k].from_dict(v,no_instance=no_instance)
                elif not (no_instance and t.__name__ == 'instancemethod') and \
                            tt != '<class' and tt != '<bound':
                    self[k] = v
                else:
                    self[k] = str(v)
        return self

    def to_pickle(self,filename,header=None):
        '''
        Write out a pickle file of the ParamStorage contents

        If header is defined, that is output first.

        '''
        # first convert to dict
        self.setup_logger()
        this = self.to_dict(no_instance=True)
        import pickle
        import gzip
        import os 
        opdir = os.path.dirname(filename)
        if opdir != '':
            if not os.path.exists( opdir ):
                try:
                    os.makedirs(opdir)
                except OSerror:
                    raise Exception(str(OSerror),"Prevented from \
                                    creating dir %s" % opdir)
        f = gzip.open(filename,'wb')
        if header:
            pickle.dump(header,f)
        pickle.dump(this,f)
        f.close()
        return True

    def from_pickle(self,filename,header=None):
        '''
        Read a pickle file of the ParamStorage contents

        If header is defined, check that against the 
        first item unpickled and return None
        and log error (if logger is defined in self)

        '''
        self.setup_logger()
        import pickle
        import gzip
        try:
            f = gzip.open(filename,'rb')
        except:
            self.logger.error("Error opening gzip file %s"%filename)
            return 
        if header:
            new_header =pickle.load(f)
            if str(new_header) != str(header):
                self.logger.error(\
                        "Error in file format for attempt at pickle read: %f"\
                        %filename)
                self.logger.error("Expecting header: %s"%header)
                self.logger.error("Obtained header: %s"%new_header)
                return
        # now get the data
        this = pickle.load(f)
        # which will be a dict so needs converting
        that = ParamStorage().from_dict(this,no_instance=True)
        f.close()
        self.update(that,combine=True)
        return that

def demonstration():
    '''
    Demonstration use of the class
    '''
    # first set up a simple dict
    x = {}
    y = {'hello':'world'}
    z = {'foo':'bar'}
    y[0] = z
    x['sub'] = y
    x['sub'][1] = {'i':'scream'}
    x['sub'][2] = {'you':'scream'}
    x['sub'][3] = {'we all':'scream'}
    x['sub'][4] = {'for':'ice cream'}

    # now convert into ParamStorage
    xx = ParamStorage().from_dict(x)
    # log contents to stdout
    xx.log(name=True)
    # convert back to 
    yy = xx.to_dict()
    print yy

    # make a new copy
    xxx = ParamStorage().from_dict(x)
    # dump as pickle
    header = "pickled egs V1.0"
    xxx.to_pickle('test/params_storage/output/test1.pkl',header=header)

    # now get them back
    yyy = ParamStorage().from_pickle('test/params_storage/output/test1.pkl',header=header)
    yyy.log(name=True)
    print yyy.to_dict()

if __name__ == "__main__":
    demonstration()
    help(ParamStorage)

