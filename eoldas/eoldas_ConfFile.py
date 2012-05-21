#!/usr/bin/env python
import pdb
import sys
import numpy as np
import os
from eoldas_ParamStorage import ParamStorage   

def type_convert(info,this):
    """
    Function to make a guess at the type of a variable
    defined as a string
   
    1. Variables from within the config file can be referred to via $
    e.g.:
    [model]
    names = eeny,meeny,miney
    [parameter]
    names = $model.names,moe

    so:
  
    model.names = ['eeny','meeny','miney'] 
    parameter.names = [['eeny','meeny','miney'],'moe']

    2. Any parameter names that begin 'assoc_' are
    special lists/arrays where the parameter values
    are defined associated with a parameter name. If this
    mechanism is used, parameter.names *must* be defined.

    For example:
   
    [parameter]
    names=  gamma,xlai, xhc,  rpl,  xkab, scen, xkw, xkm,   
            xleafn, xs1,xs2,xs3,xs4,lad


    [parameter.assoc_bounds]
    gamma = 0.01,None
    xlai = 0.01,0.99
    xhc = 0.01,10.0
    rpl = 0.001,0.10
    xkab = 0.1,0.99
    scen = 0.0,1.0
    xkw = 0.01,0.99
    xkm = 0.3,0.9
    xleafn = 0.9,2.5
    xs1 = 0.0, 4.
    xs2 = 0.0, 5.
    xs3 = None, None
    xs4 = None, None
    lad = None, None


    Statements in the config file go through an attempt at
    evaluation. This is partly to set the type (i.e. int, float etc)
    to something reasonable, but has the by-product of allowing
    quite a flexible statement definition.

    For example, we can apply a logical statement

    [general]
    rule = True
    unrule = not $general.rule

    which sets:

    general.rule     = True
    general.unrule   = False

    By default, you have several python modules that 
    you can call from within the config file. 
    
    These are those resulting from:

    import sys
    import numpy as np
    import os
    from eoldas_parser import Parser
    from eoldas_params_storage import ParamStorage

    so, for example you can call:

    [general]
    x = np.random.rand(len($parameter.names))
    datadir = .,sys.exec_prefix
    here = os.getcwdu()
    nice = os.nice(19)

    which results in:

    general.x        = [ 0.40663379  0.93222149  0.95141211  
                            0.06559793  0.71743235  0.22486385
                            0.57224872  0.73858977  0.19203518  
                            0.70350535  0.7228549   0.86616254
                            0.34699597  0.78502996]
    general.datadir  = ['.', '/usr/local/epd-7.0-2-rh5-x86_64']
    general.here     = /data/somewhere/eoldaslib
    general.nice     = 19

    Finally, an attempt is made at executing a statement. This might
    prove a little dangerous (e.g. you could (intentionally or maliciously)
    use it to delete files when parsing a conf file, but if you want to 
    do that there are plenty of other ways to achieve it). An example of 
    when this might be useful might be if you want to import a python class 
    whilst parsing the config file (for some reason). 

    [general]
    np = import numpy as np
    npp = import somethingThatDoesntExist

    which sets:
    
    general.np       = import numpy as np : exec True
    general.npp      = import somethingThatDoesntExist

    Note that if a statement has been executed (i.e. run sucessfully 
    through exec()). 
        
    Then the string ' : exec True' is added to the end of 
    its value as stored.
    
    In the case of 'import somethingThatDoesntExist', this was 
    not sucessfully interpreted via an exec() and so is simply assumed to be 
    a string.

    You might need to be a little careful in using strings. I suppose 
    it is vaguely conceivable that you might actually want to set 
    general.np to 'import numpy as np'. Note that items such as those 
    you have imported are not available to other lines of the config, 
    so import serves little purpose, other than testing perhaps.

    There might come a point where you might be doing so much coding 
    in the config file that you might be better off writing some 
    new methods/classes, but some degree of processing is of value, and 
    at no great computing complexity or processing cost. 
 
    """
    # first, try to eval it
    # for this to work, we need to put info.
    # on anything that looks like a variable
    # which are flagged by using a $
    try:
        orig_this = this
        this = this.replace('$','info.')
        try:
            return eval(this)
        except:
            try:
                exec(this)
                #this += ' : exec True'
                return this
            except:
                pass
        if this != orig_this:
            # then we have failed to interpret something
            # log an error and return None
            raise Exception( 'Failed to interpret %s' % orig_this)
    except:
        pass # dont worry if it doesnt work
    try:
        this_value = eval(this)
        return this_value
    except:
        try:
            exec(this)
            this += ' : exec True'
            return this
        except:
            pass
    # otherwise just return it, with blanks stripped
    return this.strip()


def array_type_convert(info,this):
    """

    Parameters:
        info : a dictionary 
        this : a string 

    The string this is split on CSV and each element
    passed through to be interpreted by the method
    type_convert. A 'safe' split is performed, which 
    doesn't split on [] or ().

    """
    # we dont want to split commas inside [] or ()
    if type(this) != str:
       return this
    x = safesplit(this,',')
    n = len(x)
    if n == 1:
        return type_convert(info,x[0])
    else:
        that = []
        for i in xrange(n):
            that.append(type_convert(info,x[i]))
        # youd think np.flatten  would be 
        # fine here, but it isnt
        other = []
        for i in that:
            if type(i) == list:
                for j in i:
                    other.append(j)
            else:
                other.append(i)
        that = np.array(other)
    return that

def safesplit(text, character):
    '''
    A function to split a string, taking account of [] and ()
    and quotes
    '''
    lent = len(text)
    intoken = ParamStorage()
    qtoken = ParamStorage()
    for i in ["'",'"']:
        qtoken[i] = False
    for i in ["()","[]","{}"]:
        intoken[i] = 0

    start = 0
    lst = []
    i = 0
    while i < lent:
        # are any of intoken, qtoken open
        isopen = False
        for (j,k) in intoken.iteritems():
            if text[i] == j[0]:
                intoken[j] += 1
            elif text[i] == j[1]:
                intoken[j] -= 1
            isopen = isopen or (intoken[j]!=0)
        for (j,k) in qtoken.iteritems():
            if text[i] == j[0]:
                qtoken[j] = not qtoken[j]
            isopen = isopen or qtoken[j]
        if text[i] == character and not isopen:
            lst.append(text[start:i])
            start = i+1
        elif text[i] == '\\':
            i += 2
            continue
        i += 1
    lst.append(text[start:])
    return lst

def assoc_to_flat(names,data,prev):
    '''
    Given an array of names (names)
    and a dataset (data)
    convert to a flat array

    Parameters:
        names: a list of names
        data:  a list of names OR
               a dictionary
        prev:  prev array to load into

    If data is an array of names, a boolean numpy array is returned
    of size len(names) which is True where an element of data appears 
    in names and False elsewhere. This is useful to subset an array
    of the same size as names into one associated with the data array.

    If data is a dictionary then the items associated with the keys
    in the list names is returned as a numpy array. If an item doesnt 
    exist in data, then None is returned.

    This latter use is the main purpose of assoc_to_flat(). In effect
    it loads items from a dictionary into an array, based on the keys
    in the list names.

    In the class ConfFile it is used to translate any data structure
    which has a name starting assoc_ into a numpy array of the same name
    (without the assoc_) at the same level of the hierarchy.

    '''
    out = []
    for (i,n) in enumerate(names):
        try:
            # try to pull from a key
            this = data[n]
        except:
            if type(data) == dict or type(data) == ParamStorage:
                this = prev[i]
            else:
                try:
                    # array
                    ndata = np.array(data)
                    this = bool((ndata == n).sum())
                except:
                    this = None
        out.append(this)
    return np.array(out)
 
class ConfFile(object):
    '''
    A configuration file parser class.

    Parameters:
       options: The class is initialised with "options", 
                which may be of type ParamStorage
                or a list or a string.

    If options is a list or a string, it is assumed to contain the names 
    of one or more configuration files. The list may contain lists 
    that also contain the names of configuration files.
    
    The parsed information is put into self.infos (a list containing 
    elements of ParamStorage type).
    
    The length of self.infos will normally be the same as the length of the 
    list "options". 
    
    If "options" is a string (the name of a configuration file) this is the 
    same as a list with a single element.
  
    If the list contains sub-lists, these are all read into the 
    same part of self.infos.

    The raw configuration information associated with each element 
    of self.infos is in a list self.configs. 

    If a configration file is not found or is invalid, there is a 
    fatal error ONLY of fatal=True.

    Configuration files are searched for by absolute path name, 
    if one is given. If there is a failure to read the file referred 
    to by absolute pathname, the local name of the file is assumed, and the
    search continues for that file through the directory list datadir. 
        
    If the path name is relative, the file is searched for
    through the directory list datadir.
    
    The class is derived from the object class.
    information via the command line.

    See type_convert() for some rules on defining a configration file.


    ''' 
    def __init__(self,options,name=None,loaders=[],logfile=None,\
                 logdir=None,datadir=['.','~/.eoldas'],logger=None\
                 ,env="EOLDAS",fatal=False):
        '''
        Parse the configuration file(s) referred to in the (string or list) 
        "options".

        Parameters:
        
            options     :   The class is initialised with "options", 
                            which may be of type ParamStorage
                            or a list or a string.
                            "options" may also be of type ParamStorage, 
                            in which case it is simply loaded here.

        Options:
            
            log_name    :   name of log. If none is specified, 
                            logging goes to stdout.
            log         :   boolean, specifying whether or not to do 
                            logging.
            datadir     :   a list of strings, specifying where to 
                            look for configuration files.
            env         :   an environment variable that may contain 
                            more places to search for configuration files 
                            (after datadir) (default "EOLDAS").
            fatal       :   specify whether not finding a configuration 
                            file should be fatal or not (default False)

        See type_convert() for some rules on defining a configration file.
            
        '''
        from eoldas_Lib import sortlog 
        if name == None:
           import time
           thistime = str(time.time())
           name = type(self).__name__
           name =  "%s.%s" % (name,thistime)
        self.thisname = name
        self.logdir = logdir
        self.logfile = logfile
        self.storelog = ""
        self.datadir=datadir 
        self.env=env
        self.fatal=fatal
        self.logger = sortlog(self,self.logfile,logger,name=self.thisname,logdir=self.logdir,debug=True)
        self.configs = []
        self.infos = []
        self.loaders = loaders
        if type(options) == ParamStorage:
            self.options.update(options,combine=True)
        if type(options)== list or type(options) == str:
            self.read_conf_file(options)
        #if log == True:
        #    self.loglist(self.info)
    
    def loglist(self,info):
        '''
        Utility to send log items to the logger
        '''
        
        # check to see if info is of type ParamStorage
        if type(info) != ParamStorage:
            return
        # check to see if the ParamStorage item has
        # name and/or doc defined and log as appropriate
        if hasattr(info,'__name__'):
            if hasattr(info,'__doc__'):
                self.logger.info("%s: %s" % (str(info.__name__),\
                                            str(info.__doc__)))
            if hasattr(info,'__name__'):
                self.logger.info("%s: " % (str(info.__name__)))
        if 'helper' in info.dict():
            # log might be a string or a list
            if type(info.helper) == str:
                this = info.helper.split('\n')
            else:
                this = info.helper
            if this != None: 
                for i in this:
                    self.logger.info("%s" % i)

        # loop over items in this ParamStorage and recursively call
        # loglist for them
        for (k,v) in info.iteritems():
            if k[:2] != '__':
                self.loglist(info[k])
        return

    def read_conf_file (self, conf_files,options=None):
        """
        This method reads one or more configuration files into 
        self.infos and self.configs.

        Parameters:
            conf_files     : list of one or more config files

        """
        if type(conf_files) == str:
            conf_files = [conf_files]
        # give up the idea of multiple sets of conf files
        
        fname = np.array(conf_files).flatten().tolist()
        for (i,this) in enumerate(fname):
            fname[i] = os.path.expanduser(this)
        self.logger.info ("Trying config files: %s" % fname )
        config,info,config_error = self.read_single_conf_file( \
        				fname,options=options)
        if config == False:
            self.logger.error(config_error)
        else:
            self.loglist(info) 
            self.configs.append(config)
            self.infos.append(info)
        return self.configs 

    def rescan_info(self,config,this,thisinfo,fullthis,info,d):
        '''
        Try to eval all terms
        '''
        if d > 10:
            return
        for i in thisinfo.dict().keys():
            if type(thisinfo[i]) == ParamStorage:
                self.rescan_info(config,this,thisinfo[i],fullthis,info,d+1)
            else:
                try:
                    # only do strings that have 'info' in them
                    if thisinfo[i].count('info.'):
                        thisinfo[i] = eval('%s'%str(thisinfo[i]))
                except:
                    pass
    
    def scan_info(self,config,this,info,fullthis,fullinfo):
        """
        Take a ConfigParser instance config and scan info into 
        config.info. This is called recursively if needed.

        Parameters:
            config    : the configuration object
            this      : the current item to be parsed
            info      : where this item is to go
            fullthis  : the full name of this
            fullinfo  : the full (top level) version of info.
            
        """
        from eoldas_ConfFile import assoc_to_flat
        # find the keys in the top level
        # loop over 
        thiss = np.array(this.split('.'))
        # just in case .. is used as separator
        ww = np.where(thiss != '')
        thiss = thiss[ww]
        nextone = ''
        for i in xrange(1,len(thiss)-1):
            nextone = nextone + thiss[i] + '.'
        if len(thiss) > 1:
            nextone = nextone + thiss[-1]
        # first, check if its already there
        if not hasattr(info,thiss[0]):
            info[thiss[0]] = ParamStorage()
            info[thiss[0]].helper = []
        # load up the info
        if len(thiss) == 1:
            for option in config.options(fullthis):
                fulloption = option
                # option may have a '.' separated term as well
                options = np.array(option.split('.'))
                # tidy up any double dot stuff
                ww = np.where(options != '')
                options = options[ww]
                # need to iterate to make sure it is loaded 
                # at the right level
                # of the hierachy
                this_info = info[this]
                # so now this_info is at the base
                for i in xrange(len(options)-1):
                    if not hasattr(this_info,options[i]):
                        this_info[options[i]] = ParamStorage()
                        this_info[options[i]].helper = []
                    this_info = this_info[options[i]]
                option = options[-1]
                this_info[option] = array_type_convert(fullinfo,\
                                            config.get(fullthis,fulloption))
                if option[:6] == 'assoc_':
                    noption = option[6:]
                    this_info[noption] = assoc_to_flat(\
                                fullinfo.parameter.names,this_info[option],\
                                this_info[noption])
                    is_assoc = True
                else:
                    is_assoc = False
                if not hasattr(this_info,'helper'):
                    this_info.helper = []
                ndot = len(fullthis.split('.'))
                pres = ''
                for i in xrange(1,ndot):
                    pres += '  '
                if type(this_info.helper) == str:
                    this_info.helper += "\n%s%s.%-8s = %-8s" % \
                        (pres,fullthis,fulloption,str(this_info[option]))
                elif type(this_info.helper) == list:
                    this_info.helper.append("%s%s.%-8s = %-8s" % \
                    (pres,fullthis,fulloption,\
                    str(this_info[option])))
                if is_assoc:
                    if type(this_info.helper) == str:
                        this_info.helper += "\n%s%s.%-8s = %-8s" % \
                                (pres,fullthis,fulloption.replace\
                                ('assoc_',''),str(this_info[noption]))
                    elif type(this_info.helper) == list:
                        this_info.helper.append("%s%s.%-8s = %-8s" % \
                                (pres,fullthis,fulloption.replace\
                                 ('assoc_',''),str(this_info[noption])))
        else:
            self.scan_info(config,nextone,info[thiss[0]],fullthis,fullinfo)
            if thiss[-1][:6] == 'assoc_' and thiss[0] in fullinfo.dict():
                # only do this operation when at the top level
                noption = thiss[-1][6:]
                option = thiss[-1]
                this_info = info
                fulloption = thiss[0]
                this_info = this_info[thiss[0]]
                for i in xrange(1,len(thiss)-1):
                    this_info = this_info[thiss[i]]
                    fulloption = '%s.%s' % (fulloption,thiss[i])
                fulloption = '%s.%s' % (fulloption,noption)
                #this_info[noption] = assoc_to_flat(fullinfo.parameter.names\
                #                                   ,this_info[option],\
                #                                   this_info[noption])
                if  not 'names' in this_info.dict():
                    this_info.names = fullinfo.parameter.names

                if not option in this_info.dict():
                    this_info[option] = [0]*len(this_info.names)
                if not noption in this_info.dict():
                    this_info[noption] = [0]*len(this_info.names)
                this_info[noption] = assoc_to_flat(this_info.names\
                                        ,this_info[option],\
                                        this_info[noption])   

                ndot = len(fullthis.split('.'))
                pres = ''
                for i in xrange(1,ndot):
                    pres += '  '
                if type(this_info.helper) == str:
                    this_info.helper += "\n%s%-8s = %-8s" % (pres,\
                        fulloption,str(this_info[noption]))
                elif type(this_info.helper) == list:
                    this_info.helper.append("%s%-8s = %-8s" % (pres,\
                        fulloption,str(this_info[noption])))
 
    def read_single_conf_file (self, conf_files,options=None):
        """
    
        Purpose:
        parse the information from conf_files into a 
        ConfigParser class instance and return this.
    
        Parameters:
            conf_files : list of one or more config files
    
        Options:
            options=None    : pass an options structure through
        
        Uses:
            self.datadir=['.',,'~/.eoldas'] 
                                 :  list of directories to look 
                                    for config files
            self.env=None        :  name of an environment variable 
                                    where config files
                                    can be searched for if not found 
                                    in datadir (or absolute 
                                    path name not given)
            self.fatal=False     :  flag to state whether the 
                                    call should fail if 
                                    a requested config file is not found.
    
    
        Returns:
            tuple           : (config, config_error)
        where:
            config          : ConfigParser class instance
                              or False if an error occurs
            config_error    : string giving information on error
        """
        import ConfigParser
        from eoldas_Lib import get_filename 
        # Instantiate a parser
        config = ConfigParser.ConfigParser()
        # Read the config files. If it doesn't exist, raise exception.
        # 
        if type(conf_files) == str:
            conf_files = [conf_files]
        all_conf_files = []
       
        for fname in conf_files:
            fname,fname_err =  get_filename(fname,datadir=self.datadir,\
                                            env=self.env)
            if fname_err[0] != 0:
                if self.fatal:
                    return False,False,\
                        "Cannot find configuration file %s\n%s" \
                        % (fname,fname_err[1])
            else:
                all_conf_files.append(fname)
                thisdir = os.path.dirname(fname)
                if not thisdir in self.datadir:
                    self.datadir.append(thisdir)
        if len(all_conf_files) == 0:
            return False,False,\
                "%s: No valid conf files found in list %s in dirs %s" \
                % (os.getcwd(),conf_files,self.datadir)
        config.config_files = config.read(all_conf_files)
        if len(config.config_files) == 0:
            return False,False,\
                "%s: No valid conf files found in list %s in dirs %s" \
                % (os.getcwd(),conf_files,self.datadir)
    
        # from here on, we attempt to pull specific information from
        # the conf files
        info = ParamStorage(name='info',doc=\
                    'Configuration information for %s' % \
                    str(config.config_files))
    
        # scan everything into config.info
        # but it helps to sort it to get the info in the right order
        sections = config.sections()
        #sections.sort()
        firstsections = []
        secondsections = []
        for this in sections:
            if this[:7] == 'general' or this[:9] == 'parameter':
                firstsections.append(this)
            else:
                secondsections.append(this)
        firstsections.sort()
        sections = firstsections
        [sections.append(i) for i in secondsections]
        for this in sections:
            self.logger.debug('...Section %s'%this)
            self.scan_info(config,this,info,this,info)
        self.rescan_info(config,this,info,this,info,0)

        
        self.config = config
        self.info = info  
        if options != None and type(options) == ParamStorage:
            self.info.update(options,combine=True)              

        # sort any helper text looping over self.info
        # into self.loaders
        self.__sort_help(self.info,"")
        try:
	    self.logger.info("Config: %s read correctly" \
                % str(all_conf_files))
        except:
            pass       
        return self.config,self.info,"Config: %s read correctly" \
                % str(all_conf_files)
  
    def __sort_help(self,info,name):
        '''
        sort any helper_ options into the list
        self.loaders

        loader format is of the form:
            
        self.loaders.append(["datadir",['.',self.options.here],
            "Specify where the data and or conf files are"])

        '''
        for (key,item) in info.iteritems():
            if key[:5] == 'help_':
                hkey = key[5:]
                # this is potentially help text for hkey
                # If hitem exists and is of type ParamStorage
                # then its internal text
                if hkey in info.dict().keys():
                    hitem = info[hkey]
                    thisname = '%s%s' % (name,hkey)
                    hthisname = '%s%s' % (name,key)
                    if type(item) != ParamStorage:
                        # if hkey starts with 'general.' get rid of that
                        if thisname[:8] == 'general.':
                            thisname = thisname[8:]
                        # insert into self.loaders
                        isloaded = False
                        for i in xrange(len(self.loaders)):   
                            if self.loaders[i][0] == thisname:
                                self.loaders[i][1] = hitem
                                self.loaders[i][2] = item
                                isloaded = True
                        if not isloaded:
                            self.loaders.append([thisname,hitem,item])
            elif type(item) == ParamStorage and key[:2] != '__':
                # recurse
                self.__sort_help(item,'%s%s.'%(name,key)) 

def demonstration(): 
    '''
    A test call to use ConfFile.

    We import the class.
    Then we initialise a instance of ConfFile with
    the configuration file "default.conf"

    '''
    from eoldas_ConfFile import ConfFile
    print "Testing ConfFile class with conf file default.conf"
    self = ConfFile('default.conf')
    # no log has been set up, so logging info
    # is stored in self.storelog
    print "logger info:"
    print self.storelog 

if __name__ == "__main__":
    demonstration()
    help(type_convert)
    help(array_type_convert)
    help(safesplit)
    help(assoc_to_flat)
    help(ConfFile)
