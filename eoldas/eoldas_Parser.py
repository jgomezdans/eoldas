#!/usr/bin/env python
import sys
import os
import logging
import cPickle
import pdb
import optparse
from optparse import OptionParser
import numpy as np
try:
	from IPython.Shell import IPShellEmbed
except:
	pass
from eoldas_Lib import sortlog
from eoldas_ConfFile import ConfFile
from eoldas_ParamStorage import ParamStorage

class Parser():
    def __init__(self,args,name=None,general=None,log=False,logger=None,outdir=".",getopdir=False,parse=True):
        """
        Initialise parser class.

        This sets up the class defaults.
        
        Options:
            
        general=general: this over-rides and defaults with values 
                        set in parser general can be of the form:
                         1. class ParamStorage (i.e. the same form 
                            as self.general)
                         2. a command line list (where the first 
                            item in the list is ignored
                         3. a string containing a set of command line general
            
                         See self.parse() for more details on general 
                            2 and 3 as these simply make a call to tha
                            method.
            
        log=True         If log is set to True, then logging starts 
                            when this class is instanced.
                            Note that the logfile and logdir might 
                            change if subsequent calls to Parser.parse()
                            are made
        """
        if type(args) == str:
            args = args.split()
        self.dolog = log 
        self.log = log
        self.name = args[0]
        self.args = args[1:]
        self.fullargs = args
        self.store_fullargs = args
        if name == None:
           import time
           thistime = str(time.time())
           name = type(self).__name__
           name =  "%s.%s" % (name,thistime)
	self.thisname = name
        # find the following flags:
        # --conf | -c : conf
        # --datadir   : datadir
        datadir = [".","~/.eoldas",sys.path[0]+'/../bin',sys.path[0]+'/../confs',\
		sys.path[0]+'/../system_confs',sys.path[0]+'/../eoldaslib']
        conf = "default.conf"
        logfile = None
        logdir = "."
        self.top = ParamStorage ()
        self.top.general = ParamStorage ()
        self.top.general.__helper__ = ParamStorage ()
        self.top.general.__default__ = ParamStorage ()
        self.top.general.__extras__ = ParamStorage ()
        self.top.general.conf = []
        for i in xrange(len(self.args)):
            theseargs = self.args[i].split('=')
            if theseargs[0] == "--conf":
                conf = theseargs[1]
                self.top.general.conf.append(conf)
            elif theseargs[0][0:2] == "-c":
                if len(theseargs) > 2:
                    conf = theseargs[0][2:]
                else:
                    conf = self.args[i+1]
		self.top.general.conf.append(conf)
            elif theseargs[0] == "--datadir":
                datadir1 = theseargs[1].replace('[','').\
                                            replace(']','').split()
                [datadir1.append(datadir[i]) for i in \
                                            xrange(len(datadir))]
                datadir = datadir1
            elif theseargs[0] == "--logfile":
                logfile= theseargs[1]
            elif theseargs[0] == "--logdir":
                logdir = theseargs[1]
            elif theseargs[0] == "--outdir":
                outdir = theseargs[1]
        if self.top.general.conf == []:
	    self.top.general.conf = conf 
	if logfile == None:
            logfile = conf.replace('conf','log')
        self.top.general.here = os.getcwd()
        self.top.general.datadir = datadir
        self.top.general.logfile = logfile
        self.top.general.logdir = logdir
        self.top.general.outdir = outdir
        # add here to datadir
        # in addition to '.' to take account of the change of directory
        self.top.general.datadir = self.__add_here_to_datadir(\
                    self.top.general.here,self.top.general.datadir)
        self.top.general.datadir = self.__add_here_to_datadir(\
                    self.top.general.outdir,self.top.general.datadir)
        # cd to where the output is to be 
        self.__cd(self.top.general.outdir)
        # set up the default command line options
        self.default_loader() 
        # update with anything passed here
        if general and type(general) == ParamStorage: self.top.update(\
                    self.__unload(general),combine=True)
        # read the conf files to get any cmd line options
        self.logger = sortlog(self,self.top.general.logfile,logger,name=self.thisname,\
                    logdir=self.top.general.logdir)
        self.config = ConfFile(self.top.general.conf,name=self.thisname+'.config',\
		    loaders=self.loaders,datadir=self.top.\
                    general.datadir,logger=self.logger,logdir=self.top.general.logdir,\
                    logfile=self.top.general.logfile)   
        if len(self.config.configs) == 0:
            this = "Warning: Nothing doing ... you haven't set any configuration",\
						self.config.storelog
	    try:
                self.logger(this)
	    except:
	 	print "Called with args:"
		print "eoldas",self.args
	 	pass
            raise Exception(this)
            
        # now loaders contains all of the defaults set here
        # plus those from the config (config opver-rides defaults here)
        self.loaders = self.config.loaders
        # now convert loaders into parser information
        self.parseLoader(self.loaders)
        self.parse(self.fullargs)
        if general and type(general) == ParamStorage: 
		self.top.update(self.__unload(general),combine=True)
        if general and type(general) == str: 
		self.parse(general.split())
        if general and type(general) == list: 
		self.parse(general)
        # now update the info in self.config
        for i in self.config.infos:
            i.update(self.top,combine=True)
            # so now all terms in self.config.infos
            # contain information from the config file, updated by 
            # the cmd line
            i.logger = self.logger
            i.log()
        # move the information up a level
        self.infos = self.config.infos
        self.configs = self.config.configs
        self.config_log = self.config.storelog

        #if getopdir:
        #    self.sortnames()
        self.config.loglist(self.top)

        #del(self.config.infos)
        #del(self.config.configs)
        #del(self.config)
        self.__cd(self.top.general.here)

    def __add_here_to_datadir(self,here,datadir):
        from os import sep,curdir,pardir
        if type(datadir) == str:
            datadir = [datadir]
        iadd = 0
        for i in xrange(len(datadir)):
            j = i + iadd
            if datadir[j] == curdir or datadir[j] == pardir:
                tmp = datadir[:j]
                rest = datadir[j+1:]
                tmp.append("%s%s%s" % (here,sep,datadir[j]))
                tmp.append(datadir[j])
                iadd += 1
                for k in xrange(len(rest)):
                    tmp.append(rest[k])
                datadir = tmp
        return datadir       

    def default_loader(self):
        """
        Load up parser information for first pass
        """
        self.loaders = []
        self.top.general.here = os.getcwd()
        self.loaders.append(["datadir",['.',self.top.general.here],\
            "Specify where the data and or conf files are"])
        self.loaders.append(["passer",False,\
        "Pass over optimisation (i.e. report and plot the initial values)"])
        self.loaders.append(["outdir",None,\
        "Explicitly mspecify the results and processing output directory"])
        self.loaders.append(["verbose",False,"Switch ON verbose mode","v"])
        self.loaders.append(["debug",False,optparse.SUPPRESS_HELP,"d"])
        self.loaders.append(["conf","default.conf",\
        "Specify configuration file. Set multiple files by using the flag multiple times.","c"])
        self.loaders.append(["logdir","logs",\
        "Subdirectory to put log file in"])
        self.loaders.append(["logfile","logfile.logs","Log file name"])


    def parseLoader(self,loaders):
        """
        Utility to load a set of terms from the list loaders 
        into the ParamStorage general

        If there are 3 terms in each loaders element, they refer to:
        1. name
        2. default value
        3. helper text
        If there is a fourth, it is associated with extras 
        (short parser option)
        """
        general = ParamStorage ()
        general.__default__ = ParamStorage ()
        general.__extras__ = ParamStorage ()
        general.__helper__ = ParamStorage ()

        for this in loaders:
            if len(this) > 1:
                general[this[0]] = this[1]
                general.__default__[this[0]] = this[1]
            else:
                general[this[0]] = None
                general.__default__[this[0]] = None
            if len(this) > 2:
                general.__helper__[this[0]] = this[2]
            else:
                general.__helper__[this[0]] = optparse.SUPPRESS_HELP
            if len(this) > 3:
                general.__extras__[this[0]] = "%s" % this[3]
            else:
                general.__extras__[this[0]] = None
            # make sure arrays arent numpy.ndarray
            if type(general.__default__[this[0]]) == np.ndarray:
                general.__default__[this[0]] = \
                        list(general.__default__[this[0]])

        self.top.update(self.__unload(general),combine=True)


    def __list_to_string__(self,thisstr):
        """
        Utility to convert a list to some useable string
        """
        return(str(thisstr).replace('[','_').strip("']").\
            replace('.dat','').replace("_'","_").replace(",","").\
            replace(" ","").replace("''","_").replace("___","_").\
            replace("__","_"));


    def __cd(self,outdir):
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSerror:
                print "Fatal: Prevented from creating",outdir
                sys.exit(-1)
        try:
            os.chdir(outdir)
        except:
            print "Fatal: unable to cd to",outdir
            raise Exception("Fatal: unable to cd to %s"%outdir)

    def sortnames(self):
        """
        Utility code to sort out some useful filenames & directories
        """
        if self.top.general.outdir == None:
            basename = self.top.general.basename
            confnames =  self.__list_to_string__(self.top.general.conf)
            self.top.general.outdir = basename + "_conf_" + confnames
                
        self.__cd(self.top.general.outdir)
 
    def parse(self,args,log=False):
        '''
        Given a list such as sys.argv (of that form)
        or an equivalent string parse the general and store 
        in self.parser
        '''
        self.dolog = log or self.dolog
        if type(args) == type(""):
            args = args.split()
        args = args[1:]
        self.top.general.cmdline = str(args)
        usage = "usage: %prog [general] arg1 arg2"
        parser = OptionParser(usage,version="%prog 0.1")

        # we go through items in self.top.general and set up
        # parser general for each
        for this in sorted(self.top.general.__helper__.__dict__.keys()):
            # sort out the action
            # based on type of the default
            default=self.top.general.__default__[this]
            action="store"
            thistype=type(default)
            if thistype == type(None):
                thistype = type("")

            argss = '--%s'%this 
            dest = "%s"%this
            helper = self.top.general.__helper__[this]
            if type(default) == type([]):
                # list, so append
                action="store"
            elif type(default) == type(True):
                action="store_true"
                typer = "string"
            if thistype != type([]) and thistype  != type(True):
                typer='%s' % str(thistype).split("'")[1]
                # has it got extras?
                if self.top.general.__extras__[this] != None:
                    parser.add_option('-%s'%self.top.general.__extras__[\
                        this].lower(),argss,type="string",action=action,\
                        help=helper,default=str(default))
                else:
                    parser.add_option(argss,action=action,help=helper,\
                        type="string",default=str(default))
            elif ( thistype != type(True)):
                if self.top.general.__extras__[this] != None:
                    parser.add_option('-%s'%self.top.general.__extras__[\
                        this].lower(),argss,type="string",action=action,\
                        help=helper,default=str(default))
                else:
                    parser.add_option(argss,action=action,help=helper,\
                        default=str(default))
            if thistype == type(True):
                if self.top.general.__extras__[this] != None:
                    parser.add_option('-%s'%self.top.general.__extras__[\
                        this].lower(),argss,dest=dest,action=action,\
                        help=helper,default=default)
                else:
                    parser.add_option(argss,action=action,help=helper,\
                        dest=dest,default=default)
                that = this.split('.')
                argss = '--'
                for i in xrange(len(that)-1):
                    argss = argss + "%s." % that[i]
                argss = argss + 'no_%s' % that[-1]
                helper='The opposite of --%s'%this
                action='store_false'
                typer='%s' % str(thistype).split("'")[1]
                # has it got extras?
                if self.top.general.__extras__[this] != None:
                    parser.add_option('-%s'%self.top.general.__extras__[\
                        this].capitalize(),argss,dest=dest,action=action,\
                        help=helper)
                else:
                    parser.add_option(argss,action=action,dest=dest,\
                        help=helper)


        # we have set all option types as str, so we need to interpret 
        # them in__unload
        (general, args) = parser.parse_args(args)
        #for data_file in args:
        #  general.data_file.append(data_file)
        #general.data_file = list(np.array(general.data_file).flatten())
        #general.brf= list(np.array(general.brf).flatten())
        # load these into self.general
        self.top.update(self.__unload(general.__dict__),combine=True)
        #self.sortnames()
        if self.dolog:
            self.log = set_up_logfile(self.top.general.logfile,\
                                      logdir=self.top.general.logdir)
            self.log_report()

    def __unload(self,options):
        from eoldas_ConfFile import array_type_convert
        this = ParamStorage()
        this.general = ParamStorage()
        for (k,v) in options.iteritems():
            ps = this
            that = k.split('.')
            if len(that) == 1:
                ps = this.general
            else:
                for i in xrange(len(that)-1):
                    if not hasattr(ps,that[i]):
                        ps[that[i]] = ParamStorage()
                    ps = ps[that[i]]
            # set the value v which needs to 
            # to be interpreted
            ps[that[-1]] = array_type_convert(self.top,v)
        return this



def demonstration():
    # this will read a conf file & over-ride with cmd line options
    here = os.getcwd()
    print "Parser help"
    #help(Parser)
    # start parser
    print "Testing Parser class with conf file default.conf"
    self = Parser("%s --conf=default.conf --outdir=test/parser --logfile=log.dat --logdir=logs" % 'test')
    os.chdir(here)
    print "outdir:",self.top.general.outdir
    print "log file in test/parser/logs/log.dat"
    del(self)
    print "Testing Parser class with conf file default.conf and --help"
    self = Parser("%s --no_log --conf=default.conf --outdir=test/parser --logfile=log.dat \
			--logdir=logs --help" % 'test',log=False)
    # normally called as Parser(sys.argv)


if __name__ == "__main__":
    from eoldas_Parser import Parser
    help(Parser)
    demonstration()
