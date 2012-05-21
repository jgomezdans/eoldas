#!/usr/bin/env python
import pdb
import numpy as np
#from eoldas_model import subset_vector_matrix
from eoldas_State import State
from eoldas_ParamStorage import ParamStorage
#from eoldas_Operator import *
import logging

class Spectral ( ParamStorage ):
    '''
    The purpose of this class is to perform operations on
    spectral data, such as waveband selection, spectral library loading etc.
        
    The following methods are available:
        
        1.      normalise_wavebands (self.bandnames)
        2.      load_bandpass_from_pulses (  wavelengths, bandwidth, nlw, 
                    bandpass_library={}, spectral_interval=1 )
        3.      load_bandpass_library ( bandpass_filename, 
                    bandpass_library={}, pad_edges=True )
        4.      setup_spectral_config ( self, wavelengths, nbands )
    '''

    def __init__(self,options,name=None):
        '''
        The class initialiser 
            
        This sets up the problem, i.e. loads the spectral information from
        an options structure.
            
        The options structure will contain:
            
            y_meta.names:
            --------------------
            The names of the state vector elements for a y State as string list.
            These are interpreted as wavebands associated with the y state. 
            They may be:
                1. the names of bandpass functions e.g. MODIS-b2
                2. Wavelength intervals e.g. "450-550:
                3. Single wavelength e.g. 451
            In the case of 2 or 3, we interpret the wavelengths
            directly from the string. For defined bandpass functions
            we look in a library to interpret the functions.
            
            
        Optional:
            options.operator_spectral
            --------------------
            (lmin,lmax,lstep) for the operator. If this is *not* specified
            then the model is assumed to be 'non spectral' and a flag 
            self.is_spectral is set False. In this case, we form a new set
            of state variables to solve for and don't bother setting up 
            all of the bandpass information. This means that we have to inform 
            the parameters operator to set up a new set of state vectors.
            These are based on the bandpass names self.bandnames and
            the names in options.x_meta.names.
            
            options.datadir
            --------------------
            Data directory for bandpass files
            
            options.bandpass_libraries
            --------------------
            A list of filenames containing band pass libraries

            options.bandpass_library
            --------------------
            A dictionary of existing bandpass functions
            
        This makes calls to:
            self.setup_spectral_config()
            self.load_bandpass_library()
            self.load_bandpass_from_pulses()
            self.normalise_wavebands()
            
            
        '''
        from eoldas_Lib import isfloat,sortopt
        if name == None:
           import time
           thistime = str(time.time())
           name = type(self).__name__
           name =  "%s.%s" % (name,thistime)
        self.thisname = name
        
        self.options = options
        self.logger = logging.getLogger (self.thisname+'.spectral')
        # set self.nlw, self.bandnames & associated size terms
        # we need this to discretise the bandpass functions
        self.setup_spectral_config(self.options.y_meta.state)
        
        self.is_spectral=sortopt(options.general,'is_spectral',True)
        
        if not self.is_spectral:
            self.logger.info("Non spectral operator specified")
            return
        
        self.logger.info("Spectral operator specified")
        if 'datadir' in self.options.dict():
            self.datadir = self.options.datadir
        else:
            self.datadir = ['.','~/.eoldas','..']
        self.logger.info("Data directories: %s"%str(self.datadir))
        if 'bandpass_library' in self.options.dict():
            self.logger.info("Receiving existing bandpass library")
            self.bandpass_library = self.options.bandpass_library
        else:
            self.logger.info("Starting new bandpass library")
            self.bandpass_library={}
        
        if 'bandpass_libraries' in self.options.dict():    
            self.bandpass_libraries = self.options.bandpass_libraries
            for i in self.bandpass_libraries:
                self.logger.info("Attempting to load bandpass library %s"%i)
                self.bandpass_library,is_error = \
                    self.load_bandpass_library(i,self.bandpass_library)
                if is_error[0]:
                    self.logger.error\
                    ('Cannot load file %s loaded into bandpass library'%i)
                    self.logger.error(is_error[1])
        else:
            self.logger.info("No bandpass libraries specified")
	    
	self.lstep = sortopt(self,'lstep',1)      
        self.nlw = sortopt(self,'nlw',None)

	# if you havent defined the wavelength range, use 0-10000 
	if self.nlw == None:
	    self.logger.info("No wavelength bounds defined")
	    self.logger.info("setting as [0,10000,1]")
	    self.nlw = np.arange(10000)
 	self.pseudowavelengths = []
        # Now we can start to load the band names
        for i in self.bandnames:
            # check to see if its in the library:
            if not i in self.bandpass_library:
                # try to interpret it
                ok,this = isfloat(i)
                if ok:
                    self.bandpass_library, bandpass_name = \
                        self.load_bandpass_from_pulses(str(i),\
                                this,0.5*self.lstep,self.nlw,\
                                self.bandpass_library,\
                                self.lstep)
                else:
                    # Try splitting it
                    that = i.split('-')
                    if len(that) == 2 and isfloat(that[0])[0] \
                                                and isfloat(that[1])[0]:
                        f0 = float(that[0])
                        f1 = float(that[1])
                        self.bandpass_library, bandpass_name = \
                            self.load_bandpass_from_pulses(i,\
                                0.5*(f1+f0),0.5*(f1-f0),self.nlw,\
                                self.bandpass_library,\
                                self.lstep)
                    else:
			f0 = len(self.pseudowavelengths) + self.nlw[0]
			bandpass_name = i
			self.logger.info("******************************************************")
			self.logger.info("requested waveband %s is not in the spectral library"%i)
			self.logger.info("and cannot be interpreted as a wavelength or wavelength range")
			self.logger.info("so we will set it up as a pseudowavelength and hope that")
			self.logger.info("any operator you use does not need to interpret its value")
			self.logger.info("for reference:")
			self.logger.info("	band %s"%i)
			self.logger.info("is interpreted here as:")
			self.bandpass_library, dummy  = \
                            self.load_bandpass_from_pulses(str(f0),\
                                f0,self.lstep*0.5,self.nlw,\
                                self.bandpass_library,\
                                self.lstep)
                        self.logger.info("      %d nm"%f0)
                        self.logger.info("******************************************************")
			self.pseudowavelengths.append(i)
			try:
			    self.bandpass_library[i] = self.bandpass_library[dummy]	
			except:
			    self.bandpass_library[i] = self.bandpass_library[str(f0)]
        # Now post-process the bandpass library 
        self.normalise_wavebands(self.bandnames)

    def load_bandpass_library ( bandpass_filename, bandpass_library={}, \
                                                    pad_edges=True ):
        """Loads bandpass files.
            
            Loads bandpass files in ASCII format. The data are read from 
            `bandpass_filename`, the spectral range of interest is `nlw` 
            (usually in nm) and if you have already a dictionary with named 
            bandpass functions, you can give it as `bandpass_library`. 
            
            Note that bands names that are already present
            in the library will be ignored. If the bandpass functions aren't 
            padded to 0 at the edges of the band, the `pad_edges` option will 
            set them to 0 at the edges. Otherwise, the interpolation goes a 
            bit crazy.
            
            A bandpass file contains something like:
            
            [begin MODIS-b2]
            450 0.0
            550 1.0
            650 2.0
            750 1.0
            850 0.0
            [end MODIS-b2]
            
            Npte that it doesn't need to be normalised when defined.
            
            """
        from eoldas_Lib import get_filename
        fname,fname_err = get_filename(bandpass_filename,datadir=self.datadir)
        if fname_err[0] == 0:
            bp_fp = open ( fname, 'r' )
        else:
            return bandpass_library,fname_err
    
        self.logger.info('file %s loaded into bandpass library'%fname)
                
        nlw = self.nlw
    
        data = bp_fp.readlines()
        bp_fp.close()
        bands = {}
        for ( i, dataline ) in enumerate ( data ):
            
            if dataline.find ( "[begin" ) >= 0:
                name_start = dataline.replace("[begin", "").strip().\
                    replace("]", "")
                bands[ name_start ] = []
            
            elif dataline.find ( "[end" ) >= 0:
                name_end = dataline.replace("[end", "").strip().replace("]", "")
                # Make sure the file is consistent
                if name_end != name_start:
                    fname_err[0] = True
                    fname_err[1] = \
                    "Inconsistent '[begin ]' and " + \
                    "'[end]' names (%s != %s) at line %d" \
                    % (name_start, name_end, i+1 )
                    return bandpass_library,fname_err
                name_start = None
            else:
                ( x, y ) = dataline.strip().split()
                # Double check ehtat we have a good band name
                if name_start is not None:
                    fname_err[0] = True
                    fname_err[1] = \
                    "Bandpass file  appears corrupt " + \
                    "at line " % (i+1)
                    return bandpass_library,fname_err
                bands[name_start].append ( [float(x), float(y)] )
        
        for ( band_name, filter_fncn ) in bands.iteritems() :
            if not bandpass_library.has_key( band_name ):
                filter_fncn = np.array ( filter_fncn )
                if pad_edges:
                    bandpass_library [ band_name ] = np.interp ( nlw, \
                    filter_fncn[:,0], filter_fncn[:, 1], left=0.0, right=0.0)
                
                else:
                    bandpass_library [ band_name ] = np.interp ( nlw, \
                                        filter_fncn[:,0], filter_fncn[:, 1] )
        return bandpass_library,True
    

    def setup_spectral_config ( self, bandnames ):
        """
        Observation operators that consider spectral information
        store sample spectra (e.g. of chlorophyll absoption) at some given
        spectral interval and over some defined range.
            
        To work with the operator, we need to sample out spectral
        information to this grid.
            
        We assume that the operator spectral limits are defined in:
            
        options.operator_spectral = (min,max,step)
            
        The strings with the waveband names are contained in the
        list (or array) bandnames.
            
        This method sets up:
            self.nlw    :   sampled wavelengths, e.g. [400.,401., ...2500.]
            self.bandnames
                        :   bandnames e.g. ['451','MODIS-b2']
            
        some size stores: 
            self.nbands :   len(self.bandnames)
            self.lmin   :   min(self.nlw)
            self.lmax   :   max(self.nlw)
            self.lstep  :   self.nlw[1]-self.nlw[0]
            self.nl     :   len(self.nlw)
            
        """
        self.bandnames = bandnames
        self.nbands = len(bandnames)
        try:
            if not 'bounds' in self.options.options.rt_model.dict().keys():
                self.is_spectral = False
                return
            else:
                self.is_spectral = True
        except:
            self.is_spectral = False
            return
        self.lmin = self.options.options.rt_model.bounds[0]
        self.lmax = self.options.options.rt_model.bounds[1]
        self.lstep = self.options.options.rt_model.bounds[2]
        
        # In case we want to set up bandpass functions, we need to know how
        # the spectral functions are sampled
        # nl is the number of samples. Usually 2101 :)
        self.nl = int((self.lmax - self.lmin + 1 )/self.lstep + 0.5 )
        
        # nlw is the sampled wavelength of each band. 
        #Usually ranges from 400 to 2500nm        
        self.nlw = np.arange(self.nl) * self.lstep + self.lmin
                
    medianbad = lambda self,w,f : w[np.where(np.abs(np.cumsum(f)/f.sum() - 0.5)\
                        == np.min(np.abs(np.cumsum(f)/f.sum() - 0.5)))[0][0]]

    median = lambda self,w,f : np.where(np.min(np.abs((w*f).sum() - w)) == np.abs((w*f).sum() - w))[0][0]

    def normalise_wavebands (self,bandnames):
        """Process and normalise wavebands, as well as select wv flags
            
            Parameters
            ------------
            
            bandnames : array-like
            Bandpass names that should be in self.bandpass_library. 
            
            
            Outputs
            ------------
            
            This method sets up, for each k in bandnames:
            
            self.bandpass_library[k]:
            Normalised bandpass response
            
            self.median_bandpass_library[k]:
            Normalised median bandpass response
            
            self.median_bandpass_index[k]:
            The index of the median wavelength
            
            self.bandpass_index[k]:
            The indices of the bandpass samples
            
            self.all_bands:
            Set to True if a particular wavelength is used
            
            self.median_bands:
            Set to True if a particular wavelength sample is used for the median
            
            How to use
            ------------
            We use this information to request that an observation operator
            only calculates terms for the particular wavelengths we need.
            To do that, it requires an array of flags specifying which
            bands to use. That is contained in the mask self.all_bands (or
            self.median_bands). The observation operator then only calculates
            e.g. reflectance in these bands, so we can load to the full
            spectral array array sp_data with information returned by the 
            operator, data with:
            
            sp_data[self.all_bands] = data
            
            The integral over a waveband then is:
            
            for (i,k) in enumerate(bandnames):
                refl[i] = np.dot(sp_data,self.bandpass_library[k])
                median_refl[i] = np.dot(sp_data,self.median_bandpass_library[k])
            
            Or, a slightly faster access if the arrays are large:
            
            for (i,k) in enumerate(bandnames):
                ww = self.bandpass_index[k]
                refl[i] = np.dot(sp_data[ww],self.bandpass_library[k][ww])
            
            """
        self.median_bandpass_library = {}
        self.median_bandpass_index = {}
        self.bandpass_index = {}
        self.all_bands = np.zeros_like(self.nlw).astype(float)
        self.median_bands = np.zeros_like(self.nlw).astype(int)
        self.bands_to_use = []
        self.median_bands_to_use = []
        for k in bandnames:
            self.logger.info('Loading band %s'%str(k))
	    try:
                v = self.bandpass_library[k]
            except:
	        # some confusion wrt strings and numbers so fix it
		v = self.bandpass_library[str(k)]
	        self.bandpass_library[k] = v
            # normalise 
            self.bandpass_library[k] = v/v.sum()
            # find median 
            median = self.median(self.nlw,v)
            # specify the index of the median
            self.median_bandpass_index[k] = median
            self.median_bandpass_library[k] = 0.*v
            self.median_bandpass_library[k][self.median_bandpass_index[k]] = 1
   	    self.median_bands[self.median_bandpass_index[k]] = 1
            # specify the indices of the full bandpass
            self.bandpass_index[k] = np.where(self.bandpass_library[k]>0)[0]
            self.all_bands = self.all_bands + self.bandpass_library[k]
            self.bands_to_use.append(self.bandpass_library[k])
            self.median_bands_to_use.append(self.median_bands)
            
        ww = np.where(self.all_bands > 0)[0]
        self.all_bands = ww
        ww = np.where(self.median_bands > 0)[0]
        self.median_bands = ww        
        self.bands_to_use = np.array(self.bands_to_use)
        self.median_bands_to_use = np.array(self.median_bands_to_use)
                
        
 
    
    def load_bandpass_from_pulses (  self,thisname,wavelengths, bandwidth, nlw, \
                                   bandpass_library,spectral_interval ):
        """Bandpass functions from center and bandwidth
            
            This function calculates bandpass functions from the centre 
            wavelength and the bandwdith specified by the user. 
            
            The results are stored in a spectral (the configuration storage 
            for all things spectral!). This function returns a boxcar-type 
            spectral passband function, with edges specified by half
            the bandwidth.
            
            Parameters
            ----------
            bandwidth: array-like
            wavelengths: array-like
            spectral: ParamStorage
            
            
            """
	wavelengths = np.array(wavelengths)
        bandmin = np.array(wavelengths - 0.5*bandwidth)
        bandmax = np.array(wavelengths + 0.5*bandwidth)
	if bandmin.size ==1 and bandmin < 0:
            bandmin = 0
	elif bandmin.size >1:
	    ww = np.where(bandmin<0)
	    bandmin[ww] = 0
        bandpass_names = []
        for indice, band_min  in np.ndenumerate ( bandmin ):
            if band_min == bandmax[ indice ]: # NULL wavelength
                bandpass_names.append ( "NULL" )  
            else:
                bandpass_names.append ( "%f-%f" % \
                                       ( band_min, bandmax[indice] ) )
        
        
        magnitude = np.array ( [0, 1, 1, 0] )
        for (i, bandpass_name ) in enumerate ( bandpass_names ):
            if bandpass_name != "NULL" and \
                (not bandpass_library.has_key ( bandpass_name ) ):
                limits = [float(wv) for wv in bandpass_name.split("-")]
                mini = limits[0]
                maxi = limits[1]
                x = np.array ( [ mini - spectral_interval*0.5, mini, \
                                maxi, maxi + spectral_interval*0.5] )
                xx = np.interp \
                    ( nlw, x, magnitude )
                bandpass_library[thisname] = xx
                bandpass_library[bandpass_name] = xx
        return bandpass_library,bandpass_name


if __name__ == "__main__":
    demonstration()



