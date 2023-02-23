# -*- coding: utf-8 -*-

#%% READNF
def readndf(filename):
    """
   %READNDF Reads Optotrak NDF data files into Python
    Input:
        filename    = the filename (including path and .ndf-extension) of the datafile you want to load
    Output:
        x           = x-coordinate marker positions [mm]
        y           = y-coordinate marker positions [mm]
        z           = z-coordinate marker positions [mm]
    """
    
    # Load function
    import numpy as np
    
    # Load data
    with open(filename, 'rb') as f:
        filetype = np.fromfile(f,dtype=np.int8,count=1)
        nmarkers = int(np.fromfile(f,dtype=np.int8,count=1)) # number of markers used
        ndim = int(np.fromfile(f,dtype=np.int8,count=1,offset=1)) # number of dimensions measured
        data = np.fromfile(f,dtype=np.float32,offset=252) # data of marker positions
        
    # Reshape data
    Ncol = nmarkers*ndim
    Nrow = int(data.shape[0]/Ncol)
    data = data.reshape((Nrow,Ncol))
    
    # Extract x,y,z data
    x = data[:,0::ndim]
    y = data[:,1::ndim]
    z = data[:,2::ndim]

    # Outputs x, y and z coordinates (nmarker x nsample) with respect to aligned reference frame in mm
    return x,y,z