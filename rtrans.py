import numpy as np
import pdb
import matplotlib.pyplot as plt
import glob, os
import os
import copy
from progress.bar import Bar


def rtrans():
    '''
    PURPOSE:
    Does radiative transfer for a collapsing cloud.

    Currently not finished with the radiative transfer part here
    Also need to fix the single temperature velocity calculation.

    '''
    #Define parameters

    #Resolution elements along radius of cloud
    size = 50

    #Molecule
    mole = 'co' #Other available options:  'n2h+','hco+'
    massmole = 28*1.67e-24#g
    nu_0 = 115e9#Hz (Using CO1-0)
    abundance = 1e-5#( of molecules)/(# of H2)
    
    #Central H2 number density of cloud
    nc = 1e3#cm^-3

    #Radius of cloud
    rc = 1#pc


    #Inner outer temperature and power law coefficient
    Tin = 100#K
    Tout = 10#K
    gamma = 2 #Power on 


    #Switch to turn central source on/off
    source = 'True'
    
    #Spectral information
    nures = 11 #Range of velocities in km/s
    delvel =.1 #Bin size 


    #Turbulant velocity
    vturb = 2e3#cm/s
    
    #T background (CMBR)
    T0 = 2.73#K

    c = 3e10 #cm/s
    
    #Set up velocity array
    velrange = np.arange(0, nures, delvel)*1e5
    velrange = velrange - np.mean(velrange)

    velnum = len(velrange)

    #Set up coordinates
    b = np.tile(np.arange(size), (size,1))
    x = np.transpose(b)
    r = np.sqrt(b**2 + x**2)

    #Get the density
    den = denfunc(r, nc = nc, rc = rc, abundance = abundance)
    n = den[0]
    N = den[1]
    
    Trad = np.zeros(size)+T0

    #Get tau, T and vel on both sides of the cloud
    sphere = -(r-size+1)>0
    spheremap = np.hstack((sphere[:,::-1], sphere))

    vel = makevel(massmole, Tin, Tout, gamma, size)
    velmap = np.hstack((-vel[:,::-1], vel))*spheremap

    T = Tfunc(r, gamma, Tin, Tout)
    Tmap = np.hstack((T[:,::-1], T))*spheremap
    
    #Set up initial intensity
    freq = nu_0 * velrange/c + nu_0
    I0 = Bnu(T0, freq)

    #Set up arrays
    freqarr = np.tile(freq, (size,1))
    Icol = np.tile(I0, (size,1))
    
    taucol,Texcol = makemap(size, mole, nc, rc, abundance, Tin, Tout, gamma, -size, Trad)


    #Set up progress bar
    bar = Bar('Radiative transfer', max=size*2)

    for ind in np.arange(size*2):

        
        xind = ind-size
        
        #If excitation temp is 0, set background temp
        Texcol = Texcol + (Texcol == 0)*T0

        if ind == size:
            Texcol[0] = 5800#k
            
        
        #Compute tau and Tex
        taucol, Texcol = makemap(size, mole, nc, rc, abundance, Tin, Tout, gamma, xind, Texcol)
        
        Texcol = Texcol + (Texcol == 0)*T0
        
        Texarr = np.transpose(np.tile(Texcol, (velnum,1)))

        #Compute tau at each velocity bin
        taunu = gauss(taucol, Tmap[:,ind], velmap[:,ind], nu_0, massmole, velrange, vturb = vturb)


        if ind == size:

            Icore = copy.copy(Icol[0])
            Icol[0] = Icore + Bnu(5800, freq)
        

        Itemp = copy.copy(Icol)
        
        Icol = Itemp*np.exp(-taunu)+Bnu(Texarr,freqarr)*(1-np.exp(-taunu))
        
        bar.next()

    bar.finish()

    
    totvel = np.vstack((velmap[::-1,:],velmap))
   
    #Radial Velocity plert
    plt.pcolor(totvel/(1e5))
    cbar = plt.colorbar()
    cbar.set_label('Velocity [km/s]')
    plt.xlabel('X Position [Resolution Elements]')
    plt.ylabel('Y Position [Resolution Elements]')
    plt.savefig('/Users/connorr/Desktop/rtrans/plots/RVplot.eps')

    #Temperate Profile plot
    
    totT = np.vstack((Tmap[::-1,:],Tmap))

    plt.pcolor(totT)
    cbar = plt.colorbar()
    cbar.set_label('Kinetic Temperature [K]')
    plt.xlabel('X Position [Resolution Elements]')
    plt.ylabel('Y Position [Resolution Elements]')
    plt.savefig('/Users/connorr/Desktop/rtrans/plots/Tprof.eps')


    #Number density
    ncor = n*sphere
    nmap = np.hstack((ncor[:,::-1], ncor))
    totn = np.vstack((nmap[::-1,:],nmap))

    plt.pcolor(totn)
    cbar = plt.colorbar()
    cbar.set_label(r'Number Density [cm$^{-3}$]')
    plt.xlabel('X Position [Resolution Elements]')
    plt.ylabel('Y Position [Resolution Elements]')
    plt.savefig('/Users/connorr/Desktop/rtrans/plots/nprof.eps')
    
    plt.show()
    
    pdb.set_trace()

    
    return Icol        
        


    
def gauss(tau, T, v, nu_0, massmole, velrange, vturb):
    '''
    PURPOSE:

    Broadens tau into a gaussian and shifts it based on velocity

    INPUTS:

    tau: Array of tau. (Not yet a function of nu)

    T: Array of temperatures in the cloud

    massmole: Mass of the emitting molecule

    velrange: Array of velocity space to calculate line profiles

    vturb: turbulant velocity, in cm/s
    

    OUTPUTS:

    3D Map of Tau with velocity (frequency)  dependance on the Z axis

    '''

    k = 1.38e-16#cgs units
     
    taunu = np.zeros((np.shape(tau)[0], len(velrange)))

    #Calculate the Dopler velocity
    vtherm = np.sqrt(2*k*T/massmole)
    vdop = np.sqrt(vtherm**2 + vturb**2)

    #Calculate the velocity bin size
    vbin = velrange[1]-velrange[0]

    for i in np.arange(np.shape(tau)[0]):
        #for j in np.arange(np.shape(tau)[1]):
        taunu[i,:] = tau[i] * vbin/(vdop[i]*np.sqrt(np.pi)) * np.exp(-(velrange - v[i])**2/(vdop[i]**2))
            
    return taunu

def Inuout(Iin, tau, T, nu):
    '''
    PURPOSE:
    
    Updates the intensity in a cell as a function of nu

    INPUT:
    
    Iin: Incoming intensity as a function of nu

    tau: Optical depth as a function of nu

    T: Kinetic temperature of the gas
    
    nu: Array containing nu

    '''
    
    Iout = Iin*exp(-tau) + Bnu(T,nu)*(1-np.exp(-tau))
    
    return Iout
    

def Bnu(T, nu):

    h = 6.63e-27#erg s
    c = 3e10#cm/s
    k = 1.38e-16#erg K^-1
    
    B = 2*h*nu**3/c**2/(np.exp(h*nu/(k*T))-1)

    return B

def makemap(length, mole, nc, rc, abundance, Tin, Tout, power, x, Trad):

    '''
    PURPOSE:

    Returns an array of values of tau for each location in the collapsing
    sphere based on the outputs from radex. 

    Parameters for the model are defined below

    '''

    #Path to jobfiles
    path = '/Users/connorr/Desktop/Radex/bin/jobfiles/'

    #Set up the grid
    b = np.arange(length)

    r = np.sqrt(b**2 + x**2)


    #Calculate density and temperature
    T = Tfunc(r, power, Tin, Tout)

    
    den = denfunc(r, nc = nc, rc = rc, abundance = abundance)
    n = den[0]
    N = den[1]

    #Make the job files
    for bind in b:
#        for xind in x[:,0]:
        input(mole, T[bind], N[bind], n[bind], bind, x, path, Trad[bind])

    #Run the job files
    run(path = path)

    tau,Tex = extract(r,length, mole, path)


    #Assume that tau = 0 outside of sphere    
    sphere = -(r-length+1)>0

    return tau*sphere, Tex*sphere
                                                            
def makevel(m_mol, T_in, T_out, gamma, size):
    k = 1.38e-16

    r_max = np.sqrt(size**2+size**2)
    vel_arr = np.zeros((size,size))

    for i in range(0,size):
        for j in range(1,size):
            r = np.sqrt(i**2+j**2)
            vel_arr[i,j] = vel_rad(vel(r/r_max,m_mol,T_in,T_out,gamma),j,i)
    return vel_arr


def vel(r,m_mol,T_in,T_out,gamma):

    # r is in units of cloud radius
    k = 1.38e-16
    v = np.sqrt(-(4*k/m_mol)*(T_in*np.log(r)+((T_in-T_out)/(gamma*(gamma+1)))*(r**gamma-1)))
    #NOTE: NEED TO MAKE CHANGE FOR GAMMA = 0 CASE
    
    return(v)

def vel_rad(v_center,x,b):

    r = np.sqrt(x**2+b**2)
    v_rad = v_center*x/r
    return(v_rad)



def extract(r,length,mole,path):
    '''
    PURPOSE:
    
    Extracts tau + Tex from the jobfiles and then returns an array.
    
    INPUTS:
    
    size: Length of one size of R array

    mole: String containing name of molecule e.g. 'co'

    OPTIONAL INPUTS:
    
    Path to jobfiles

    OUTPUTS:

    Array containing all the information about tau ordered by x and b.

    '''

    fnames = glob.glob(path+'*.out')

    #Set the number of rows to skip
    #Seems to be right number for co, but I do remember something about
    #it changing between molecules so may want to check the .out files
    skip = 13

    #Read in data
    unsorted = []
    tau = np.zeros(length)
    Tex = np.zeros(length)
    
    #For now only grab the first line
    line = 0

    for f in fnames:
        #Grab the impact parameter and x.
        bind = f.split(path+mole+'_')[1].split('_')[0]
#        xind = f.split(bind+'_')[1].split('.')[0]

        temptau = np.loadtxt(f, skiprows = skip, usecols = [7])
        tempTex = np.loadtxt(f, skiprows = skip, usecols = [6])
        tau[np.int(bind)] = temptau[line]
        Tex[np.int(bind)] = tempTex[line]
        

        #Remove the jobfiles
        os.system('rm '+f+' 2>/dev/null')
        os.system('rm '+f[0:-3]+'inp 2>/dev/null')

    #Remove all the files that did not run.
    #Note: If there are more than 10000 files this will fail.
    os.system('rm '+path+'*inp 2>/dev/null')

    return tau, Tex
    
def denfunc(r, nc, rc, abundance):

    '''
    PURPOSE:

    Returns a density field given by the Bonnie-Ebert sphere for a given radius

    INPUTS:

    r: array of radii

    OPTIONAL INPUTS:
    
    rhoc: Central number density in cm-3

    rc: Cloud radius in pc

    abundance: Specify the abundance relative to H2.

    OUTPUT:

    Two arrays, one containing the volume density at each location and the other containing 
    the column density at each location.

    '''
    
    pc = 3.09e18#cm
    
    rout = len(r)-1

    n = nc * (rout**2/(rout**2 + r**2))

    l = (rc * pc)/rout

    N = n*l*abundance

    return (n,N)
    

def Tfunc(r, power, Tin, Tout, constant = False):

    '''
    PURPOSE:

    Returns a value for temperature as a function of radius.

    INPUTS:

    r: array of radii

    OPTIONAL INPUTS:
    
    power: Power law form of the temperature of the cloud.

    Tin: Inner temperature of the cloud. set to 100 by default.
    
    Tout: Outer temperate of the cloud. Set to 10 by default.

    OUTPUT:

    Array containing the temperature at each location.

    '''
    
    rout = len(r)-1
    
    T = Tin - ((Tin-Tout)/(rout**power))*r**power

    return T
    

def input(mole,tkin,N,den,b,x,path, trad):

    '''
    PURPOSE: 

    Creates input files for radex

    INPUTS:
    
    mole: String containting the molecule you want to model, currently 3 available, 'hco+', 'co', and 'hcn'. 

    tkin: Kinetic temperate of the cell

    N: Column density of the cell

    den: Volume density of the cell

    b: Impact parameter from the center of the cloud. Should be in the form of the index of the resolution element
       Define the line of sight through the center of the cloud as b=0

    x: x coordinate measured from the center of the cloud. Again should be in the form of the index of a resolution element.

    OPTIONAL INPUTS:
    path: Path containing the output job files. 
    trad: Radiation temperature, by default set to 2.73.
    '''
    
    #Makes the jobfiles
    
    linewidth = 4


    #format b with 5 significant digits
    bstr = str(b)
    places = 5    
    bzero = '0'*(places-len(bstr))+bstr

    xstr = str(abs(x))
    xzero = '0'*(places-len(xstr))+xstr


    filename = mole+'_'+bzero+'_'+xzero

    infile = open(path+filename+'.inp', 'w')

    #Set the range to look for lines for a specific molecule
    if mole == 'hco+':
        minmax = ['50','1400']
    elif mole == 'co':
        minmax = ['10', '1000']
    elif mole == 'hcn':
        minmax = ['1','1400']
    elif mole == 'n2h+':
        minmax = ['1','100']

        
    Densci = "{:.2E}".format(den)
    Nsci = "{:.2E}".format(N)
        
    infile.write(mole+'.dat\n')
    infile.write(filename+'.out\n')
    infile.write(minmax[0]+' '+minmax[1]+'\n')
    infile.write(str(tkin)+'\n')
    infile.write('1\n')
    infile.write('H2\n')
    infile.write(str(den)+'\n')
    infile.write(str(trad)+'\n')
    infile.write(str(Nsci)+'\n')
    infile.write(str(linewidth)+'\n')
    infile.write(str(0)+'\n')


def run(path):
    #Runs ALL the jobfiles in the path at once
    filelist = glob.glob(path+'*.inp')

    for file in filelist:
        os.system('radex < '+file+'> NUL 2>&1')
        os.system('mv '+ os.getcwd()+'/'+str.split(file[0:-3],path)[1]+'out '+file[0:-3]+'out'+' 2>/dev/null')


