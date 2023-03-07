###READ HERE###
# Examples of usage of every function given at the end of ther script###
###############

import numpy as np
import pandas as pd
from jdcal import gcal2jd,jd2gcal
import georinex as gr
import os
import matplotlib.pyplot as plt
import glob
import xarray
import seaborn as sns
sns.set()
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 


#######################
# SP3 ORBITS #
#######################

def rt_to_pd(filepath):
    dat = gr.load(filepath)
    nav_all = dat.to_dataframe()
    nav_all.reset_index(inplace=True) 
    
    #GPS time
    Y= np.array(nav_all['time'].dt.year)
    M= np.array(nav_all['time'].dt.month)
    D= np.array(nav_all['time'].dt.day)
    H= np.array(nav_all['time'].dt.hour)
    minutes= np.array(nav_all['time'].dt.minute)
    sec= np.array(nav_all['time'].dt.second)
    JD = np.vectorize(gcal2jd)(Y,M,D)[0]+np.vectorize(gcal2jd)(Y,M,D)[1]+H/24+minutes/60/24+sec/60/60/24
    nav_all['JD']=JD
    return nav_all.dropna()


def GPSweek(Y,M,D,H,minutes,sec):
    '''
    INPUTS:
        year,month,day,hour,minutes and seconds
    OUTPUT:
        GPS Week
    '''

    JD = gcal2jd(Y,M,D)[0]+gcal2jd(Y,M,D)[1]+H/24+minutes/60/24+sec/60/60/24
    GPS_wk = np.fix((JD-2444244.5)/7)
    GPS_sec_wk = round( ( ((JD-2444244.5)/7)-GPS_wk)*7*24*3600)
    return(GPS_wk,GPS_sec_wk)

def Kepler_eq_solver(M,e,epsilon,max_iter):
    '''
    INPUTS:
        mean anomaly (in radiants),eccentricity, tolerance, maximum iterations
    OUTPUT:
        eccentric anomaly (in rad)
    '''
    En=M
    for n in range(0,max_iter):
        if abs(M-En+e*np.sin(En))<epsilon:
            return En
        En=M+e*np.sin(En)
    print('Max number of iterations reached')

def read_sp3(filepath):
    fid=open(filepath,'r')
    # Go through the header (23 lines long)
    Greg_time= np.empty((0,6))
    sp3_obs_all=np.empty((0,8))
    
    for i in range(0,23):
        current_line = fid.readline().rstrip()
        # Store the number of satellites in the SP3 file    
        if i==2:
            current_line = current_line[1:len(current_line)]
            F = current_line.split()
            no_sat = F[0]
        
    # Begin going through times and observations
    end_of_file = 0
    i = 0; j = 1
    while end_of_file != 1:        
        current_line = current_line[1:len(current_line)]
        F = current_line.split()
        # Load GPS Gregorian time into variables
        Y = int(F[0])
        M = int(F[1])
        D = int(F[2])
        H = int(F[3])
        minutes = int(F[4])
        sec = float(F[5])
        Greg_time=np.vstack((Greg_time,[Y, M, D, H, minutes, sec]))
        JD = gcal2jd(Y,M,D)[0]+gcal2jd(Y,M,D)[1]+H/24+minutes/60/24+sec/60/60/24
        # Convert GPS Gregorian time to GPS week and GPS TOW
        GPS_wk, GPS_TOW = GPSweek(Y,M,D,H,minutes,sec)

        # Store satellite PRN and appropriate observations
        for n in range(int(no_sat)):
            
            #Go to the next line
            current_line = fid.readline().rstrip()
            #current_line = current_line(2:length(current_line));
            current_line = current_line[2:len(current_line)]
            F = current_line.split()
            
            # Save PRN, positions, and clock error
            PRN = F[0]; x = F[1]; y = F[2]; z = F[3]; clk_err = F[4]

            # Create observation vector
            sp3_obs_all= np.vstack((sp3_obs_all,[GPS_wk, GPS_TOW,JD, PRN, x, y, z, clk_err]))
        
        #Go to next line - check to see if it is the end of file
        current_line = fid.readline().rstrip()
        if not current_line:
            end_of_file = 1
        if current_line.find('EOF')!=-1:
            end_of_file = 1
    cols = ['GPS_wk', 'GPS_TOW','JD', 'PRN', 'x', 'y', 'z', 'clk_err']
    sp3_obs_all = pd.DataFrame(sp3_obs_all,columns=cols, dtype=float)
    return sp3_obs_all


def read_sp3_days(start,n_days):

    '''
    contatenate more sp3 files into one pandas dataset
    INPUTS:
        start: string, containing the first day in the form dd/mm/yyyy
        n_days. integer, number of days
    '''

    dates = pd.date_range(start=start, periods=n_days)
    JD=dates.to_julian_date()
    GPS_wk = np.fix((JD-2444244.5)/7)
    week_day=(pd.Series(dates.weekday)+1).replace(7, 0, inplace=False).to_numpy()

    #create dataframe
    df = pd.DataFrame(columns=['GPS_wk', 'GPS_TOW','JD', 'PRN', 'x', 'y', 'z', 'clk_err'])
    for d in range(n_days):
        
        file='sio'+str(int(GPS_wk[d]))+str(week_day[d])+'.sp3'
        if not os.path.isfile('sp3_data/'+file):
            print('downloading '+file)
            os.system('wget ftp://cddis.gsfc.nasa.gov/gnss/products/'+str(int(GPS_wk[d]))+'/'+file+'.Z -P sp3_data/')  
            
            os.system("uncompress sp3_data/"+file+'.Z')
        df = df.append(read_sp3('sp3_data/'+file))
    df.index=np.arange(len(df))
    return df

def read_igs_days(start,n_days):

    '''
    contatenate more sp3 files into one pandas dataset
    INPUTS:
        start: string, containing the first day in the form dd/mm/yyyy
        n_days. integer, number of days
    '''

    dates = pd.date_range(start=start, periods=n_days)
    JD=dates.to_julian_date()
    GPS_wk = np.fix((JD-2444244.5)/7)
    week_day=(pd.Series(dates.weekday)+1).replace(7, 0, inplace=False).to_numpy()

    #create dataframe
    df = pd.DataFrame(columns=['GPS_wk', 'GPS_TOW','JD', 'PRN', 'x', 'y', 'z', 'clk_err'])
    for d in range(n_days):
        
        file='igs'+str(int(GPS_wk[d]))+str(week_day[d])+'.sp3'
        print(file)
        if not os.path.isfile('igs_data/'+file):
            print('downloading '+file)
            os.system('wget ftp://cddis.gsfc.nasa.gov/gnss/products/'+str(int(GPS_wk[d]))+'/'+file+'.Z -P sp3_data/')  
            
            os.system("uncompress igs_data/"+file+'.Z')
        df = df.append(read_sp3('igs_data/'+file))
    df.index=np.arange(len(df))
    return df


#######################
#     ULTRA RAPID     #
#######################
def read_cont_days(start,n_days):
    dates = pd.date_range(start=start, periods=n_days)
    JD=dates.to_julian_date()
    GPS_wk = np.fix((JD-2444244.5)/7)
    week_day=(pd.Series(dates.weekday)+1).replace(7, 0, inplace=False).to_numpy()

    #create dataframe
    df = pd.DataFrame(columns=['ECEF', 'sv', 'time', 'position', 'clock', 'velocity', 'dclock', 'JD'])
    for d in range(n_days):
        file='igc'+str(int(GPS_wk[d]))+str(week_day[d])+'.sp3'
        if not os.path.isfile('igc_data/'+file):
            
            os.system('wget https://cddis.nasa.gov/archive/gnss/products/rtpp/'+str(int(GPS_wk[d]))+'/'+file+'.Z -P igc_data/')  
            os.system("uncompress igc_data/"+file+'.Z')
        df = df.append(rt_to_pd('igc_data/'+file))
    return df



def sat_cont_days(start,n_days,sv):
    sp3_cont_orbits=read_cont_days(start,n_days)
    sat_cont=sp3_cont_orbits[sp3_cont_orbits['sv']==sv]

    X=sat_cont[sat_cont['ECEF']=='x']['position']
    Y=sat_cont[sat_cont['ECEF']=='y']['position']
    Z=sat_cont[sat_cont['ECEF']=='z']['position']
    
    tx=sat_cont[sat_cont['ECEF']=='x']['JD'].to_numpy()
    ty=sat_cont[sat_cont['ECEF']=='y']['JD'].to_numpy()
    tz=sat_cont[sat_cont['ECEF']=='z']['JD'].to_numpy()
    data_cont=pd.DataFrame((tx,ty,tz,X,Y,Z)).T
    data_cont.columns=['tx','ty','tz','X','Y','Z']
    data_cont=data_cont[(data_cont != 0).all(1)]
    data_cont=data_cont.drop(['tx','ty'],axis=1);  
    data_cont.columns=['JD','x','y','z']
    return data_cont.dropna()




#######################
# BROADCAST EPHEMERIS #
#######################

def read_broadcast_days():

    '''
    up to now roks only with the files available in the folder broadcast_data
    has to be improved

    OUTPUT: pandas dataframe with orbital elements and corrections
    '''

    folder='broadcast_data/'
    uncomp_files = glob.glob(folder+'/*')
    uncomp_files.sort()
    daydata=[]
    for file in uncomp_files:
        daydata.append(gr.load(file))
    obs = xarray.concat((day for day in daydata), dim='time')
    df = obs.to_dataframe()
    return df

def get_orbits(nav_data):

    '''
    from a pandas dataframe, compute the coordinates xyz
    INPUT: pandas dataframe with orbital elements (The output of the function read_broadcast_days filtered
    by space vehicle and index reset, without NA)
    '''
    
    #set constants
    mu = 3.986004418e14       # gravitational constantWGS84 meters^3/sec^2
    odot = 7.2921151467e-5    # Earth’s rotation rate for WGS84 rad/sec
    c = 299792458             # speed of light m/s
    e=nav_data['Eccentricity']
    
    t=nav_data['Toe']
    a=nav_data['sqrtA']**2   #semi-major axis
    n0=np.sqrt(mu/(a**3))    #computed mean motion rad/s
    
    #t is GPS system at time of transmission corrected for transit time
    tk=t-nav_data['Toe']     #Time from ephemeris reference epoch
    
    '''
    tk shall be the actual total time difference between 
    the time t and the epoch time toe ,
    and must account for beginning or end of week crossovers. 
    That is, if tk is greater than 302400, subtract 604800 from tk.
    If tk is less than −302400 seconds, add 604,800 seconds to tk.
    ''' 
    
    tk[tk>302400]=tk[tk>302400]-604800
    tk[tk<-302400]=tk[tk>302400]+604800
    
    n=n0+nav_data['DeltaN']  #corrected mean motion
    Mk=nav_data['M0']+n*tk   #mean anomaly
    
    #compute eccentric anomaly KEPLER EQ
    Ek=np.vectorize(Kepler_eq_solver)(Mk, e,1e-15,10000)
    
    fk=np.arctan2(np.sqrt(1-e**2)*np.sin(Ek),np.cos(Ek)-e) #true anomaly
    Ek=np.arccos((e+np.cos(fk))/(1+e*np.cos(fk)))
    phik=fk+nav_data['omega']  #argument of latitude
    
    #second harmonic perturbations
    delta_uk=nav_data['Cus']*np.sin(2*phik)+nav_data['Cuc']*np.cos(2*phik)
    delta_rk=nav_data['Crs']*np.sin(2*phik)+nav_data['Crc']*np.cos(2*phik)
    delta_ik=nav_data['Cis']*np.sin(2*phik)+nav_data['Cic']*np.cos(2*phik)
    
    uk=phik+delta_uk #corrected argument of latitude
    rk=a*(1-e*np.cos(Ek))+delta_rk #corrected radius
    ik=nav_data['Io']+delta_ik+nav_data['IDOT']*tk #corrected inclination
    
    #positions in orbital plane
    xk_prime=rk*np.cos(uk)
    yk_prime=rk*np.sin(uk)
    
    Omega_k=nav_data['Omega0']+(nav_data['OmegaDot']-odot)*tk- odot*nav_data['Toe'] #corrected longitude of ascending node 
    
    #Earth-fixed coordinates
    xk=xk_prime*np.cos(Omega_k)-yk_prime*np.cos(ik)*np.sin(Omega_k)
    yk=xk_prime*np.sin(Omega_k)+yk_prime*np.cos(ik)*np.cos(Omega_k)
    zk=yk_prime*np.sin(ik)
    
    JD = nav_data['time'].apply(lambda x: x.to_julian_date())

    orbs = np.array((nav_data['GPSWeek'].values,t.values,JD.values,xk.values,yk.values,zk.values)).T
    cols = ['GPS_wk', 'Toe','JD', 'x', 'y', 'z']
    orbs = pd.DataFrame(orbs,columns=cols)
    return(orbs)


def propagate_orbits(nav_data,t_range):
    
    #set constants
    mu = 3.986004418e14       # gravitational constantWGS84 meters^3/sec^2
    odot = 7.2921151467e-5    # Earth’s rotation rate for WGS84 rad/sec
    c = 299792458             # speed of light m/s
    orbs=np.empty((0,5))
    
    for t in t_range:
        GPS_wk = np.fix((t-2444244.5)/7)
        GPS_sec_wk = np.round( ( ((t-2444244.5)/7)-GPS_wk)*7*24*3600)

      
        data_week = nav_data[nav_data['GPSWeek']<=GPS_wk]
        data_week = nav_data[nav_data['GPSWeek']==max(data_week['GPSWeek'])]
    
        upd_= data_week[data_week['Toe']<=GPS_sec_wk]
        upd_data=upd_[upd_['Toe']==max(upd_['Toe'])]
            
        e=upd_data['Eccentricity']
        
        a = upd_data['sqrtA']**2   #semi-major axis
        n0 = np.sqrt(mu/(a**3))    #computed mean motion rad/s
    
        #t is GPS system at time of transmission corrected for transit time
        tk = GPS_sec_wk-upd_data['Toe']     #Time from ephemeris reference epoch
    
        '''
        tk shall be the actual total time difference between 
        the time t and the epoch time toe ,
        and must account for beginning or end of week crossovers. 
        That is, if tk is greater than 302400, subtract 604800 from tk.
        If tk is less than −302400 seconds, add 604,800 seconds to tk.
        ''' 
        
        tk[tk>302400]-=604800
        tk[tk<-302400]+=+604800

        n=n0+upd_data['DeltaN']  #corrected mean motion
        Mk=upd_data['M0']+n*tk   #mean anomaly
    
        #compute eccentric anomaly KEPLER EQ
        Ek=np.vectorize(Kepler_eq_solver)(Mk, e,1e-15,10000)
    
        fk=np.arctan2(np.sqrt(1-e**2)*np.sin(Ek),np.cos(Ek)-e) #true anomaly
        Ek=np.arccos((e+np.cos(fk))/(1+e*np.cos(fk)))
        phik=fk+upd_data['omega']  #argument of latitude
    
        #second harmonic perturbations
        delta_uk=upd_data['Cus']*np.sin(2*phik)+upd_data['Cuc']*np.cos(2*phik)
        delta_rk=upd_data['Crs']*np.sin(2*phik)+upd_data['Crc']*np.cos(2*phik)
        delta_ik=upd_data['Cis']*np.sin(2*phik)+upd_data['Cic']*np.cos(2*phik)
    
        uk=phik+delta_uk #corrected argument of latitude
        rk=a*(1-e*np.cos(Ek))+delta_rk #corrected radius
        ik=upd_data['Io']+delta_ik+upd_data['IDOT']*tk #corrected inclination
    
        #positions in orbital plane
        xk_prime=rk*np.cos(uk)
        yk_prime=rk*np.sin(uk)
    
        Omega_k=upd_data['Omega0']+(upd_data['OmegaDot']-odot)*tk - odot*upd_data['Toe'] #corrected longitude of ascending node 
    
        #Earth-fixed coordinates
        xk=(xk_prime*np.cos(Omega_k)-yk_prime*np.cos(ik)*np.sin(Omega_k))/1000
        yk=(xk_prime*np.sin(Omega_k)+yk_prime*np.cos(ik)*np.cos(Omega_k))/1000
        zk=(yk_prime*np.sin(ik))/1000
        

        orbs = np.vstack((orbs,
                        np.array((upd_data['GPSWeek'].values,np.zeros(len(yk.values))+t
                                ,xk.values,yk.values,zk.values)).T))
    cols = ['GPS_wk', 't', 'x', 'y', 'z']
    orbs = pd.DataFrame(orbs,columns=cols)
    return(orbs)

#######################
# PLOT ORBITS #
#######################

def plot_orbits(time,coordinates):
    mode=input('Choose orbits plot mode: [3D,time_vs_xyz,time_vs_one]')
    assert mode in ['3D','time_vs_xyz','time_vs_one'], "wrong choice"
    if mode=='3D':
        fig=plt.figure(figsize=(7,7))
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.plot(coordinates[:,0],coordinates[:,1],coordinates[:,2])

        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_zlabel('z [km]')
        ax.grid(False)
        plt.tight_layout()
        plt.show()
    if mode=='time_vs_xyz':
        fig=plt.figure(figsize=(16,8))
        p1, = plt.plot(time,coordinates[:,0],'o-',label='x')
        p2, = plt.plot(time,coordinates[:,1],'o-',label='y')
        p3, = plt.plot(time,coordinates[:,2],'o-',label='z')
        plt.legend(handles=[p1, p2, p3], bbox_to_anchor=(1, 1.02), loc='upper left',fontsize=15)
        plt.xlabel('time [JD]',fontsize=20)
        plt.ylabel('km',fontsize=20)
        plt.tight_layout()
        plt.savefig("xyz.png")
        plt.show()
    if mode=='time_vs_one':
        axe=['x','y','z']
        coord=int(input('which coordinate?[1,2,3]'))
        
        fig=plt.figure(figsize=(16,8))
        plt.plot(time,coordinates[:,coord-1])
        plt.xlabel('time [JD]',fontsize=20)
        plt.ylabel(axe[coord-1]+' [km]',fontsize=20)
        plt.tight_layout()
        plt.savefig("1-coord-orbits.png")

        


''' USAGE EXAMPLES

####### NAVIGATION DATA ###########
#read navigation data into pandas dataframe
nav_data=read_broadcast_days()

#filter nav_data by satellite, in this case sv 28
nav_sat=nav_data.loc['G28'].dropna() #+ drop na
nav_sat.reset_index(inplace=True) #reset indexes to get time into a column

#get xyz and time
nav_sat=get_orbits(nav_sat)

#plot 
fig=plt.figure(figsize=(16,8))
plt.plot(nav_sat['JD'],nav_sat['x'],'o-')


####### SP3 ###########
#read 2 weeks of data from 1st jan to 14th
sp3_orbits=read_sp3_days('1/1/2019',14)

#filter by satellite
sat=28
sp3_sat = sp3_orbits[sp3_orbits['PRN']==sat]

#plot
fig=plt.figure(figsize=(16,8))
plt.plot(sp3_sat['JD'],sp3_sat['x'],'o-')

'''
