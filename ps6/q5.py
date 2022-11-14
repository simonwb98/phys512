import numpy as np
from matplotlib import pyplot as plt
import h5py
from scipy import signal
import json
import os

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    tp=template[0]
    tx=template[1]
    return tp,tx
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    gpsStart=meta['GPSstart'][()]
    utc=meta['UTCstart'][()]
    duration=meta['Duration'][()]
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

source = r"C:\Users\User\Desktop\other\PHYS-512\ligo_stuff\LOSC_Event_tutorial\BBH_events_v3.json"

def power_spectrum(arr, window):
    N = len(arr)
    # plt.plot(arr)
    window = signal.get_window(window, N)
    windowed_arr = arr*window
    spectral_density = np.fft.fft(windowed_arr)
    return np.abs(spectral_density)**2

def smooth_data(arr, sigma):
    # this is basically jons code
    N = len(arr)
    tau = np.arange(N)
    tau[N//2:] = tau[N//2:] - N
    gauss = (1/sigma*np.sqrt(2*np.pi))**5*np.exp(-0.5*((tau)/sigma)**2)
    arr_ft = np.fft.fft(arr)
    gauss_ft = np.fft.fft(gauss)
    return np.fft.ifft(arr_ft*gauss_ft)


# get all events stored in BBH_events dict
events=json.load(open(source))
window = 'blackman'
sigma = 10 

for event in events:
    # define path for event
    file_name = events[event]
    window = 'blackman'
    
    # get data for each event
    strain1,dt1,utc1 = read_file(os.path.join(file_name['fn_H1']))
    Hanford = power_spectrum(strain1, window)
    
    strain2,dt2,utc2 = read_file(os.path.join(file_name['fn_L1']))
    Livingston = power_spectrum(strain2, window)
    
    # calculate frequency in Hz
    N_Han = len(Hanford)
    N_Liv = len(Livingston)
    
    f_Han = np.arange(N_Han)/(N_Han*dt1)
    f_Liv = np.arange(N_Liv)/(N_Liv*dt2)

    # smooth the data
    smooth_Han = smooth_data(Hanford, sigma)
    smooth_Liv = smooth_data(Livingston, sigma)
    
    
    # plot each event separately
    plt.loglog(f_Han, smooth_Han, label=f'Hanford, {event}')
    plt.loglog(f_Liv, smooth_Liv, label=f'Livingston, {event}')
    plt.legend()
    plt.xlabel('f [Hz]')
    plt.ylabel('Power Spectrum S(f)')
    plt.savefig(f'{event}.jpg')
    plt.clf()
    
    # PART (b)
    
    # get templates for matched filtering
    tp, tx = read_template(file_name['fn_template'])
    N = len(tp)
    window = signal.get_window(window, N)
    
    # noise matrices
    Ninv_Han = 1/smooth_Han
    Ninv_Liv = 1/smooth_Liv
    
    # whiten data
    white_Han = np.sqrt(Ninv_Han)*np.fft.fft(strain1*window)
    white_Liv = np.sqrt(Ninv_Liv)*np.fft.fft(strain2*window)
    
    # get template ft and whiten
    tp_ft = np.fft.fft(tp*window)
    white_temp_Han = np.sqrt(Ninv_Han)*tp_ft
    white_temp_Liv = np.sqrt(Ninv_Liv)*tp_ft
    
    rhs1 = np.fft.ifft(np.conj(white_temp_Han)*white_Han)
    rhs2 = np.fft.ifft(np.conj(white_temp_Liv)*white_Liv)
    
    # get x axis
    t1 = np.arange(len(white_Han))*dt1
    t2 = np.arange(len(white_Liv))*dt2
    
    plt.plot(t1, rhs1, label=f'Hanford, {event}')
    plt.plot(t2, rhs2, label=f'Livingston, {event}')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('probability of event')
    plt.savefig(f'matched_filter_{event}.jpg', dpi=300)
    plt.clf()
    
assert(1==0)

# Loading template
tp,tx=read_template(tmp_name)

# Creating window function
x=np.linspace(-np.pi/2,np.pi/2,len(strain))
win=np.cos(x)**1


# FT of data
noise_ft=np.fft.rfft(win*strain)
tobs=dt*len(noise_ft)
dnu=1/tobs
nu=np.arange(len(noise_ft))*dnu
nu[0]=0.5*nu[1]

plt.loglog(nu, np.abs(noise_ft)**2, label = "Power Spectrum")


# Smoothing our data to create weights
noise_smooth=smooth_vector(np.abs(noise_ft)**2, 20)
plt.loglog(nu[:-1], noise_smooth, label = "Smoothed Power Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.legend()
# plt.savefig("figs/a6q5_comp_smooth_ps.jpg")
plt.show()



# # Plotting the whitened data
# plt.clf()
# plt.loglog(nu[:-1], np.abs(noise_ft[:-1])**2/noise_smooth, label = "Whitened Power Spectrum")
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude")
# plt.legend()
# # plt.savefig("figs/a6q5_whitened_ps.jpg")
# plt.show()