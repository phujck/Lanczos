import numpy as np
import matplotlib.pyplot as plt
import hams
from scipy.signal import blackman
from scipy.signal import stft
from scipy.signal import savgol_filter
def iFT(A):
    """
    Inverse Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.ifft(minus_one * A)
    result *= minus_one
    result *= np.exp(1j * np.pi * A.size / 2)
    return result


def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    # test
    A = np.array(A)
    k=A.size
    A=np.pad(A, (0, 4*k), 'constant')
    minus_one = (-1) ** np.arange(A.size)
    # result = np.fft.fft(minus_one * A)
    result = np.fft.fft(minus_one * A,n=k)
    minus_one = (-1) ** np.arange(result.size)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    return result

def smoothing(A, b=1,c=8,d=0):
    if b==1:
       b= int(A.size /50)
    if b % 2 == 0:
        b=b+1
    j=savgol_filter(A, b, c,deriv=d)
    return j

def current(sys, phi, neighbour):
    conjugator = np.exp(-1j * phi) * neighbour
    c =sys.a * sys.t * 2*np.imag(conjugator)
    return c

Tracking=False
Track_Branch=False


# These are Parameters I'm using
# number=2
# nelec = (number, number)
# nx = 4
# ny = 0
# t = 0.191
# U = 0.1 * t
# delta = 2
# cycles = 10

# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
number=1
number2=1
nelec = (number, number)
nx = 4
nx2=4
ny = 0
t = 0.191
t2=0.191
U = 0.1*t
U2= 0.1*t
delta = 1
delta2=1
cycles = 1
cycles2=1



prop = hams.system(nelec, nx, ny, U, t, delta, cycles, bc='pbc')
factor=prop.factor
delta1=delta

# load files
parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta.npy' % (nx,cycles,U,t,number,delta)
J_field=np.load('./data/original/Jfield'+parameternames)
phi_original=np.load('./data/original/phi'+parameternames)
phi_reconstruct = np.load('./data/original/phirecon'+parameternames)
neighbour=np.load('./data/original/neighbour'+parameternames)
two_body=np.load('./data/original/twobody'+parameternames)
error=np.load('./data/original/error'+parameternames)

parameternames2='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta.npy' % (nx2,cycles2,U2,t2,number2,delta2)
J_field2=np.load('./data/original/Jfield'+parameternames2)
two_body2=np.load('./data/original/twobody'+parameternames2)
neighbour2=np.load('./data/original/neighbour'+parameternames2)
phi_original2=np.load('./data/original/phi'+parameternames2)
error2=np.load('./data/original/error'+parameternames2)



omegas = (np.arange(len(J_field)) - len(J_field) / 2) / prop.cycles
omegas2 = (np.arange(len(J_field2)) - len(J_field2) / 2) / prop.cycles

delta2=delta*factor

t = np.linspace(0.0, cycles, len(J_field))
t2=np.linspace(0.0, cycles, len(J_field2))

# smoothing- don't use.

# J_field=smoothing(J_field)
# neighbour_real=smoothing(neighbour.real)
# neighbour_imag=smoothing(neighbour.imag)
# two_body_imag=smoothing(two_body.imag)
# two_body_real=smoothing(two_body.real)
# neighbour=np.array(neighbour_real+1j*neighbour_imag)
# two_body=np.array(two_body_real+1j*two_body_imag)



# Plot the current expectation
# plt.plot(t, J_field.real, label='$\\frac{U}{t_0}=5$')
# plt.plot(t2, J_field2.real, label='$\\frac{U}{t_0}=0.1$')

plt.plot(t, J_field.real, label='scaled')
plt.plot(t2, J_field2.real, label='unscaled')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('Current expectation')
plt.show()



# Phi field
cross_times_up=[]
cross_times_down=[]
plt.plot(t, phi_original, label='original',linestyle='dashed')
# for k in range (1,2):
#     if k != 0:
#         line=k*np.ones(len(t)) * np.pi / 2
#         idx_pos = np.argwhere(np.diff(np.sign(phi_original - line))).flatten()
#         idx_neg = np.argwhere(np.diff(np.sign(phi_original + line))).flatten()
#         idx_up=min(idx_pos[0],idx_neg[0])
#         idx_down=max(idx_pos[-1],idx_neg[-1])
#         # idx_up=idx_up[0]
#         # idx_down=idx_down[-1]
#         # plt.plot(t, line, color='red')
#         # plt.plot(t[idx],line[idx], 'ro')
#         cross_times_up.append(idx_up)
#         cross_times_down.append(idx_down)
# # cross_times_up=np.concatenate(cross_times).ravel()
# plt.plot(t[cross_times_up],phi_original[cross_times_up],'go')
# plt.plot(t[cross_times_down],phi_original[cross_times_down],'ro')
# for xc in cross_times_up:
#     plt.hlines(phi_original[xc],0,t[xc],color='green', linestyle='dashed')
# for xc in cross_times_down:
#     plt.hlines(phi_original[xc],t[xc],t[-1],color='red', linestyle='dashed')
# cross_times_up=(t[cross_times_up])
# cross_times_down=(t[cross_times_down])
# if Tracking:
#     plt.plot(t[:J_field_track.size], phi_track, label='Tracking', linestyle='dashdot')
# if Track_Branch:
#     plt.plot(t[:phi_track_branch.size], phi_track_branch, label='Tracking with Branches', linestyle='dotted',color='yellow')
# plt.plot(t, np.ones(len(t)) * np.pi / 2, color='red')
# plt.plot(t, np.ones(len(t)) * -1 * np.pi / 2, color='red')
# plt.yticks(np.arange(-1.5*np.pi, 2*np.pi, 0.5*np.pi),[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(-1.5*np.pi, 2*np.pi, .5*np.pi)])
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('$\\phi$')
plt.show()



# Double occupancy plot
# plt.plot(t, D)
# plt.xlabel('Time [cycles]')
# plt.ylabel('Double occupancy')
# plt.show()



# Current gradients
two_body = np.array(two_body)
extra = 2. * np.real(np.exp(-1j * phi_original)*two_body)
diff = phi_original - np.angle(neighbour)
two_body2 = np.array(two_body2)
extra2 = 2. * np.real(np.exp(-1j * phi_original2)*two_body2)
diff2 = phi_original2 - np.angle(neighbour2)
J_grad = -2. * prop.a * prop.t * np.gradient(phi_original, delta) * np.abs(neighbour) * np.cos(diff)
J_grad2 = -2. * prop.a * prop.t * np.gradient(phi_original2, delta2) * np.abs(neighbour2) * np.cos(diff2)
exact=np.gradient(J_field,delta)
exact2=np.gradient(J_field,delta)

# eq 32 should have a minus sign on the second term, but
eq32 = J_grad +prop.a * prop.t* prop.U * extra
# eq32= -prop.a * prop.t * prop.U * extra
eq33 = J_grad + 2. * prop.a * prop.t * (np.gradient(np.angle(neighbour), delta) * np.abs(neighbour) * np.cos(diff) - np.gradient(
    np.abs(neighbour), delta) * np.sin(diff))

# Just in case you want to plot from a second simulation

eq32_2= J_grad2 + prop.a * t2*U2 * extra2/prop.factor
eq33_2 = J_grad2 + 2. * prop.a * prop.t * (np.gradient(np.angle(neighbour2), delta2) * np.abs(neighbour2) * np.cos(diff2) - np.gradient(
    np.abs(neighbour2), delta2) * np.sin(diff2))

# plot various gradient calculations
plt.plot(t, eq33, label='Gradient calculated via expectations', linestyle='dashdot')
plt.plot(t, exact, label='Numerical current gradient')
plt.plot(t, eq32, linestyle='dashed',
         label='Gradient using commutators')
plt.xlabel('Time [cycles]')
plt.ylabel('Current Expectation Gradient')
plt.legend()
plt.show()


# error plot
plt.plot(t,error,label='First')
plt.title('Error estimation')
plt.legend()
plt.show()


# gradient deviations
plt.plot(t, abs(exact-eq33), label='expectation gradient deviation')
plt.plot(t, abs(exact-eq32), label='commutator gradient deviation')
# scaling error to see it on the same axis as the gradient deviations
plt.plot(t,(error-error[0])*max(abs(exact-eq32))/max(error-error[0]),label='propagator error estimation')
plt.legend()
plt.show()

print("average deviation from gradient when calculated via expectations")
print(np.sqrt(np.mean((exact-eq33)**2)))

print("average deviation from gradient when calculated via commutators")
print(np.sqrt(np.mean((exact-eq32)**2)))




# different windowing functions.

# epsilon=int(t.size/30)
# window=np.ones(t.size)
# window[:epsilon]=np.sin(np.pi * t[:epsilon] / (2.*t_delta*epsilon)) ** 2.
# window[:-epsilon]=np.sin(np.pi * (t[-1]-t[:-epsilon]) / (2.*t_delta*epsilon)) ** 2.

window=blackman(len(J_field))
window2=blackman(len(J_field2))

# plot the spectrum.
xlines=[2*i-1 for i in range(1,15)]
plt.semilogy(omegas, (abs(FT(np.gradient(J_field,delta1) * window)) ** 2), label='$\\frac{U}{t_0}=5$')
plt.semilogy(omegas2, (abs(FT(np.gradient(J_field2,delta2) * window2)) ** 2), label='$\\frac{U}{t_0}=0.1$')

for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
if Tracking:
    plt.semilogy(omegas, abs(FT(np.gradient(J_field_track[:prop.n_time]) * blackman(prop.n_time))) ** 2,
                 label='Tracking')
if Track_Branch:
    plt.semilogy(omegas, abs(FT(np.gradient(J_field_track_branch[:prop.n_time]) * blackman(prop.n_time))) ** 2,
                 label='Tracking With Branches')
plt.legend()
plt.title("output dipole acceleration")
plt.xlim([0, 60])
plt.xlabel('$\\frac{\omega}{\omega_0}$')
# plt.ylim([1e-8,1e5])
plt.show()



# This stuff isn't needed right now


# plt.semilogy(omegas, abs(FT(phi_original[:prop.n_time] * blackman(prop.n_time))) ** 2, label='original')
# if Tracking:
#     plt.semilogy(omegas, abs(FT(phi_track[:prop.n_time]* blackman(prop.n_time))) ** 2, label='Tracking')
#     for xc in xlines:
#         plt.axvline(x=xc, color='black', linestyle='dashed')
# if Track_Branch:
#     plt.semilogy(omegas, abs(FT(phi_track_branch[:prop.n_time] * blackman(prop.n_time))) ** 2,
#                  label='Tracking With Branch')
# plt.legend()
# plt.title("input-field")
# plt.xlim([0, 30])
#
# plt.show()
#
# Y, X, Z1 = stft(phi_original.real, 1, nperseg=prop.n_time/10, window=('gaussian', 2/prop.field))
# Z1 = np.abs(Z1) ** 2
# # plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.LogNorm())
# plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1)
#
# plt.title('STFT Magnitude-Tracking field')
# plt.ylim([0, 10])
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time(cycles)')
# plt.show()
#
#
# if Tracking:
#     Y, X, Z1 = stft(phi_track.real, 1, nperseg=100, nfft=omegas.size/2, window=('gaussian', 2/(prop.field)))
#     print(Y)
#     Z1 = np.abs(Z1)**2
#     plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.PowerNorm(gamma=0.85))
#     plt.ylim(0,8)
#     # plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1,cmap='plasma')
#     plt.colorbar()
#     for xc in cross_times_up:
#         plt.axvline(x=xc, color='green', linestyle='dashed')
#     for xc in cross_times_down:
#         plt.axvline(x=xc, color='red', linestyle='dashed')
#     plt.title('STFT Magnitude-Tracking field')
#     plt.ylim([0, 10])
#     plt.ylabel('Frequency/$\\omega_0$')
#     plt.xlabel('Time(cycles)')
#     plt.show()
#
# if Track_Branch:
#     Y, X, Z1 = stft((phi_track_branch).real, 1, nperseg=100, nfft=omegas.size/2, window=('gaussian', 2/(prop.field)))
#     Z1 = np.abs(Z1)**2
#     plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.PowerNorm(gamma=0.6))
#     # plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1,cmap='plasma')
#     plt.colorbar()
#     plt.axvline(x=time1, color='black', linestyle='dashed')
#     for xc in cross_times_up:
#         if xc > time1:
#             plt.axvline(x=xc, color='green', linestyle='dashed')
#     for xc in cross_times_down:
#         if xc >time1:
#             plt.axvline(x=xc, color='red', linestyle='dashed')
#
#     plt.title('STFT Magnitude-Tracking with branch cut')
#     plt.ylim([0, 15])
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time(cycles)')
#     plt.show()

