import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

class FieldDisplay:
    def __init__(self, maxSize_m, dx, y_min, y_max, probePos, sourcePos):
        plt.ion()
        self.probePos = probePos
        self.sourcePos = sourcePos
        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot(np.arange(0, maxSize_m, dx), [0]*int(maxSize_m/dx))[0]
        self.ax.plot(probePos*dx, 0, 'xr')
        self.ax.plot(sourcePos*dx, 0, 'ok')
        self.ax.set_xlim(0, maxSize_m)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel('Ez, В/м')
        self.ax.grid()
        
    def updateData(self, data):
        self.line.set_ydata(data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Probe:
    def __init__(self, probePos, maxTime, dt):
        self.maxTime = maxTime
        self.dt = dt
        self.probePos = probePos
        self.t = 0
        self.E = np.zeros(self.maxTime)
        
    def addData(self, data):
        self.E[self.t] = data[self.probePos]
        self.t += 1

def showProbeSignal(probe, y_min, y_max):
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, probe.maxTime*probe.dt, probe.dt), probe.E)
        ax.set_xlabel('t, c')
        ax.set_ylabel('Ez, В/м')
        ax.set_xlim(0, probe.maxTime*probe.dt)
        ax.grid()
        fig.show()

def showProbeSpectrum(probe):
    sp = np.abs(fft(probe.E))
    sp = fftshift(sp)
    df = 1/(probe.maxTime*probe.dt)
    freq = np.arange(-probe.maxTime*df /2, probe.maxTime*df/2, df)
    fig, ax = plt.subplots()
    ax.plot(freq, sp/max(sp))
    ax.set_xlabel('f, Гц')
    ax.set_ylabel('|S|/|Smax|')
    ax.set_xlim(-probe.maxTime*df /2, probe.maxTime*df/2)
    ax.grid()
    fig.show()
    
        
eps = 7
W0 = 120*np.pi
maxTime = 1200
maxSize_m = 3
dx = maxSize_m/750
maxSize = int(maxSize_m/dx)
probePos = int(maxSize_m/5*3/dx)
sourcePos = int(maxSize_m/2/dx)
Sc = 1
dt = dx*np.sqrt(eps)*Sc/3e8
probe = Probe(probePos, maxTime, dt)
display = FieldDisplay(maxSize_m, dx, -1.5, 1.5, probePos, sourcePos)
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)
Sc1 = Sc/np.sqrt(eps)
k1 = -1 / (1 / Sc1 + 2 + Sc1)
k2 = 1 / Sc1 - 2 + Sc1
k3 = 2 * (Sc1 - 1 / Sc1)
k4 = 4 * (1 / Sc1 + Sc1)
Ezq_old1 = np.zeros(3)
Ezq_old2 = np.zeros(3)
A_max = 1e5
F_max = 2.5e9
w_g = np.sqrt(np.log(A_max)) / (np.pi * F_max)/dt
d_g = w_g * np.sqrt(np.log(A_max))
for q in range(1, maxTime):
    Hy[1:] = Hy[1:] +(Ez[:-1]-Ez[1:])*Sc/W0
    Ez[:-1] = Ez[:-1] + (Hy[:-1] - Hy[1:])*Sc*W0/eps
    Ez[sourcePos] += np.exp(-((q - d_g) / w_g) ** 2)
    Ez[0] = (k1*(k2*(Ez[2]+Ezq_old2[0])+k3*(Ezq_old1[0]+Ezq_old1[2]-Ez[1]-Ezq_old2[1])-k4*Ezq_old1[1])-Ezq_old2[2])
    Ezq_old2[:] = Ezq_old1[:]
    Ezq_old1[:] = Ez[:3]
    probe.addData(Ez)
    if q % 2 == 0:
        display.updateData(Ez)

showProbeSignal(probe, -1, 1)
showProbeSpectrum(probe)




