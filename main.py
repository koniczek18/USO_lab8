import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import signal

def model1(x,t):
    u=1
    z1=x[0]
    z2=x[1]
    z1p=z1*np.log(z2)
    z2p=-z2*np.log(z1)+z2*u
    return np.array([z1p,z2p])

def model1Modified(x,t):
    u=1
    x1=x[0]
    x2=x[1]
    x1p=x2
    x2p=-x1+u
    return np.array([x1p,x2p])

def zadanie1(active):
    if active:
        t = np.linspace(0, 10, 201)
        sim1=odeint(model1,y0=[1,1],t=t)
        if True:
            plt.figure('Non-linear system 1')
            plt.plot(t,sim1[:,0],label='z1')
            plt.plot(t,sim1[:,1],label='z2')
            plt.legend()
        sim2 = odeint(model1Modified, y0=[0, 0], t=t)
        if True:
            plt.figure('Non-linear system 1 Modified')
            if True:
                plt.plot(t,sim2[:,0],label='x1')
                plt.plot(t,sim2[:,1],label='x2')
            if True:
                plt.plot(t,np.exp(sim2[:,0]),label='z1')
                plt.plot(t, np.exp(sim2[:, 1]), label='z2')
            plt.legend()
        plt.show()

def model2(x,t,J,R,m,g,d,u):
    x1=x[0]
    x2=x[1]
    xp1=x2
    xp2=u/J-(d*x2)/J-(m*g*R*np.sin(x1))/J
    return [xp1,xp2]


def zadanie2(active):
    if active:
        J=1
        g=10
        R=1
        m=9
        d=0.5
        t = np.linspace(0, 10, 201)
        A=np.array([[0,1],[-(m*g*R)/J,-d/J]])
        B=np.array([[0],[1/J]])
        C=np.array([1,0])
        D=np.array([[0]])
        pendulum=signal.StateSpace(A,B,C,D)
        if True:
            u = 0
            sim3 = odeint(model2, y0=[0, 0], t=t, args=(J, R, m, g, d, u))
            u_sim2=np.full_like(t,u)
            tout,yout,xout=signal.lsim2(pendulum,u_sim2,t)
            plt.figure('Pendulum for u=0')
            plt.plot(tout,yout,label='lsim2')
            plt.plot(t,sim3[:,0],label='odeint')
            plt.legend()
        if True:
            u = 5
            sim4 = odeint(model2, y0=[0, 0], t=t, args=(J, R, m, g, d, u))
            u_sim2 = np.full_like(t, u)
            tout, yout, xout = signal.lsim2(pendulum, u_sim2, t)
            plt.figure('Pendulum for u=5')
            plt.plot(tout, yout, label='lsim2')
            plt.plot(t, sim4[:, 0], label='odeint')
            plt.legend()
        if True:
            u = 20
            sim5 = odeint(model2, y0=[0, 0], t=t, args=(J, R, m, g, d, u))
            u_sim2 = np.full_like(t, u)
            tout, yout, xout = signal.lsim2(pendulum, u_sim2, t)
            plt.figure('Pendulum for u=20')
            plt.plot(tout, yout, label='lsim2')
            plt.plot(t, sim5[:, 0], label='odeint')
            plt.legend()
        if True:
            u = 45*np.sqrt(2)
            sim6 = odeint(model2, y0=[0, 0], t=t, args=(J, R, m, g, d, u))
            u_sim2 = np.full_like(t, u)
            tout, yout, xout = signal.lsim2(pendulum, u_sim2, t)
            plt.figure('Pendulum for u=45*sqrt(2)')
            plt.plot(tout, yout, label='lsim2')
            plt.plot(t, sim6[:, 0], label='odeint')
            plt.legend()
        if True:
            u = 70
            sim7 = odeint(model2, y0=[0, 0], t=t, args=(J, R, m, g, d, u))
            u_sim2 = np.full_like(t, u)
            tout, yout, xout = signal.lsim2(pendulum, u_sim2, t)
            plt.figure('Pendulum for u=70')
            plt.plot(tout, yout, label='lsim2')
            plt.plot(t, sim7[:, 0], label='odeint')
            plt.legend()
        newA=np.array([[0,1],[-(np.sqrt(2)*m*g*R)/(2*d),-d/J]])
        newB=np.array([[0],[1/J]])
        pendulumMod = signal.StateSpace(newA, newB, C, D)
        if True:
            u = 45 * np.sqrt(2)
            sim8 = odeint(model2, y0=[np.pi/4, 0], t=t, args=(J, R, m, g, d, u))
            u=u- (45 * np.sqrt(2))
            u_sim2 = np.full_like(t, u)
            tout, yout, xout = signal.lsim2(pendulumMod, u_sim2, t,X0=[0,0])
            plt.figure('Modified pendulum for u=45*sqrt(2)')
            plt.plot(tout, yout+(np.pi/4), label='lsim2')
            plt.plot(t, sim8[:, 0], label='odeint')
            plt.legend()
        if True:
            u = (45 * np.sqrt(2)) +2
            sim9 = odeint(model2, y0=[np.pi/4, 0], t=t, args=(J, R, m, g, d, u))
            u = u - (45 * np.sqrt(2))
            u_sim2 = np.full_like(t, u)
            tout, yout, xout = signal.lsim2(pendulumMod, u_sim2, t,X0=[0,0])
            plt.figure('Modified pendulum for u=45*sqrt(2)+2')
            plt.plot(tout, yout+(np.pi/4), label='lsim2')
            plt.plot(t, sim9[:, 0], label='odeint')
            plt.legend()
        if True:
            u = (45 * np.sqrt(2)) +10
            sim10 = odeint(model2, y0=[np.pi/4, 0], t=t, args=(J, R, m, g, d, u))
            u = u - (45 * np.sqrt(2))
            u_sim2 = np.full_like(t, u)
            tout, yout, xout = signal.lsim2(pendulumMod, u_sim2, t,X0=[0,0])
            plt.figure('Modified pendulum for u=45*sqrt(2)+10')
            plt.plot(tout, yout+(np.pi/4), label='lsim2')
            plt.plot(t, sim10[:, 0], label='odeint')
            plt.legend()
        if True:
            u = (45 * np.sqrt(2)) +30
            sim11 = odeint(model2, y0=[np.pi/4, 0], t=t, args=(J, R, m, g, d, u))
            u = u - (45 * np.sqrt(2))
            u_sim2 = np.full_like(t, u)
            tout, yout, xout = signal.lsim2(pendulumMod, u_sim2, t,X0=[0,0])
            plt.figure('Modified pendulum for u=45*sqrt(2)+30')
            plt.plot(tout, yout+(np.pi/4), label='lsim2')
            plt.plot(t, sim11[:, 0], label='odeint')
            plt.legend()
        plt.show()

def SDC(t,xa,xb,u):
    x0=np.array([xa,xb])
    J = 1
    g = 10
    R = 1
    m = 9
    d = 0.5
    plt.figure('SDC')
    for x in range(0,len(t)-1):
        A=np.array([[0,1],[-(m*g*R*np.sin(xa))/(J*xa),-d/J]])
        B=np.array([[0],[1/J]])
        C=np.array([1,0])
        D=np.array([[0]])
        system=signal.StateSpace(A, B, C, D)
        temp=t[x+1]-t[x]
        X0=x0
        tout, yout, xout =signal.lsim2(system,[u,u], [0,t[x+1]-t[x]],X0=x0)
        x0=xout[1,:]
        plt.plot(tout+t[x],yout)
    plt.show()


def zadanie2cont(active):
    if active:
        J = 1
        g = 10
        R = 1
        m = 9
        d = 0.5
        t = np.linspace(0, 10, 201)
        u=0
        sim12=odeint(model2,y0=[np.pi/4, 0], t=t, args=(J, R, m, g, d, u))
        if True:
            plt.figure('u=0')
            plt.plot(t, sim12[:, 0], label='odeint')
            plt.legend()
        plt.show()
        SDC(t,np.pi/4, 0,u)


if __name__ == '__main__':
    zadanie1(True)
    zadanie2(True)
    zadanie2cont(True)