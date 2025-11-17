#!/usr/bin/env python3

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import time

G = 1.0 #since in code units

class TwoBody:
    def __init__(self, m1=5.0, m2=1.0):
        self.m1,self.m2=m1,m2
    
    def deriv(self, s):#define derivatives
        x1,y1,x2,y2,vx1,vy1,vx2,vy2 = s
        rx,ry=x2-x1,y2-y1
        r2=rx*rx+ry*ry
        r = math.sqrt(r2)#distance
        
        if r<1e-10:#avoid singularity!!
            return np.array([vx1,vy1,vx2,vy2,0,0,0,0])
        
        f=G/(r2*r)#force magnitude
        ax1,ay1=f*self.m2*rx,f*self.m2*ry
        ax2,ay2=-f*self.m1*rx,-f*self.m1*ry
        
        return np.array([vx1,vy1,vx2,vy2,ax1,ay1,ax2,ay2])
    
    def rk4(self, s, dt):#RK4 step
        k1 = self.deriv(s)
        k2 = self.deriv(s + 0.5*dt*k1)
        k3 = self.deriv(s + 0.5*dt*k2)
        k4 = self.deriv(s + dt*k3)
        return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    def rk45(self, s, dt, tol=1e-6):#RK45 step
        # Cash-Karp coeff
        a = [0,1/5,3/10,3/5,1,7/8]
        b = [[0,0,0,0,0,0],
             [1/5,0,0,0,0,0],
             [3/40,9/40,0,0,0,0],
             [3/10,-9/10,6/5,0,0,0],
             [-11/54,5/2,-70/27,35/27,0,0],
             [1631/55296,175/512,575/13824,44275/110592,253/4096,0]]
        
        c4 = [37/378,0,250/621,125/594,0,512/1771]
        c5 = [2825/27648,0,18575/48384,13525/55296,277/14336,1/4]
        
        k = [self.deriv(s)]#first slope
        for i in range(1,6):#compute k2 to k6
            y_temp = s.copy()
            for j in range(i):
                y_temp += dt*b[i][j]*k[j]
            k.append(self.deriv(y_temp))
        
        y4 = s + dt*sum(c4[i]*k[i] for i in range(6))
        y5 = s + dt*sum(c5[i]*k[i] for i in range(6))
        
        err = np.linalg.norm(y5-y4)
        dt_new = 0.9*dt*(tol/err)**(1/5) if err > 0 else 2*dt
        dt_new = max(0.2*dt, min(5*dt, dt_new))
        #accept or reject step(this makes adaptive clever then rk4)
        if err <= tol:#
            return y5, dt, dt_new, err
        else:
            return s, 0.0, dt_new, err
    
    def init_orbit(self, sep=1.0, ecc=0.3):#initial conditions
        mtot = self.m1 + self.m2
        r1, r2 = self.m2*sep/mtot, self.m1*sep/mtot
        
        vc = math.sqrt(G*mtot/sep)
        vf = vc*math.sqrt((1-ecc)/(1+ecc))
        
        return np.array([-r1,0,r2,0,0,-vf*self.m2/mtot,0,vf*self.m1/mtot])
    
    def run_rk4(self, s0, dt=0.005, T=15.0):#RK4 integration
        n = int(T/dt)
        s = s0.copy()
        hist = np.zeros((n+1,8))
        times = np.linspace(0,T,n+1)
        
        hist[0] = s
        t0 = time.time()
        
        for i in range(1,n+1):
            s = self.rk4(s,dt)
            hist[i] = s
        
        return times, hist, {'steps':n, 'time':time.time()-t0}
    
    def run_rk45(self, s0, dt=0.01, T=15.0, tol=5e-7):#RK45 integration
        s = s0.copy()
        t, dt = 0.0, dt
        
        times, hist = [0.0], [s.copy()]
        steps, reject = 0, 0
        
        t0 = time.time()
        while t < T:
            if t+dt > T:
                dt = T-t
            
            new_s, dt_used, dt_next, err = self.rk45(s, dt, tol)#`attempt step
            
            if dt_used > 0:#step accepted
                s = new_s
                t += dt_used
                times.append(t)
                hist.append(s.copy())
                steps += 1
                dt = dt_next
            else:#step rejected
                reject += 1
                dt = dt_next
            
            dt = max(1e-8, min(0.1, dt))
        
        acc_rate = steps/(steps+reject) if steps+reject > 0 else 0
        return np.array(times), np.array(hist), {'steps':steps, 'time':time.time()-t0, 'acc_rate':acc_rate}
    
    def energy(self, s):#calculate energy
        x1,y1,x2,y2,vx1,vy1,vx2,vy2 = s
        ke = 0.5*self.m1*(vx1*vx1+vy1*vy1) + 0.5*self.m2*(vx2*vx2+vy2*vy2)#kinetic energy
        
        rx, ry = x2-x1, y2-y1
        r = math.sqrt(rx*rx + ry*ry)
        pe = -G*self.m1*self.m2/r if r > 1e-10 else 0#potential energy
        
        return ke + pe#total en.
    
    def energy_stats(self, times, hist):
        E = np.array([self.energy(hist[i]) for i in range(len(times))])
        E0 = E[0]
        drift = np.abs(E - E0)
        return {'E':E, 'drift':drift, 'rel_drift':np.max(drift)/abs(E0)}
#now plotting....
def make_plots(sim, rk4_data, rk45_data):#generate plots
    t4, h4, s4 = rk4_data
    t45, h45, s45 = rk45_data
    
    e4 = sim.energy_stats(t4, h4)
    e45 = sim.energy_stats(t45, h45)
    
    # Plot 1: Core analysis (4 plots)
    fig = plt.figure(figsize=(14,9))
    
    # Energy drift
    plt.subplot(2,2,1)
    plt.semilogy(t4, e4['drift'], 'b-', lw=2, label='RK4')
    plt.semilogy(t45, e45['drift'], 'r-', lw=2, label='RK45')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('|Energy Drift|')
    plt.title('Energy Conservation')
    plt.legend()
    
    
    
    plt.tight_layout()
    plt.savefig('LABWORK/core_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    r4 = np.sqrt((h4[:,2]-h4[:,0])**2 + (h4[:,3]-h4[:,1])**2)
    r45 = np.sqrt((h45[:,2]-h45[:,0])**2 + (h45[:,3]-h45[:,1])**2)
    
    # Plot 1: Interbody distance
    fig = plt.figure(figsize=(10,6))
    plt.plot(t4, r4, 'b-', lw=2, label='RK4')
    plt.plot(t45, r45, 'r--', lw=2, label='RK45')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.title('Interbody Distance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('LABWORK/interbody_distance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Phase space
    fig = plt.figure(figsize=(10,6))
    vr4 = np.gradient(r4, t4)
    vr45 = np.gradient(r45, t45)
    plt.plot(r4, vr4, 'b-', lw=2, label='RK4')
    plt.plot(r45, vr45, 'r--', lw=2, label='RK45')
    plt.grid(True, alpha=0.3)
    plt.xlabel('r')
    plt.ylabel('dr/dt')
    plt.title('Phase Space')
    plt.legend()
    plt.tight_layout()
    plt.savefig('LABWORK/phase_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Energy evolution  
    fig = plt.figure(figsize=(10,6))
    plt.plot(t4, e4['E'], 'b-', lw=2, label='RK4')
    plt.plot(t45, e45['E'], 'r--', lw=2, label='RK45')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy Evolution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('LABWORK/energy_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    
    return ['LABWORK/core_analysis.png', 'LABWORK/interbody_distance.png', 'LABWORK/phase_space.png', 'LABWORK/energy_evolution.png', 'LABWORK/summary.png']

def make_anim(t, h, fname='LABWORK/orbit.gif'):
    fig, ax = plt.subplots(figsize=(8,8))
    
    x1, y1, x2, y2 = h[:,0], h[:,1], h[:,2], h[:,3]
    
    # Limits
    all_x, all_y = np.concatenate([x1,x2]), np.concatenate([y1,y2])
    m = 0.1*max(np.ptp(all_x), np.ptp(all_y))
    ax.set_xlim(np.min(all_x)-m, np.max(all_x)+m)
    ax.set_ylim(np.min(all_y)-m, np.max(all_y)+m)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Two-Body Orbit')
    
    # Elements
    trail1, = ax.plot([],[], 'b-', alpha=0.6, lw=2, label='Heavy')
    trail2, = ax.plot([],[], 'r-', alpha=0.6, lw=2, label='Light') 
    body1, = ax.plot([],[], 'bo', ms=10)
    body2, = ax.plot([],[], 'ro', ms=6)
    ax.plot(0,0, 'k+', ms=12, mew=2, label='CM')
    ax.legend()
    
    def animate(i):
        trail1.set_data(x1[:i+1], y1[:i+1])
        trail2.set_data(x2[:i+1], y2[:i+1])
        body1.set_data([x1[i]], [y1[i]])
        body2.set_data([x2[i]], [y2[i]])
        return trail1, trail2, body1, body2
    
    # Optimize frames
    n_frames = min(150, len(x1))
    skip = len(x1)//n_frames
    frames = range(0, len(x1), skip)
    
    anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=True)
    anim.save(fname, writer=PillowWriter(fps=10))
    plt.close()
    return fname

def main():
    print("Two-Body Analysis - RK4 vs RK45")
    print("="*40)
    
    # Setup
    sim = TwoBody(5.0, 1.0)
    s0 = sim.init_orbit(1.0, 0.3)
    T = 15.0
    
    print(f"Config: m1={sim.m1}, m2={sim.m2}, e=0.3, T={T}")
    
    # Run sims
    print("\nRK4 simulation...")
    t4, h4, s4 = sim.run_rk4(s0, 0.005, T)
    print(f"Done: {s4['steps']:,} steps, {s4['time']:.3f}s")
    
    print("\nRK45 simulation...")
    t45, h45, s45 = sim.run_rk45(s0, T=T)
    print(f"Done: {s45['steps']:,} steps, {s45['time']:.3f}s, acc:{s45['acc_rate']:.1%}")
    
    # Analysis
    print("\nGenerating plots...")
    os.makedirs('LABWORK', exist_ok=True)
    plots = make_plots(sim, (t4,h4,s4), (t45,h45,s45))
    
    print("\nCreating animation...")
    anim = make_anim(t45, h45)
    
    # Results
    e4 = sim.energy_stats(t4, h4)
    e45 = sim.energy_stats(t45, h45)
    
    print(f"\nRESULTS:")
    print(f"RK4:  {e4['rel_drift']:.2e} energy drift, {s4['steps']:,} steps")
    print(f"RK45: {e45['rel_drift']:.2e} energy drift, {s45['steps']:,} steps")
    print(f"RK45 is {s4['steps']/s45['steps']:.1f}x more efficient")
    
    print(f"\nFiles:")
    for p in plots + [anim]:
        print(f"  {p}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()