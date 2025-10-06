# bh_cloud_postproc_fix_efolding.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import csv
import os

M_sun_seconds = 4.925490947e-6    # GM_sun / c^3 in seconds
eV_to_sinv = 1.519267516e15       # 1 eV in s^-1 (1/ħ)
seconds_per_year = 3.154e7

def evolve_cloud_bh(M0_solar,
                    Mcloud_solar,
                    mu_eV,
                    t_end_years,
                    n_points=1000,
                    stop_if_depleted=True,
                    method='RK45'):

    M0_sec = M0_solar * M_sun_seconds
    Mcloud_sec = Mcloud_solar * M_sun_seconds
    mu_sinv = mu_eV * eV_to_sinv
    t_end_sec = t_end_years * seconds_per_year

    def dydt(t, y):
        M_BH = max(y[0], 0.0)
        M_cloud = max(y[1], 0.0)
        if M_cloud <= 0.0:
            return [0.0, 0.0]
        Gamma = 16.0 * (mu_sinv**6) * (M_BH**5)   # s^-1
        dM_BH_dt = Gamma * M_cloud
        dM_cloud_dt = -Gamma * M_cloud
        return [dM_BH_dt, dM_cloud_dt]

    # Event: stop when cloud is depleted (in seconds units)
    def cloud_depleted(t, y):
        tiny_cloud_solar = 1e-15
        tiny_cloud_sec = tiny_cloud_solar * M_sun_seconds
        return y[1] - tiny_cloud_sec
    cloud_depleted.terminal = True
    cloud_depleted.direction = -1

    y0 = [M0_sec, Mcloud_sec]

    sol = solve_ivp(dydt, [0.0, t_end_sec], y0, method=method, rtol=1e-8, atol=1e-15,
                    events=(cloud_depleted if stop_if_depleted else None),
                    dense_output=True, max_step=np.inf)

    # determine final time for evaluation
    if stop_if_depleted and sol.t_events and len(sol.t_events[0]) > 0:
        t_final = sol.t_events[0][0]
    else:
        t_final = sol.t[-2]

    # sample solution
    t_eval = np.linspace(0.0, t_final, n_points)
    y_eval = sol.sol(t_eval)

    # convert sampled arrays to user units
    t_years = t_eval / seconds_per_year
    M_BH_solar = y_eval[0] / M_sun_seconds
    M_cloud_solar = y_eval[1] / M_sun_seconds

    # compute accretion rate in M_sun/yr
    acc_rate_Msun_per_yr = []
    for M_BH_sec, M_cloud_sec in zip(y_eval[0], y_eval[1]):
        #if M_cloud_sec <= 0:
        #    acc_rate_Msun_per_yr.append(0.0)
        #    continue
        Gamma = 16.0 * (mu_sinv**6) * (M_BH_sec**5)
        dMdt_sec_per_sec = Gamma * M_cloud_sec
        dMdt_Msun_per_sec = dMdt_sec_per_sec / M_sun_seconds
        acc_rate_Msun_per_yr.append(dMdt_Msun_per_sec * seconds_per_year)
    acc_rate_Msun_per_yr = np.array(acc_rate_Msun_per_yr)

    # compute e-folding time (years)
    t_efold_years = []
    for M_BH_sec in y_eval[0]:
        if M_BH_sec <= 0:
            t_efold_years.append(np.inf)
            continue
        Gamma = 16.0 * (mu_sinv**6) * (M_BH_sec**5)   # s^-1
        tE = 1.0 / Gamma / seconds_per_year
        t_efold_years.append(tE)
    t_efold_years = np.array(t_efold_years)

    # ----- POSTPROCESSING: extend line aesthetically if solver stopped early -----
    t_end_sec = t_end_years * seconds_per_year
    if sol.t[-1] < t_end_sec:
        tiny = 1e-15
        total_mass = M0_solar + Mcloud_solar
        M_cloud_solar[-1] = tiny
        M_BH_solar[-1] = total_mass - tiny
        #if acc_rate_Msun_per_yr.size > 0:
        #    acc_rate_Msun_per_yr[-1] = 0.0
        #if t_efold_years.size > 0:
            #t_efold_years[-2] = sol.t[-1]/seconds_per_year

        # add extra flat points
        extra_points = 2
        t_extra = np.linspace(t_years[-1], t_years[-1]*1.05, extra_points)
        M_BH_extra = np.full_like(t_extra, M_BH_solar[-1])
        M_cloud_extra = np.full_like(t_extra, tiny)
        acc_extra = np.zeros_like(t_extra)
        efold_extra = np.full_like(t_extra, np.inf)

        t_years = np.concatenate([t_years, t_extra])
        M_BH_solar = np.concatenate([M_BH_solar, M_BH_extra])
        M_cloud_solar = np.concatenate([M_cloud_solar, M_cloud_extra])
        acc_rate_Msun_per_yr = np.concatenate([acc_rate_Msun_per_yr, acc_extra])
        t_efold_years = np.concatenate([t_efold_years, efold_extra])

    info = {
        'sol': sol,
        't_years': t_years,
        'M_BH_solar': M_BH_solar,
        'M_cloud_solar': M_cloud_solar,
        'acc_rate_Msun_per_yr': acc_rate_Msun_per_yr,
        't_efold_years': t_efold_years,
        'mu_sinv': mu_sinv,
        'M0_sec': M0_sec,
        'Mcloud_sec': Mcloud_sec
    }
    return info

def plot_and_save(info, out_prefix='bh_cloud'):
    t = info['t_years']
    M_BH = info['M_BH_solar']
    M_cloud = info['M_cloud_solar']
    acc = info['acc_rate_Msun_per_yr']
    t_efold = info['t_efold_years']
    
    SMALL_SIZE = 16
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 15
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    

    # Mass evolution
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(t, M_cloud,'orange', label='Cloud mass ($M_\\odot$)')
    ax1.plot(t, M_BH,'b', label='BH mass ($M_\\odot$)')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Mass ($M_\\odot$)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_ylim(M0_solar*0.1,Mcloud_solar*10)
    ax1.grid(True, which='both', ls=':')
    ax1.annotate('$\mu=1.73\cdot10^{-17}$ eV', xy=(100, M0_solar*0.2), xytext=(100, M0_solar*0.2))
    fname1 = f'{out_prefix}_masses.png'
    fig1.tight_layout()
    fig1.savefig(fname1, dpi=200)

    # Accretion rate
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(t, 1.0/t_efold)
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Accretion rate ($M_\\odot$/yr)')
    ax2.set_yscale('log')
    ax2.grid(True, ls=':')
    fname2 = f'{out_prefix}_accretion_rate.png'
    fig2.tight_layout()
    fig2.savefig(fname2, dpi=200)

    # E-folding time
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.plot(t, t_efold)
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('E-folding time (years)')
    ax3.set_yscale('log')
    ax3.grid(True, ls=':')
    fname3 = f'{out_prefix}_efolding.png'
    fig3.tight_layout()
    fig3.savefig(fname3, dpi=200)
    
    
    # M\mu time
    fig4, ax4 = plt.subplots(figsize=(8,4))
    ax4.plot(t, M_BH*mu_eV*4.925490947e-6*1.519267516e15)
    ax4.set_xlabel('Time (years)')
    ax4.set_ylabel('$M\\mu$')
    ax4.set_yscale('log')
    ax4.grid(True, ls=':')
    fname4 = f'{out_prefix}_Mmu.png'
    fig4.tight_layout()
    fig4.savefig(fname4, dpi=200)

    # Save CSV
    fname_csv = f'{out_prefix}_evolution.csv'
    with open(fname_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time_years', 'M_BH_Msun', 'M_cloud_Msun', 
                         'acc_rate_Msun_per_yr', 't_efold_years'])
        for tt, mbh, mcl, ar, te in zip(t, M_BH, M_cloud, acc, t_efold):
            writer.writerow([f'{tt:.6e}', f'{mbh:.12e}', f'{mcl:.12e}', 
                             f'{ar:.12e}', f'{te:.12e}'])

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    return fname1, fname2, fname3, fname4, fname_csv

# Run example
if __name__ == "__main__":
    M0_solar = 1.0e3
    Mcloud_solar = 1.0e6
    mu_eV = 1.73e-17
    t_end_years = 1.0e25

    info = evolve_cloud_bh(M0_solar, Mcloud_solar, mu_eV, t_end_years, n_points=1000)
    png1, png2, png3, pn4, csvfile = plot_and_save(info, out_prefix='bh_cloud')

    print("Saved plots:", png1, png2, png3)
    print("Saved CSV:", csvfile)
    print("Final BH mass (M_sun):", info['M_BH_solar'][-1])
    print("Final cloud mass (M_sun):", info['M_cloud_solar'][-1])
    if info['sol'].t_events and len(info['sol'].t_events[0])>0:
        print("Cloud depleted at t =", info['sol'].t_events[0][0]/seconds_per_year, "years")
