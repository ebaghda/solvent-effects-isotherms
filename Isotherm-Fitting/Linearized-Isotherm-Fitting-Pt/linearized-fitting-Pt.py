import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
plt.style.use("../solvent-effects-reactions/style/simple_bold.mplstyle")
fig_dpi = 1200

from import_data import import_data
C_water, theta_water, theta_err_water, C_cosolvent, theta_cosolvent, theta_err_cosolvent = import_data()

# Linearize the isotherm data to plot C/theta vs C for both the Water case data and cosolvent case data with propagated std. err.
water_C_over_theta = C_water/theta_water
cosolvent_C_over_theta = C_cosolvent/theta_cosolvent
propagated_coverage_error_water = water_C_over_theta*np.sqrt((theta_err_water/theta_water)**2)
propagated_coverage_error_cosolvent = cosolvent_C_over_theta*np.sqrt((theta_err_cosolvent/theta_cosolvent)*2)

# Fit water data, defining gamma = 1 to determine K
popt, pcov_water = curve_fit(lambda C, K, theta_max: C/theta_max+1/(theta_max*K), C_water, water_C_over_theta, sigma=propagated_coverage_error_water)
K_fit_water, theta_max_fit_water = popt
K_fit_water_SE, theta_max_fit_water_SE = np.sqrt(np.diag(pcov_water))/np.sqrt(len(C_water))
print(f"K = {K_fit_water:.3f}, θ_max = {theta_max_fit_water:.3f}")
fitted_curve = C_water/theta_max_fit_water+1/(theta_max_fit_water*K_fit_water)
r2_sklearn_water = r2_score(water_C_over_theta, fitted_curve)
print(f"Water R-squared: {r2_sklearn_water:.6f}")

plt.figure(figsize=(5,5))
plt.plot(C_water, fitted_curve, 'darkgray', zorder=0)
plt.errorbar(C_water, water_C_over_theta, yerr=propagated_coverage_error_water, ecolor='k', elinewidth=1, capsize=5, fmt='none', zorder=1)
plt.scatter(C_water, water_C_over_theta, c='darkblue', edgecolors='k', s=35)
plt.xlabel("Concentration C (mM)")
plt.ylabel(r"Concentration/Apparent Coverage C/θ (mM)")
plt.annotate(f"Pt, Water", (0.06, 0.91), font = "Arial", fontweight='bold', xycoords="axes fraction", fontsize=16)
plt.annotate(f"K = {K_fit_water:.1f} ± {K_fit_water_SE:0.1f} $mM^{{-1}}$\n$θ_{{max}}$ = {theta_max_fit_water:0.3f} ± {theta_max_fit_water_SE:.3f}\nCOD = {r2_sklearn_water:0.5f}", (0.06, 0.73), font = "Arial", xycoords="axes fraction", fontsize=14)
plt.annotate(r"$slope = \frac{1}{\theta_{max} }$", xy=(0.52,0.25), usetex=True, fontsize=18, xycoords="axes fraction")
plt.annotate(r"$intercept = \frac{1}{\theta_{max}K_{water}}$", xy=(0.4,0.1), usetex=True, fontsize=18, xycoords="axes fraction")
plt.savefig(f"./Isotherm-Fitting/Linearized-Isotherm-Fitting-Pt/linearized_isotherm_water_{fig_dpi}dpi.png", dpi=fig_dpi, bbox_inches='tight')
plt.show()
plt.close()

#Fit cosolvent data, using the water-case K to determine gamma
popt_cosolvent, pcov_cosolvent = curve_fit(lambda C, gamma, theta_max: C/theta_max+1/(theta_max*gamma*K_fit_water), C_cosolvent, cosolvent_C_over_theta, sigma=propagated_coverage_error_cosolvent)
gamma_fit_cosolvent, theta_max_fit_cosolvent = popt_cosolvent
gamma_fit_cosolvent_SE, theta_max_fit_cosolvent_SE = np.sqrt(np.diag(pcov_cosolvent))/np.sqrt(len(C_cosolvent))
print(f"gamma = {gamma_fit_cosolvent:.3f} ± {gamma_fit_cosolvent_SE:0.3f}, θ_max = {theta_max_fit_cosolvent:.3f} ± {theta_max_fit_water_SE:0.3}")
fitted_curve_cosolvent = C_cosolvent/theta_max_fit_cosolvent+1/(theta_max_fit_cosolvent*gamma_fit_cosolvent*K_fit_water)
r2_sklearn_cosolvent = r2_score(cosolvent_C_over_theta, fitted_curve_cosolvent)
print(f"Cosolvent R-squared: {r2_sklearn_cosolvent:.5f}")

plt.figure(figsize=(5,5))
plt.plot(C_cosolvent, fitted_curve_cosolvent, 'darkgray', zorder=0)
plt.errorbar(C_cosolvent, cosolvent_C_over_theta, yerr=propagated_coverage_error_cosolvent, ecolor='k', elinewidth=1, capsize=5, fmt="none", zorder=1)
plt.scatter(C_cosolvent, cosolvent_C_over_theta, c='darkgreen', marker='s', edgecolors='k', s=35)
plt.xlabel("Concentration C (mM)")
plt.ylabel("Concentration/Apparent Coverage C/θ (mM)")
plt.annotate(f"Pt, 20 mole% IPA", (0.06, 0.91), font = "Arial", fontweight='bold', xycoords="axes fraction", fontsize=16)
plt.annotate(f"γ = {gamma_fit_cosolvent:.2f} ± {gamma_fit_cosolvent_SE:0.2f}\n$θ_{{max}}$ = {theta_max_fit_cosolvent:0.3f} ± {theta_max_fit_cosolvent_SE:.3f}\nCOD = {r2_sklearn_cosolvent:0.5f}", (0.06, 0.73), font = "Arial", xycoords="axes fraction", fontsize=14)

plt.annotate(r"$slope = \frac{1}{\theta_{max} }$", xy=(0.52,0.25), usetex=True, fontsize=18, xycoords="axes fraction")
plt.annotate(r"$intercept = \frac{1}{\theta_{max}\gamma K_{water}}$", xy=(0.4,0.1), usetex=True, fontsize=18, xycoords="axes fraction")


plt.savefig(f"./Isotherm-Fitting/Linearized-Isotherm-Fitting-Pt/linearized_isotherm_cosolvent_{fig_dpi}dpi.png", dpi=fig_dpi, bbox_inches='tight')
plt.show()
plt.close()


