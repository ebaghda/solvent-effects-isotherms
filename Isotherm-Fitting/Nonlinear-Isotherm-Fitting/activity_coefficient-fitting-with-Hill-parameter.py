import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use("../solvent-effects-reactions/style/simple_bold.mplstyle")
fig_dpi = 1200

# Define the Langmuir isotherm function
def langmuir_isotherm(C: float, K, theta_max, gamma, n):
    return (theta_max * (K * gamma * C)**n) / (1 + (K * gamma * C)**n)

# Function to fit Langmuir model and return parameters with errors
def fit_langmuir(C, theta, theta_err, K_fixed=None, gamma_fixed=None, n_fixed=None):
    def model(C, K, theta_max, gamma, n):
        return langmuir_isotherm(C, K, theta_max, gamma, n)

    initial_guess = [1.0, max(theta), 1.0, 2]  # Initial guesses for K, θ_max, γ, and n

    # Fixing K (cosolvent case)
    if K_fixed is not None:
        def model_fixed(C, theta_max, gamma, n):
            return langmuir_isotherm(C, K_fixed, theta_max, gamma, n)
        
        popt, pcov = curve_fit(model_fixed, C, theta, sigma=theta_err, absolute_sigma=True, 
                               p0=[max(theta), 1.0, 2])  # Ensure n is included
        perr = np.sqrt(np.diag(pcov)) / np.sqrt(len(C))  # Standard error
        
        return K_fixed, popt[0], popt[1], popt[2], 0, perr[0], perr[1], perr[2]  # Fixed K, fitted θ_max, γ, n, and errors

    # Fixing gamma (water case, γ = 1)
    elif gamma_fixed is not None:
        def model_fixed(C, K, theta_max, n):  # Ensure n is included
            return langmuir_isotherm(C, K, theta_max, gamma_fixed, n)
        
        popt, pcov = curve_fit(model_fixed, C, theta, sigma=theta_err, absolute_sigma=True, 
                               p0=[1.0, max(theta), 2])  # Ensure n is included
        perr = np.sqrt(np.diag(pcov)) / np.sqrt(len(C))  # Standard error
        
        return popt[0], popt[1], popt[2], gamma_fixed, perr[0], perr[1], perr[2], 0  # Fitted K, θ_max, n, fixed γ, and errors

    else:
        popt, pcov = curve_fit(model, C, theta, sigma=theta_err, absolute_sigma=True, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov)) / np.sqrt(len(C))  # Standard error
        
        return popt[0], popt[1], popt[2], popt[3], perr[0], perr[1], perr[2], perr[3]  # Fitted K, θ_max, γ, n, and errors

from import_data import import_data
C_water, theta_water, theta_err_water, C_cosolvent, theta_cosolvent, theta_err_cosolvent = import_data()

# Fit the water case (γ = 1, variable θ_max)
gamma_fixed = 1
K_water, theta_max_water, n_water, gamma_water, K_err_water, theta_max_err_water, n_err_water, gamma_err_water = fit_langmuir(
    C_water, theta_water, theta_err_water, gamma_fixed=gamma_fixed
)

# Fit the cosolvent case (Fix K from water case, fit γ and θ_max)
K_cosolvent, theta_max_cosolvent, n_cosolvent, gamma_cosolvent, K_err_cosolvent, theta_max_err_cosolvent, n_err_cosolvent, gamma_err_cosolvent = fit_langmuir(
    C_cosolvent, theta_cosolvent, theta_err_cosolvent, K_fixed=K_water
)

# Generate fit curves
C_fit = np.logspace(-4, 2, 100)  # Log-spaced concentration values
theta_fit_water = langmuir_isotherm(C_fit, K_water, theta_max_water, gamma_water, n_water)
theta_fit_cosolvent = langmuir_isotherm(C_fit, K_cosolvent, theta_max_cosolvent, gamma_cosolvent, n_cosolvent)

# Plot results
fig, ax = plt.subplots(figsize=(6, 6))  # Square aspect ratio

# Water case (Dark Blue Circles with Black Outline, Black Error Bars)
ax.errorbar(
    C_water, theta_water, yerr=theta_err_water, fmt='o', 
    markerfacecolor='darkblue', markeredgecolor='black', markersize=8, 
    ecolor='black', capsize=3, label=f"Water"
)
ax.plot(C_fit, theta_fit_water, linestyle="--", color='darkblue',
        label=f"Water Fit (K={K_water:.2f}±{K_err_water:.2f}, θ_max={theta_max_water:.2f}±{theta_max_err_water:.2f}, γ=1, n={n_water:.2f}±{n_err_water:.2f})")

# Cosolvent case (Dark Green Squares with Black Outline, Black Error Bars)
ax.errorbar(
    C_cosolvent, theta_cosolvent, yerr=theta_err_cosolvent, fmt='s', 
    markerfacecolor='green', markeredgecolor='black', markersize=8, 
    ecolor='black', capsize=3, label=f"20 mol% IPA"
)
ax.plot(C_fit, theta_fit_cosolvent, linestyle="--", color='green',
        label=f"20 mol% IPA Fit (K={K_cosolvent:.2f}, θ_max={theta_max_cosolvent:.2f}±{theta_max_err_cosolvent:.2f}, γ={gamma_cosolvent:.2f}±{gamma_err_cosolvent:.2f}, n={n_cosolvent:.2f}±{n_err_cosolvent:.2f})")
ax.annotate(text='20 mol% IPA', xy=(1, 1.1), color='green', weight='bold')
ax.annotate(text='water', xy=(3, 0.70), color='darkblue', weight='bold')

# Log-scale x-axis and labels
ax.set_xscale("log")
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("Formate Concentration C (mM)")
ax.set_ylabel("Apparent Coverage θ (-)")
fig.tight_layout()
#fig.savefig(f"./isotherms_Hill2_{fig_dpi}dpi.png", dpi=fig_dpi)
plt.show()

# Print results
print(f"Water case: K = {K_water:.2f} ± {K_err_water:.2f}, θ_max = {theta_max_water:.2f} ± {theta_max_err_water:.2f}, γ = {gamma_water} (fixed), n = {n_water:.2f} ± {n_err_water:.2f}")
print(f"Cosolvent case: K = {K_cosolvent:.2f} (fixed), θ_max = {theta_max_cosolvent:.3f} ± {theta_max_err_cosolvent:.3f}, γ = {gamma_cosolvent:.2f} ± {gamma_err_cosolvent:.2f}, n = {n_cosolvent:.2f} ± {n_err_cosolvent:.2f}")
