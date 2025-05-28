import numpy as np
import matplotlib.pyplot as plt
import emcee

fig_dpi = 1200

def langmuir_isotherm(C, K, theta_max, gamma=1):
    return (theta_max * K * gamma * C) / (1 + K * gamma * C)

# Log-likelihood function (WATER case, gamma fixed at 1)
def log_likelihood_water(params, C, theta, theta_err):
    K, theta_max = params  # Only two fitting parameters
    model = langmuir_isotherm(C, K, theta_max, gamma=1)
    return -0.5 * np.sum(((theta - model) / theta_err) ** 2)

# Log-prior function (WATER case, gamma fixed)
def log_prior_water(params):
    K, theta_max = params
    if 0 < K < 500 and 0 < theta_max < 2:
        return 0.0
    return -np.inf

# Posterior probability (WATER case, gamma fixed)
def log_posterior_water(params, C, theta, theta_err):
    lp = log_prior_water(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_water(params, C, theta, theta_err)

# Log-likelihood function (COSOLVENT case, K fixed)
def log_likelihood_cosolvent(params, C, theta, theta_err, K_fixed):
    theta_max, gamma = params  # Only θ_max and γ are fitted
    model = langmuir_isotherm(C, K_fixed, theta_max, gamma)
    return -0.5 * np.sum(((theta - model) / theta_err) ** 2)

# Log-prior function (COSOLVENT case, K fixed)
def log_prior_cosolvent(params):
    theta_max, gamma = params
    if 0 < theta_max < 2 and 0 < gamma < 20:
        return 0.0
    return -np.inf

# Posterior probability (COSOLVENT case, K fixed)
def log_posterior_cosolvent(params, C, theta, theta_err, K_fixed):
    lp = log_prior_cosolvent(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_cosolvent(params, C, theta, theta_err, K_fixed)

# MCMC runner function
def run_mcmc(log_posterior_func, C, theta, theta_err, initial_guess, nwalkers=100, nsteps=10000, args=()):
    ndim = len(initial_guess)
    p0 = initial_guess + 0.1 * np.random.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_func, args=args)
    sampler.run_mcmc(p0, nsteps, progress=True)
    
    samples = sampler.get_chain(discard=int(0.3 * nsteps), thin=20, flat=True)
    return sampler, samples

# Water case (gamma = 1 fixed)
from import_data import import_data
C_water, theta_water, theta_err_water, C_cosolvent, theta_cosolvent, theta_err_cosolvent = import_data()


water_initial_guess = [10, 1]  # No gamma in fitting
sampler_water, samples_water = run_mcmc(log_posterior_water, C_water, theta_water, theta_err_water, water_initial_guess, args=(C_water, theta_water, theta_err_water))

# Extract best-fit parameters for water
water_params = np.median(samples_water, axis=0)
K_water_fixed = water_params[0]  # Use this in cosolvent case

# Cosolvent case (K fixed from water case)
from import_data import import_data
C_water, theta_water, theta_err_water, C_cosolvent, theta_cosolvent, theta_err_cosolvent = import_data()

cosolvent_initial_guess = [1, 1]  # Only θ_max and γ are fitted
sampler_cosolvent, samples_cosolvent = run_mcmc(log_posterior_cosolvent, C_cosolvent, theta_cosolvent, theta_err_cosolvent, cosolvent_initial_guess, args=(C_cosolvent, theta_cosolvent, theta_err_cosolvent, K_water_fixed))

# Extract best-fit parameters for cosolvent
cosolvent_params = np.median(samples_cosolvent, axis=0)

# Generate fits
C_fit = np.logspace(-4, 2, 1000)

fig, ax = plt.subplots(figsize=(5, 5))

# Plot experimental data
ax.errorbar(C_water, theta_water, yerr=theta_err_water, fmt='o', markerfacecolor='darkblue', markeredgecolor='black', markersize=8, ecolor='black', capsize=3, label="Water")
ax.errorbar(C_cosolvent, theta_cosolvent, yerr=theta_err_cosolvent, fmt='s', markerfacecolor='green', markeredgecolor='black', markersize=8, ecolor='black', capsize=3, label="20 mol% IPA")

# Show Bayesian fitting convergence
num_fits = 20  # Number of intermediate fits to show
indices_water = np.linspace(0, len(samples_water) - 1, num_fits, dtype=int)
indices_cosolvent = np.linspace(0, len(samples_cosolvent) - 1, num_fits, dtype=int)

for i in indices_water:
    theta_fit_intermediate = langmuir_isotherm(C_fit, samples_water[i, 0], samples_water[i, 1], gamma=1)
    ax.plot(C_fit, theta_fit_intermediate, linestyle="--", color='lightsteelblue', alpha=0.8, zorder=0)

for i in indices_cosolvent:
    theta_fit_intermediate = langmuir_isotherm(C_fit, K_water_fixed, samples_cosolvent[i, 0], samples_cosolvent[i, 1])
    ax.plot(C_fit, theta_fit_intermediate, linestyle="--", color='darkseagreen', alpha=0.8, zorder=0)

# Plot final best fits as solid lines
theta_fit_final_water = langmuir_isotherm(C_fit, K_water_fixed, water_params[1], gamma=1)
ax.plot(C_fit, theta_fit_final_water, linestyle="-", color='darkblue', linewidth=1, label="Final Fit (Water)")

theta_fit_final_cosolvent = langmuir_isotherm(C_fit, K_water_fixed, cosolvent_params[0], cosolvent_params[1])
ax.plot(C_fit, theta_fit_final_cosolvent, linestyle="-", color='green', linewidth=1, label="Final Fit (Cosolvent)")

ax.set_xscale("log")
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("Formate Concentration C (mM)")
ax.set_ylabel("Apparent Coverage θ (-)")
ax.legend()

fig.tight_layout()
plt.savefig(f"./Bayesian_isotherms_convergence_{fig_dpi}dpi.png", dpi=fig_dpi)
plt.show()

# Print final parameter estimates
print(f"Water case: K = {K_water_fixed:.2f}, θ_max = {water_params[1]:.2f}, γ = 1 (fixed)")
print(f"Cosolvent case: K = {K_water_fixed:.2f} (fixed), θ_max = {cosolvent_params[0]:.2f}, γ = {cosolvent_params[1]:.2f}")

import corner

# Function to compute parameter estimates and uncertainties
def get_param_estimates(samples, labels):
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    medians = percentiles[1]
    lower_errors = medians - percentiles[0]
    upper_errors = percentiles[2] - medians

    param_estimates = {}
    for i, label in enumerate(labels):
        param_estimates[label] = (medians[i], lower_errors[i], upper_errors[i])
    
    return param_estimates

# Labels for parameters
water_labels = [r"$K$", r"$\theta_{max}$"]
cosolvent_labels = [r"$\theta_{max}$", r"$\gamma$"]

# Compute parameter estimates and uncertainties
water_estimates = get_param_estimates(samples_water, water_labels)
cosolvent_estimates = get_param_estimates(samples_cosolvent, cosolvent_labels)

# Print parameter estimates with uncertainties
print("\nFinal Parameter Estimates with Uncertainties:")
print("Water Case (γ = 1 fixed):")
for label, (median, lower, upper) in water_estimates.items():
    print(f"  {label}: {median:.3f} (+{upper:.3f} / -{lower:.3f})")

print("\nCosolvent Case (K fixed from water fit):")
for label, (median, lower, upper) in cosolvent_estimates.items():
    print(f"  {label}: {median:.3f} (+{upper:.3f} / -{lower:.3f})")

# Generate corner plots
fig_water = corner.corner(samples_water, labels=water_labels, truths=[water_estimates[l][0] for l in water_labels], 
                          quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
fig_water.suptitle("MCMC Posterior Distributions (Water Case)", fontsize=14)

fig_cosolvent = corner.corner(samples_cosolvent, labels=cosolvent_labels, truths=[cosolvent_estimates[l][0] for l in cosolvent_labels], 
                              quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
fig_cosolvent.suptitle("MCMC Posterior Distributions (Cosolvent Case)", fontsize=14)

plt.show()
