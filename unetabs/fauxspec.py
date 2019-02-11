""" Module to generate fake training samples"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np

from pkg_resources import resource_filename

import numba as nb

import pdb


@nb.jit(nopython=True, cache=True)
def sum_tau(pixx, randt, randx, rands):
    Nline = randt.size
    tau_tot = np.zeros((pixx.size, Nline)) #, dtype=float)
    for iline in range(Nline):
        tau = randt[iline] * np.exp(-(pixx - randx[iline]) ** 2 / 2 / rands[iline] ** 2)
        tau_tot[:,iline] = tau
    #
    return tau_tot

def noisy(nsample=10000, npix=64**2, avg_Nline=50, seed=12345,
          abs_thresh=0.95, verbose=True, tau0=(0.1,10)):
    """
    Generate the simplest, noise random dataset

    See the Simple_Dataset Notebook for further details

    Args:
        nsample:
        npix:
        avg_Nline: int, optional
          On average per spectrum
        seed: int, optional
        tau0 (tuple, optional):  min/max for Uniform draws of tau0
        abs_thresh (float, optional):  Defined threshold without noise for an absorbed pixel
        verbose (bool, optional):

    Returns:

    """
    # Random numbers
    rstate = np.random.RandomState(seed)

    # Random Gaussian parameters
    sigma = (3., 7)  # Uniform
    s2n = (3, 20)  # Uniform; per pixel

    # Init
    all_flux = np.zeros((npix, nsample), dtype=float)
    all_lbls = np.zeros((npix, nsample), dtype=int)
    pixx = np.arange(npix)

    ran_s2n = rstate.uniform(s2n[0], s2n[1], nsample)

    for kk in range(nsample):
        if verbose:
            if (kk%50) == 0:
                print(kk)
        Nline = int(np.round(avg_Nline + np.sqrt(avg_Nline) * rstate.randn(1)))  # Should replace with Poisson?
        # Position
        randx = npix * rstate.rand(Nline)
        # Sigma
        rands = sigma[0] + (sigma[1] - sigma[0]) * rstate.rand(Nline)
        # Tau0
        randt = tau0[0] + (tau0[1] - tau0[0]) * rstate.rand(Nline)
        # Tau
        itau_tot = sum_tau(pixx, randt, randx, rands)
        tau_tot = np.sum(itau_tot, axis=1)

        # Save labels
        tmp_flux = np.exp(-tau_tot)
        all_lbls[tmp_flux < abs_thresh, kk] = 1

        # Add noise -- including it in the core of strong lines which isn't quite right
        noise = rstate.normal(0., 1/ran_s2n[kk], npix)
        all_flux[:, kk] = tmp_flux + noise

    return all_flux, all_lbls, ran_s2n


def simplest(nsample=10000, npix=64**2, avg_Nline=50, seed=12345):
    """
    Generate the simplest, noiseless random dataset

    See the Simple_Dataset Notebook for further details

    Args:
        nsample:
        npix:
        avg_Nline: int, optional
          On average per spectrum
        seed: int, optional

    Returns:

    """
    # Random numbers
    rstate = np.random.RandomState(seed)

    # Random Gaussian parameters
    sigma = (3., 7)  # Uniform
    tau0 = (0.1, 10)  # Uniform

    # Init
    all_flux = np.zeros((npix, nsample), dtype=float)
    all_lbls = np.zeros((npix, nsample), dtype=int)
    pixx = np.arange(npix)

    for kk in range(nsample):
        if (kk%50) == 0:
            print(kk)
        Nline = int(np.round(avg_Nline + np.sqrt(avg_Nline) * rstate.randn(1)))  # Should replace with Poisson?
        # Position
        randx = npix * rstate.rand(Nline)
        # Sigma
        rands = sigma[0] + (sigma[1] - sigma[0]) * rstate.rand(Nline)
        # Tau0
        randt = tau0[0] + (tau0[1] - tau0[0]) * rstate.rand(Nline)
        # Tau
        itau_tot = sum_tau(pixx, randt, randx, rands)
        tau_tot = np.sum(itau_tot, axis=1)

        # Save
        all_flux[:, kk] = np.exp(-tau_tot)
        all_lbls[all_flux[:, kk] < 0.95, kk] = 1


    return all_flux, all_lbls


def main(flg):

    if flg & (2**0):
        print("Generating the simplest training model")
        # Simple first test case
        all_flux, all_lbls = simplest()
        # Write
        np.save(resource_filename('unetabs', 'data/training/simple_flux.npy'), all_flux, allow_pickle=False)
        np.save(resource_filename('unetabs', 'data/training/simple_lbls.npy'), all_lbls, allow_pickle=False)

    if flg & (2**1):
        print("Generating the noisy training model")
        # Simple noise
        all_flux, all_lbls, s2n = noisy(seed=12345)
        # Write
        np.save(resource_filename('unetabs', 'data/training/noisy_flux.npy'), all_flux, allow_pickle=False)
        np.save(resource_filename('unetabs', 'data/training/noisy_lbls.npy'), all_lbls, allow_pickle=False)
        np.save(resource_filename('unetabs', 'data/training/noisy_s2n.npy'), s2n, allow_pickle=False)
        # Test
        print("Generating the noisy test")
        all_flux, all_lbls, s2n = noisy(nsample=1000, seed=1234)
        np.save(resource_filename('unetabs', 'data/training/noisy_test_flux.npy'), all_flux, allow_pickle=False)
        np.save(resource_filename('unetabs', 'data/training/noisy_test_lbls.npy'), all_lbls, allow_pickle=False)
        np.save(resource_filename('unetabs', 'data/training/noisy_test_s2n.npy'), s2n, allow_pickle=False)


# Command line execution
if __name__ == '__main__':

    flg=0
    #flg += 2**0  # Simplest
    flg += 2**1  # Simplest with noise

    main(flg)


