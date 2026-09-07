"""Shared signal-PDF builders."""

import zfit
from zfit.models.physics import GeneralizedCB

from .parameters import create_signal_parameters


def _generalized_cb(obs, parameters):
    return GeneralizedCB(
        obs=obs,
        mu=parameters["mean"],
        sigmal=parameters["sigmaL"],
        sigmar=parameters["sigmaR"],
        alphal=parameters["alphaL"],
        nl=parameters["nL"],
        alphar=parameters["alphaR"],
        nr=parameters["nR"],
    )


def build_cb_with_gauss(obs, decay, parameters=None):
    parameters = create_signal_parameters(decay, parameters)
    double_cb = _generalized_cb(obs, parameters)
    gauss = zfit.pdf.Gauss(
        obs=obs,
        mu=parameters["mean"],
        sigma=parameters["g_sigma"],
    )
    model = zfit.pdf.SumPDF([double_cb, gauss], [parameters["sig_frac"]])
    return model, parameters


def build_cb_with_double_gauss(obs, decay, parameters=None):
    parameters = create_signal_parameters(decay, parameters)
    double_cb = _generalized_cb(obs, parameters)
    gauss1 = zfit.pdf.Gauss(
        obs=obs,
        mu=parameters["mean"],
        sigma=parameters["g_sigma1"],
    )
    gauss2 = zfit.pdf.Gauss(
        obs=obs,
        mu=parameters["mean"],
        sigma=parameters["g_sigma2"],
    )
    gauss_mix = zfit.pdf.SumPDF([gauss1, gauss2], [parameters["gauss_frac"]])
    model = zfit.pdf.SumPDF([double_cb, gauss_mix], [parameters["sig_frac"]])
    return model, parameters
