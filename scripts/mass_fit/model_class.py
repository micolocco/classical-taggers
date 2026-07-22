from abc import ABC, abstractmethod

import numpy as np
import zfit
from zfit.models.physics import GeneralizedCB

#TODO 
# -make default parameters more centralized
# -test whether this works at all
# -go from particle based model to decay based model directly
# -add Bs2DsPi model


def _build_bd_sigpeak(self, obs, parameters=None): # doubleCB with 1 gaussian
    default_parameters = {
        "mean": zfit.Parameter("mean_bd",    5275.0, 5250.00, 5350.0),
        "sigmaL": zfit.Parameter("sigmaL",     12.0,    0.10,   30.0),
        "sigmaR": zfit.Parameter("sigmaR",     12.0,    0.10,   30.0),
        "g_sigma": zfit.Parameter("g_sigma",    6.0,    0.10,   30.0),
        "sig_frac": zfit.Parameter("sig_frac",  0.5,    0.00,    1.0),
        "alphaL": zfit.Parameter("alphaL",      3.0,    0.00,    5.0),
        "nL": zfit.Parameter("nL",              1.5,    0.01,   15.0),
        "alphaR": zfit.Parameter("alphaR",      3.0,    0.00,    5.0),
        "nR": zfit.Parameter("nR",              1.6,    0.01,   15.0),
    }
    parameters = self._merge_parameters(default_parameters, parameters)

    signal_mean = parameters["mean"]
    double_cb = GeneralizedCB(
        obs=obs,
        mu=signal_mean,
        sigmal=parameters["sigmaL"],
        sigmar=parameters["sigmaR"],
        alphal=parameters["alphaL"],
        nl=parameters["nL"],
        alphar=parameters["alphaR"],
        nr=parameters["nR"],
    )
    gauss = zfit.pdf.Gauss(obs=obs, mu=signal_mean, sigma=parameters["g_sigma"])
    return zfit.pdf.SumPDF([double_cb, gauss], [parameters["sig_frac"]]), parameters

def _build_bu_sigpeak(self, obs, parameters=None): # doubleCB with 2 gaussian
    default_parameters = {
        "mean":       zfit.Parameter("mean_bu",  5275.0, 5250.00, 5350.0),
        "sigmaL":     zfit.Parameter("sigmaL",     12.0,    0.10,   30.0),
        "sigmaR":     zfit.Parameter("sigmaR",     12.0,    0.10,   30.0),
        "g_sigma1":   zfit.Parameter("g_sigma1",    6.0,    0.10,   30.0),
        "g_sigma2":   zfit.Parameter("g_sigma2",   18.0,    0.10,   50.0),
        "sig_frac":   zfit.Parameter("sig_frac",    0.5,    0.00,    1.0),
        "gauss_frac": zfit.Parameter("gauss_frac",  0.5,    0.00,    1.0),
        "alphaL":     zfit.Parameter("alphaL",      3.0,    0.00,    5.0),
        "nL":         zfit.Parameter("nL",          1.5,    0.01,   15.0),
        "alphaR":     zfit.Parameter("alphaR",      3.0,    0.00,    5.0),
        "nR":         zfit.Parameter("nR",          1.6,    0.01,   15.0),
    }
    parameters = self._merge_parameters(default_parameters, parameters)

    signal_mean = parameters["mean"]
    double_cb = GeneralizedCB(
        obs=obs,
        mu=signal_mean,
        sigmal=parameters["sigmaL"],
        sigmar=parameters["sigmaR"],
        alphal=parameters["alphaL"],
        nl=parameters["nL"],
        alphar=parameters["alphaR"],
        nr=parameters["nR"],
    )
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=signal_mean, sigma=parameters["g_sigma1"])
    gauss2 = zfit.pdf.Gauss(obs=obs, mu=signal_mean, sigma=parameters["g_sigma2"])
    gauss_mix = zfit.pdf.SumPDF([gauss1, gauss2], [parameters["gauss_frac"]])
    return zfit.pdf.SumPDF([double_cb, gauss_mix], [parameters["sig_frac"]]), parameters


class GenericMassModel(ABC):
	Bd_Bs_mass_shift = 87.45

	def __init__(self, obs, tex_decay, simulation, is_selected, n_events, tail_parameters=None):
		self.obs = obs
		self.tex_decay = tex_decay
		self.simulation = simulation
		self.is_selected = is_selected
		self.n_events = n_events
		self.tail_parameters = tail_parameters

	@staticmethod
	def _merge_parameters(default_parameters, parameters):
		if parameters is None:
			return default_parameters
		merged_parameters = default_parameters.copy()
		merged_parameters.update(parameters)
		return merged_parameters

	@staticmethod
	def construct_background_model(obs, is_selected, parameters=None):
		if not is_selected:
			default_parameters = {
				"c1": zfit.Parameter("c1", 0.05),
				"c2": zfit.Parameter("c2", -0.2),
				"c3": zfit.Parameter("c3", 0.0),
				"c4": zfit.Parameter("c4", 0.01),
			}
			parameters = GenericMassModel._merge_parameters(default_parameters, parameters)
			return zfit.pdf.Chebyshev(obs=obs, coeffs=[parameters["c1"], parameters["c2"], parameters["c3"], parameters["c4"]]), parameters

		default_parameters = {"lambda": zfit.Parameter("lambda", -0.01, -1.0, -5e-4)}
		parameters = GenericMassModel._merge_parameters(default_parameters, parameters)
		return zfit.pdf.Exponential(obs=obs, lambda_=parameters["lambda"]), parameters

	@classmethod
	def from_decay_type(cls, obs, tex_decay, simulation, is_selected, n_events, tail_parameters=None):
		if r"$B^+" in tex_decay:
			return BuMassModel(obs, tex_decay, simulation, is_selected, n_events, tail_parameters)
		if r"$B^{0}_{s}" in tex_decay:
			return BsMassModel(obs, tex_decay, simulation, is_selected, n_events, tail_parameters)
		if r"$B^{0}" in tex_decay:
			return BdMassModel(obs, tex_decay, simulation, is_selected, n_events, tail_parameters)
		raise ValueError(f"Unsupported decay type for model construction: {tex_decay}")



	def build_bs_region_context(self, signal_params, yield_signal, yield_bkg, lower_bound=5300):
		pass

	def build(self):
		pass

	def plot_mass_fit(self, ax1, x_plot, binwidth, model, params, yield_signal, yield_bkg):
		pass


class BuMassModel(GenericMassModel):
	def build(self):
		self._context = {"yield_signal": zfit.Parameter("yield_signal", self.n_events, 0, self.n_events * 1.01),}
		signal_model, signal_params = self._build_bu_sigpeak(self.obs, self.tail_parameters)
		self._context["signal_model"] = signal_model
		self._context["signal_params"] = signal_params

		if self.simulation:
			self._context["model"] = signal_model.create_extended(self._context["yield_signal"])
			return self._context

		yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.1, 0, self.n_events)
		background_model, background_params = self.construct_background_model(self.obs, self.is_selected)
		self._context.update(
			{
				"yield_bkg": yield_bkg,
				"model": zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_model.create_extended(yield_bkg)]),
				"background_model": background_model,
				"background_params": background_params,
			}
		)
		return self._context

	def plot_mass_fit(self, ax1, x_plot, binwidth, model, params, yield_signal, yield_bkg):
		signal = model if self.simulation else model.models[0]
		background = None if self.simulation else model.models[1]
		signal_pdf_eval = signal.pdf(x_plot, norm_range=self.obs)
		signal_scaled = params[yield_signal]["value"] * signal_pdf_eval * binwidth
		if self.simulation:
			ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
			return signal_scaled

		background_pdf_eval = background.pdf(x_plot, norm_range=self.obs)
		background_scaled = params[yield_bkg]["value"] * background_pdf_eval * binwidth
		ax1.fill_between(x_plot, background_scaled, label="Combinatorial", color="lightgray", linewidth=2)
		ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
		ax1.plot(x_plot, signal_scaled + background_scaled, label="Total Fit", color="darkred", linewidth=3)
		return signal_scaled + background_scaled


class BdMassModel(GenericMassModel):
    def build(self):
        self._context = {"yield_signal": zfit.Parameter("yield_signal", self.n_events, 0, self.n_events * 1.01),}
        signal_model, signal_params = self._build_bd_sigpeak(self.obs, self.tail_parameters)
        self._context["signal_model"] = signal_model
        self._context["signal_params"] = signal_params

        if self.simulation:
            self._context["model"] = signal_model.create_extended(self._context["yield_signal"])
            return self._context

        if not self.is_selected:
            yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.5, 0, self.n_events)
            background_model, background_params = self.construct_background_model(self.obs, False)
            self._context.update(
                {
                    "yield_bkg": yield_bkg,
                    "model": zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_model.create_extended(yield_bkg)]),
                    "background_model": background_model,
                    "background_params": background_params,
                }
            )
            return self._context

        #Create Bs peak which is same as Bd but shifted and scaled
        signal_params = self._context["signal_params"]
        shifted_parameters = signal_params.copy()
        mean_bs = zfit.ComposedParameter("mean_bs", lambda mean: mean + self.Bd_Bs_mass_shift, params=signal_params["mean"])
        shifted_parameters["mean"] = mean_bs
        secondary_model, _ = self._build_b0_pdf(self.obs, shifted_parameters)

        yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.1, 0, self.n_events)
        frac_secondary = zfit.Parameter("frac_secondary", 0.1, 0, 1)
        background_model, background_params = self.construct_background_model(self.obs, True)
        background_mix = zfit.pdf.SumPDF([secondary_model, background_model], [frac_secondary])
        self._context.update(
            {
                "yield_bkg": yield_bkg,
                "model": zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_mix.create_extended(yield_bkg)]),
                "background_model": background_mix,
                "background_params": background_params,
                "secondary_model": secondary_model,
                "frac_secondary": frac_secondary,
                "use_secondary": True,
            }
        )
        return self._context

    def plot_mass_fit(self, ax1, x_plot, binwidth, model, params, yield_signal, yield_bkg):
        signal = model if self.simulation else model.models[0]
        background = None if self.simulation else model.models[1]
        signal_pdf_eval = signal.pdf(x_plot, norm_range=self.obs)
        signal_scaled = params[yield_signal]["value"] * signal_pdf_eval * binwidth
        if self.simulation:
            ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
            return signal_scaled

        if not self.is_selected:
            background_pdf_eval = background.pdf(x_plot, norm_range=self.obs)
            background_scaled = params[yield_bkg]["value"] * background_pdf_eval * binwidth
            ax1.fill_between(x_plot, background_scaled, label="Combinatorial", color="lightgray", linewidth=2)
            ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
            ax1.plot(x_plot, signal_scaled + background_scaled, label="Total Fit", color="darkred", linewidth=3)
            return signal_scaled + background_scaled

        secondary_model = background.models[0]
        comb_model = background.models[1]
        frac_secondary = params["frac_secondary"]["value"]
        secondary_pdf_eval = secondary_model.pdf(x_plot, norm_range=self.obs)
        background_pdf_eval = comb_model.pdf(x_plot, norm_range=self.obs)
        secondary_scaled = params[yield_bkg]["value"] * frac_secondary * secondary_pdf_eval * binwidth
        background_scaled = params[yield_bkg]["value"] * (1 - frac_secondary) * background_pdf_eval * binwidth
        secondary_label = r"$B^{0}_{s} \to J/\psi K^*$" if r"$B^{0} \to" in self.tex_decay else r"$B^{0} \to J/\psi K^*$"
        ax1.fill_between(x_plot, background_scaled, label="Combinatorial", color="lightgray", linewidth=2)
        ax1.fill_between(x_plot, background_scaled, background_scaled + secondary_scaled, label=secondary_label, color="goldenrod", linewidth=2)
        ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
        ax1.plot(x_plot, signal_scaled + background_scaled + secondary_scaled, label="Total Fit", color="darkred", linewidth=3)
        return signal_scaled + background_scaled + secondary_scaled


class BsMassModel(GenericMassModel):
	def construct_signal_model(self, obs, parameters=None):
		base_signal_model, parameters = self._build_b0_pdf(obs, parameters)
		if self.tex_decay.startswith(r"$B^{0}_{s}"):
			shifted_parameters = parameters.copy()
			mean_bs = zfit.ComposedParameter("mean_bs", lambda mean: mean + self.Bd_Bs_mass_shift, params=parameters["mean"])
			shifted_parameters["mean"] = mean_bs
			return self._build_b0_pdf(obs, shifted_parameters)
		return base_signal_model, parameters

	def construct_secondary_model(self, obs, parameters=None):
		base_signal_model, parameters = self._build_b0_pdf(obs, parameters)
		shifted_parameters = parameters.copy()
		mean_bs = zfit.ComposedParameter("mean_bs", lambda mean: mean + self.Bd_Bs_mass_shift, params=parameters["mean"])
		shifted_parameters["mean"] = mean_bs
		shifted_signal_model, _ = self._build_b0_pdf(obs, shifted_parameters)
		if self.tex_decay.startswith(r"$B^{0}_{s}"):
			return shifted_signal_model, base_signal_model, parameters
		return base_signal_model, shifted_signal_model, parameters

	def build(self):
		self._context = self._context = {"yield_signal": zfit.Parameter("yield_signal", self.n_events, 0, self.n_events * 1.01),}
		base_signal_model, shifted_signal_model, signal_params = self.construct_secondary_model(self.obs, self.tail_parameters)
		self._context["signal_params"] = signal_params
		if self.tex_decay.startswith(r"$B^{0}_{s}"):
			signal_model = shifted_signal_model
			secondary_model = base_signal_model
		else:
			signal_model = base_signal_model
			secondary_model = shifted_signal_model

		self._context["signal_model"] = signal_model

		if self.simulation:
			self._context["model"] = signal_model.create_extended(self._context["yield_signal"])
			return self._context

		if not self.is_selected:
			yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.5, 0, self.n_events)
			background_model, background_params = self.construct_background_model(self.obs, False)
			self._context.update(
				{
					"yield_bkg": yield_bkg,
					"model": zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_model.create_extended(yield_bkg)]),
					"background_model": background_model,
					"background_params": background_params,
				}
			)
			return self._context

		yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.1, 0, self.n_events)
		frac_secondary = zfit.Parameter("frac_secondary", 0.4, 0, 1)
		background_model, background_params = self.construct_background_model(self.obs, True)
		background_mix = zfit.pdf.SumPDF([secondary_model, background_model], [frac_secondary])
		self._context.update(
			{
				"yield_bkg": yield_bkg,
				"model": zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_mix.create_extended(yield_bkg)]),
				"background_model": background_mix,
				"background_params": background_params,
				"secondary_model": secondary_model,
				"frac_secondary": frac_secondary,
				"use_secondary": True,
			}
		)
		return self._context

	def plot_mass_fit(self, ax1, x_plot, binwidth, model, params, yield_signal, yield_bkg):
		signal = model if self.simulation else model.models[0]
		background = None if self.simulation else model.models[1]
		signal_pdf_eval = signal.pdf(x_plot, norm_range=self.obs)
		signal_scaled = params[yield_signal]["value"] * signal_pdf_eval * binwidth
		if self.simulation:
			ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
			return signal_scaled

		if not self.is_selected:
			background_pdf_eval = background.pdf(x_plot, norm_range=self.obs)
			background_scaled = params[yield_bkg]["value"] * background_pdf_eval * binwidth
			ax1.fill_between(x_plot, background_scaled, label="Combinatorial", color="lightgray", linewidth=2)
			ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
			ax1.plot(x_plot, signal_scaled + background_scaled, label="Total Fit", color="darkred", linewidth=3)
			return signal_scaled + background_scaled

		secondary_model = background.models[0]
		comb_model = background.models[1]
		frac_secondary = params["frac_secondary"]["value"]
		secondary_pdf_eval = secondary_model.pdf(x_plot, norm_range=self.obs)
		background_pdf_eval = comb_model.pdf(x_plot, norm_range=self.obs)
		secondary_scaled = params[yield_bkg]["value"] * frac_secondary * secondary_pdf_eval * binwidth
		background_scaled = params[yield_bkg]["value"] * (1 - frac_secondary) * background_pdf_eval * binwidth
		secondary_label = r"$B^{0} \to J/\psi K^*$" if self.tex_decay.startswith(r"$B^{0}_{s}") else r"$B^{0}_{s} \to J/\psi K^*$"
		ax1.fill_between(x_plot, background_scaled, label="Combinatorial", color="lightgray", linewidth=2)
		ax1.fill_between(x_plot, background_scaled, background_scaled + secondary_scaled, label=secondary_label, color="goldenrod", linewidth=2)
		ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
		ax1.plot(x_plot, signal_scaled + background_scaled + secondary_scaled, label="Total Fit", color="darkred", linewidth=3)
		return signal_scaled + background_scaled + secondary_scaled

	def build_bs_region_context(self, signal_params, yield_signal, yield_bkg, mass_range, lower_bound=5300):
		obs_restricted = zfit.Space("mass", limits=(lower_bound, mass_range[1]))

		model_B0, signal_params = self._build_b0_pdf(obs_restricted, signal_params)
		mean_bs = zfit.ComposedParameter("mean_bs", lambda mean: mean + self.Bd_Bs_mass_shift, params=signal_params["mean"])
		bs_parameters = signal_params.copy()
		bs_parameters["mean"] = mean_bs
		model_Bs, signal_params = self._build_b0_pdf(obs_restricted, bs_parameters)
		background_model, background_params = self.construct_background_model(obs_restricted, True)

		signal_params["mean"].floating = False
		signal_params["sigmaL"].floating = False
		signal_params["sigmaR"].floating = False
		signal_params["g_sigma"].floating = False
		signal_params["sig_frac"].floating = False
		background_params["lambda"].floating = False

		frac_secondary = zfit.Parameter("frac_secondary", 0.4, 0, 1)
		background_mix = zfit.pdf.SumPDF([model_B0, background_model], [frac_secondary])
		model = zfit.pdf.SumPDF([model_Bs.create_extended(yield_signal), background_mix.create_extended(yield_bkg)])

		return {
			"obs": obs_restricted,
			"model": model,
			"signal_params": signal_params,
			"background_model": background_mix,
			"background_params": background_params,
			"frac_secondary": frac_secondary,
			"use_secondary": True,
		}

