from abc import ABC, abstractmethod

import numpy as np
import zfit
import zfit_physics
from zfit.models.physics import GeneralizedCB
import matplotlib.pyplot as plt

#TODO 
# -make default parameters more centralized
# -go from particle based model to decay based model directly
#    -> Naming still missing


def _build_double_CB_with_gauss(self, obs, parameters=None): # doubleCB with 1 gaussian
	default_parameters = {
		"mean": zfit.Parameter("mean_bd",	   5275.0, 5250.00, 5350.0),
		"sigmaL": zfit.Parameter("sigmaL",	   12.0,	  0.10,   30.0),
		"sigmaR": zfit.Parameter("sigmaR",	   12.0,	  0.10,   30.0),
		"g_sigma": zfit.Parameter("g_sigma",	6.0,	  0.10,   30.0),
		"sig_frac": zfit.Parameter("sig_frac",  0.5,	  0.00,	   1.0),
		"alphaL": zfit.Parameter("alphaL",	    3.0,	  0.00,	   5.0),
		"nL": zfit.Parameter("nL",			    1.5,	  0.01,   15.0),
		"alphaR": zfit.Parameter("alphaR",	    3.0,	  0.00,	   5.0),
		"nR": zfit.Parameter("nR",			    1.6,	  0.01,   15.0),
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

def _build_double_CB_with_double_gauss(self, obs, parameters=None): # doubleCB with 2 gaussian
	default_parameters = {
		"mean":	   zfit.Parameter("mean_bu",  5275.0, 5250.00, 5350.0),
		"sigmaL":	 zfit.Parameter("sigmaL",	 12.0,	0.10,   30.0),
		"sigmaR":	 zfit.Parameter("sigmaR",	 12.0,	0.10,   30.0),
		"g_sigma1":   zfit.Parameter("g_sigma1",	6.0,	0.10,   30.0),
		"g_sigma2":   zfit.Parameter("g_sigma2",   18.0,	0.10,   50.0),
		"sig_frac":   zfit.Parameter("sig_frac",	0.2,	0.01,	1.0),
		"gauss_frac": zfit.Parameter("gauss_frac",  0.5,	0.00,	1.0),
		"alphaL":	 zfit.Parameter("alphaL",	  3.0,	0.00,	5.0),
		"nL":		 zfit.Parameter("nL",		  1.5,	0.01,   15.0),
		"alphaR":	 zfit.Parameter("alphaR",	  3.0,	0.00,	5.0),
		"nR":		 zfit.Parameter("nR",		  1.6,	0.01,   15.0),
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
		if r"$B^{0}_{s} \to D_{s}^{-} \pi^+$" in tex_decay:
			return Bs2DsPiMassModel(obs, tex_decay, simulation, is_selected, n_events, tail_parameters)
		if r"$B^{0}_{s}" in tex_decay:
			return BsMassModel(obs, tex_decay, simulation, is_selected, n_events, tail_parameters)
		if r"$B^{0}" in tex_decay:
			return BdMassModel(obs, tex_decay, simulation, is_selected, n_events, tail_parameters)
		raise ValueError(f"Unsupported decay type for model construction: {tex_decay}")

	def build_bs_region_context(self, signal_params, yield_signal, yield_bkg, lower_bound=5300):
		pass

	@abstractmethod
	def construct_signal_model(self, obs, parameters=None):
		raise NotImplementedError()

	def build(self):
		pass

	def plot_mass_fit(self, ax1, x_plot, binwidth, model, params, yield_signal, yield_bkg):
		pass


class BuMassModel(GenericMassModel):
	def construct_signal_model(self, obs, parameters=None):
		default_parameters = {
			"mean": zfit.Parameter("mean_bu", 5275.0, 5250.00, 5350.0),
		}
		parameters = self._merge_parameters(default_parameters, parameters)
		signal_model, signal_params = _build_double_CB_with_double_gauss(self, obs, parameters)
		return signal_model, signal_params

	def build(self):
		self._context = {"yield_signal": zfit.Parameter("yield_signal", self.n_events, 0, self.n_events * 1.01)}
		signal_model, signal_params = self.construct_signal_model(self.obs, self.tail_parameters)
		self._context["signal_params"] = signal_params
		self._context["signal_model"] = signal_model
		self._drawing_backgrounds = {}

		if self.simulation:
			self._context["model"] = signal_model.create_extended(self._context["yield_signal"])
			return self._context

		yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.1, 0, self.n_events)
		background_model, background_params = self.construct_background_model(self.obs, self.is_selected)
		self._drawing_backgrounds["comb_model"] = {
			"model": background_model,
			"frac": None,
			"yield": yield_bkg,
			"label": "Combinatorial",
			"color": "lightgray",
		}
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
		signal_pdf_eval = signal.pdf(x_plot, norm_range=self.obs)
		signal_scaled = params[yield_signal]["value"] * signal_pdf_eval * binwidth

		ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
		if self.simulation:
			return signal_scaled

		background_pdf_evals = {}
		for name, bg_info in self._drawing_backgrounds.items():
			bg_model = bg_info["model"]
			bg_frac = bg_info["frac"]
			if bg_frac is None:
				other_fracs = [info["frac"] for n, info in self._drawing_backgrounds.items() if n != name and info["frac"] is not None]
				frac = 1 - sum([params[frac]["value"] for frac in other_fracs]) if other_fracs else 1.0
			else:
				frac = params[bg_frac]["value"]
			bg_pdf_eval = bg_model.pdf(x_plot, norm_range=self.obs)
			background_pdf_evals[name] = params[yield_bkg]["value"] * frac * bg_pdf_eval * binwidth

		background_scaled = np.zeros_like(x_plot)
		for name, bg_scaled in background_pdf_evals.items():
			ax1.fill_between(
				x_plot,
				background_scaled,
				background_scaled + bg_scaled,
				label=self._drawing_backgrounds[name]["label"],
				color=self._drawing_backgrounds[name]["color"],
				linewidth=0,
				alpha=0.8,
			)
			background_scaled += bg_scaled
		ax1.plot(x_plot, signal_scaled + background_scaled, label="Total Fit", color="black", linewidth=2)
		return signal_scaled + background_scaled


class BdMassModel(GenericMassModel):
	def construct_signal_model(self, obs, parameters=None):
		default_parameters = {
			"mean": zfit.Parameter("mean_bd", 5275.0, 5250.00, 5350.0),
		}
		parameters = self._merge_parameters(default_parameters, parameters)
		signal_model, signal_params = _build_double_CB_with_gauss(self, obs, parameters)
		return signal_model, signal_params

	def build(self):
		self._context = {"yield_signal": zfit.Parameter("yield_signal", self.n_events, 0, self.n_events * 1.01)}
		signal_model, signal_params = self.construct_signal_model(self.obs, self.tail_parameters)
		self._context["signal_params"] = signal_params
		self._context["signal_model"] = signal_model
		self._drawing_backgrounds = {}

		if self.simulation:
			self._context["model"] = signal_model.create_extended(self._context["yield_signal"])
			return self._context

		if not self.is_selected:
			yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.5, 0, self.n_events)
			background_model, background_params = self.construct_background_model(self.obs, False)
			self._drawing_backgrounds["comb_model"] = {
				"model": background_model,
				"frac": None,
				"yield": yield_bkg,
				"label": "Combinatorial",
				"color": "lightgray",
			}
			self._context.update(
				{
					"yield_bkg": yield_bkg,
					"model": zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_model.create_extended(yield_bkg)]),
					"background_model": background_model,
					"background_params": background_params,
				}
			)
			return self._context

		mean_bs = zfit.ComposedParameter("mean_bs", lambda mean: mean + self.Bd_Bs_mass_shift, params=signal_params["mean"])
		shifted_parameters = signal_params.copy()
		shifted_parameters["mean"] = mean_bs
		secondary_model, _ = _build_double_CB_with_gauss(self, self.obs, shifted_parameters)

		yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.1, 0, self.n_events)
		frac_secondary = zfit.Parameter("frac_secondary", 0.1, 0, 1)
		background_model, background_params = self.construct_background_model(self.obs, True)
		background_mix = zfit.pdf.SumPDF([secondary_model, background_model], [frac_secondary])
		self._drawing_backgrounds["comb_model"] = {
			"model": background_model,
			"frac": None,
			"yield": yield_bkg,
			"label": "Combinatorial",
			"color": "lightgray",
		}
		self._drawing_backgrounds["model_bs"] = {
			"model": secondary_model,
			"frac": frac_secondary,
			"yield": yield_bkg,
			"label": r"$B^{0}_{s} \to J/\psi K^*$",
			"color": "goldenrod",
		}
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
		signal_pdf_eval = signal.pdf(x_plot, norm_range=self.obs)
		signal_scaled = params[yield_signal]["value"] * signal_pdf_eval * binwidth

		ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
		if self.simulation:
			return signal_scaled

		background_pdf_evals = {}
		for name, bg_info in self._drawing_backgrounds.items():
			bg_model = bg_info['model']
			bg_frac = bg_info['frac']

			print(f"Drawing background: {name}, Fraction: {bg_frac}, color: {bg_info['color']}, label: {bg_info['label']}")

			if bg_frac is None:
				# Get the fraction as 1 - all other fractions
				other_fracs = [info['frac'] for n, info in self._drawing_backgrounds.items() if n != name and info['frac'] is not None]
				if other_fracs:
					frac = 1 - sum([params[frac]["value"] for frac in other_fracs])
				else:
					frac = 1.0
			else: 
				frac = params[bg_frac]["value"]

			print(f"Background: {name}, Fraction: {frac}")

			bg_pdf_eval = bg_model.pdf(x_plot, norm_range=self.obs)
			background_pdf_evals[name] = params[yield_bkg]["value"] * frac * bg_pdf_eval * binwidth

		

		background_scaled = np.zeros_like(x_plot)
		for name, bg_scaled in background_pdf_evals.items():
			ax1.fill_between(x_plot, background_scaled, background_scaled + bg_scaled, label=self._drawing_backgrounds[name]['label'], color=self._drawing_backgrounds[name]['color'], linewidth=0, alpha=0.8)
			background_scaled += bg_scaled
		ax1.plot(x_plot, signal_scaled + background_scaled, label="Total Fit", color="black", linewidth=2)



		return signal_scaled + background_scaled

class BsMassModel(GenericMassModel):
	def construct_signal_model(self, obs, parameters=None):
		default_parameters = {
			"mean": zfit.Parameter("mean_bs", 5366.0, 5340.00, 5380.0),
		}
		parameters = self._merge_parameters(default_parameters, parameters)
		signal_model, signal_params = _build_double_CB_with_gauss(self, obs, parameters)
		return signal_model, signal_params

	def construct_secondary_model(self, obs, parameters=None):
		base_signal_model, signal_params = self.construct_signal_model(obs, parameters)
		mean_bd = zfit.ComposedParameter("mean_bd_from_bs", lambda mean: mean - self.Bd_Bs_mass_shift, params=signal_params["mean"])
		shifted_parameters = signal_params.copy()
		shifted_parameters["mean"] = mean_bd
		secondary_model, _ = _build_double_CB_with_gauss(self, obs, shifted_parameters)
		return base_signal_model, secondary_model, signal_params

	def build(self):
		self._context = {"yield_signal": zfit.Parameter("yield_signal", self.n_events, 0, self.n_events * 1.01)}
		signal_model, secondary_model, signal_params = self.construct_secondary_model(self.obs, self.tail_parameters)
		self._context["signal_params"] = signal_params
		self._context["signal_model"] = signal_model
		self._drawing_backgrounds = {}

		if self.simulation:
			self._context["model"] = signal_model.create_extended(self._context["yield_signal"])
			return self._context

		if not self.is_selected:
			yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.5, 0, self.n_events)
			background_model, background_params = self.construct_background_model(self.obs, False)
			self._drawing_backgrounds["comb_model"] = {
				"model": background_model,
				"frac": None,
				"yield": yield_bkg,
				"label": "Combinatorial",
				"color": "lightgray",
			}
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
		self._drawing_backgrounds["model_bd"] = {
			"model": secondary_model,
			"frac": frac_secondary,
			"yield": yield_bkg,
			"label": r"$B^{0} \to J/\psi K^*$",
			"color": "goldenrod",
		}
		self._drawing_backgrounds["comb_model"] = {
			"model": background_model,
			"frac": None,
			"yield": yield_bkg,
			"label": "Combinatorial",
			"color": "lightgray",
		}
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
		signal_pdf_eval = signal.pdf(x_plot, norm_range=self.obs)
		signal_scaled = params[yield_signal]["value"] * signal_pdf_eval * binwidth

		ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
		if self.simulation:
			return signal_scaled

		background_pdf_evals = {}
		for name, bg_info in self._drawing_backgrounds.items():
			bg_model = bg_info["model"]
			bg_frac = bg_info["frac"]
			if bg_frac is None:
				other_fracs = [info["frac"] for n, info in self._drawing_backgrounds.items() if n != name and info["frac"] is not None]
				frac = 1 - sum([params[frac]["value"] for frac in other_fracs]) if other_fracs else 1.0
			else:
				frac = params[bg_frac]["value"]
			bg_pdf_eval = bg_model.pdf(x_plot, norm_range=self.obs)
			background_pdf_evals[name] = params[yield_bkg]["value"] * frac * bg_pdf_eval * binwidth

		background_scaled = np.zeros_like(x_plot)
		for name, bg_scaled in background_pdf_evals.items():
			ax1.fill_between(
				x_plot,
				background_scaled,
				background_scaled + bg_scaled,
				label=self._drawing_backgrounds[name]["label"],
				color=self._drawing_backgrounds[name]["color"],
				linewidth=0,
				alpha=0.8,
			)
			background_scaled += bg_scaled
		ax1.plot(x_plot, signal_scaled + background_scaled, label="Total Fit", color="black", linewidth=2)
		return signal_scaled + background_scaled

	def build_bs_region_context(self, signal_params, yield_signal, yield_bkg, mass_range, lower_bound=5300):
		obs_restricted = zfit.Space("mass", limits=(lower_bound, mass_range[1]))

		model_Bs, signal_params = _build_double_CB_with_gauss(self, obs_restricted, signal_params)
		mean_bd = zfit.ComposedParameter("mean_bd_region", lambda mean: mean - self.Bd_Bs_mass_shift, params=signal_params["mean"])
		bd_parameters = signal_params.copy()
		bd_parameters["mean"] = mean_bd
		model_Bd, _ = _build_double_CB_with_gauss(self, obs_restricted, bd_parameters)
		background_model, background_params = self.construct_background_model(obs_restricted, True)

		for key in ["mean", "sigmaL", "sigmaR", "g_sigma", "sig_frac"]:
			if key in signal_params:
				signal_params[key].floating = False
		background_params["lambda"].floating = False

		frac_secondary = zfit.Parameter("frac_secondary", 0.4, 0, 1)
		background_mix = zfit.pdf.SumPDF([model_Bd, background_model], [frac_secondary])
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

class Bs2DsPiMassModel(GenericMassModel):
	def construct_signal_model(self, obs, parameters=None):
		default_parameters = {
			"mean":	   zfit.Parameter("mean_bs", 5366.0,  5340.0, 5380.0),
			"g_sigma": zfit.Parameter("g_sigma",   30.0,	 0.1,   100.0),
		}

		parameters = self._merge_parameters(default_parameters, parameters)


		signal_model, signal_params = _build_double_CB_with_gauss(self, self.obs, parameters)
		return signal_model, signal_params

	def build(self):
		self._context = {"yield_signal": zfit.Parameter("yield_signal", self.n_events, 0, self.n_events * 1.01),}
		signal_model, signal_params = self.construct_signal_model(self.obs, self.tail_parameters)
		self._context["signal_params"] = signal_params
		self._context["signal_model"] = signal_model
		self._drawing_backgrounds = {}

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


		#Exponential contribution of background
		comb_model, background_params = self.construct_background_model(self.obs, True, )
		exp_yield = zfit.Parameter("exp_yield", self.n_events * 0.5, 0, self.n_events)
		frac_comb = zfit.Parameter("frac_comb", 0.3, 0.2, .40)
		self._drawing_backgrounds['comb_model'] = {'model': comb_model, 'frac' : frac_comb, 'yield': exp_yield, 'label': 'Combinatorial', 'color': 'lightgray'}


		#2 Gaussian for partially reconstructed background
		part_yield = zfit.Parameter("part_yield", self.n_events * 0.2, 0, self.n_events)	
		frac_part = zfit.Parameter("frac_part", 0.32, 0.3, .55)
		mu1_part = zfit.Parameter("mu1_part", 5130, 5075, 5175)
		sigma1_part = zfit.Parameter("sigma1_part", 100, 15, 200)
		part_gaus1 = zfit.pdf.Gauss(obs=self.obs, mu=mu1_part, sigma=sigma1_part)
		part_ratio = zfit.Parameter("part_ratio", 0.2, 0.01, .99)
		mu2_part = zfit.Parameter("mu2_part", 5200, 5100, 5300)
		sigma2_part = zfit.Parameter("sigma2_part", 50, 15, 200)
		part_gaus2 = zfit.pdf.Gauss(obs=self.obs, mu=mu2_part, sigma=sigma2_part)

		model_part = zfit.pdf.SumPDF([part_gaus1, part_gaus2], [part_ratio])

		self._drawing_backgrounds['model2_part'] = {'model': model_part, 'frac' : frac_part, 'yield': part_yield, 'label': 'Partially Reconstructed', 'color': 'mediumorchid'}






		#Gaussian for Lambda_b background
		Lb_yield = zfit.Parameter("Lb_yield", self.n_events * 0.1, 0, self.n_events)
		mu_Lb = zfit.Parameter("mu_Lb", 5480, 5450, 5550)
		sigma1_Lb = zfit.Parameter("sigma1_Lb", 25, 10, 100)
		sigma2_Lb = zfit.Parameter("sigma2_Lb", 70, 10, 100)

		Lb_ratio = zfit.Parameter("Lb_ratio", 0.5, 0.01, .99)
		gauss1_Lb = zfit.pdf.Gauss(obs=self.obs, mu=mu_Lb, sigma=sigma1_Lb)
		gauss2_Lb = zfit.pdf.Gauss(obs=self.obs, mu=mu_Lb, sigma=sigma2_Lb)
		# model_Lb = zfit.pdf.SumPDF([gauss1_Lb, gauss2_Lb], [Lb_ratio])
		model_Lb = gauss1_Lb

		frac_Lb = zfit.Parameter("frac_Lb", 0.03, 0.02, .15)
		self._drawing_backgrounds['model_Lb'] = {'model': model_Lb, 'frac' : frac_Lb, 'yield': Lb_yield, 'label': r'$\Lambda_b^0 \rightarrow \bar{\Lambda_c} \pi^+$', 'color': 'mediumseagreen'}


		#Bd2DsPi contribution. Same as signal just shifted and scaled
		Bd2DsPi_yield = zfit.Parameter("Bd2DsPi_yield", self.n_events * 0.1, 0, self.n_events)
		mu_Bd2DsPi = zfit.ComposedParameter("mu_Bd2DsPi", lambda mean: mean - self.Bd_Bs_mass_shift, params=signal_params["mean"])
		shifted_parameters = signal_params.copy()
		shifted_parameters["mean"] = mu_Bd2DsPi
		model_Bd2DsPi, _ = _build_double_CB_with_gauss(self, self.obs, shifted_parameters)
		frac_Bd2DsPi = zfit.Parameter("frac_Bd2DsPi", 0.05, 0.02, .08)
		self._drawing_backgrounds['model_Bd2DsPi'] = {'model': model_Bd2DsPi, 'frac' : frac_Bd2DsPi, 'yield': Bd2DsPi_yield, 'label': r'$B^0 \rightarrow D^+_s \pi^-$', 'color': 'steelblue'}
		# self._drawing_backgrounds['model_Bd2DsPi'] = {'model': model_Bd2DsPi, 'frac' : None, 'yield': Bd2DsPi_yield, 'label': r'$B^0 \rightarrow D^+_s \pi^-$', 'color': 'steelblue'}


		#Bs2DsK contribution
		Bs2DsK_yield = zfit.Parameter("Bs2DsK_yield", self.n_events * 0.5, 0, self.n_events)
		mu_DsK = zfit.Parameter("mu_DsK", 5320, 5310, 5340)
		sigma1_DsK = zfit.Parameter("sigma1_DsK", 10, 10, 25)
		sigma2_DsK = zfit.Parameter("sigma2_DsK", 50, 10, 80)
		DsK_ratio = zfit.Parameter("DsK_ratio", 0.5, 0.01, .99)
		gauss1_DsK = zfit.pdf.Gauss(obs=self.obs, mu=mu_DsK, sigma=sigma1_DsK)
		gauss2_DsK = zfit.pdf.Gauss(obs=self.obs, mu=mu_DsK, sigma=sigma2_DsK)

		model_DsK = gauss1_DsK

		self._drawing_backgrounds['model_DsK'] = {'model': model_DsK, 'frac' : None, 'yield': Bs2DsK_yield, 'label': r'$B_s \rightarrow D_s^- K^+$', 'color': 'goldenrod'}

		# background_mix = zfit.pdf.SumPDF([comb_model, model_part, model_Lb, model_DsK ], [exp_yield, part_yield, Lb_yield, Bs2DsK_yield])
		# model =zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), comb_model.create_extended(exp_yield), model_part.create_extended(part_yield), model_Lb.create_extended(Lb_yield), model_DsK.create_extended(Bs2DsK_yield)])


		yield_bkg = zfit.Parameter("yield_bkg", self.n_events * 0.1, 0, self.n_events)
		# background_mix = zfit.pdf.SumPDF([comb_model, model_part, model_Lb, model_Bd, model_DsK ], [frac_part, frac_Lb, frac_Bd, frac_Bs2DsK])
		# model = zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_mix.create_extended(yield_bkg)])


		# background_mix = zfit.pdf.SumPDF([comb_model, model_part, model_Lb, model_Bd2DsPi, model_DsK ], [frac_comb, frac_part, frac_Lb, frac_Bd2DsPi, frac_Bd2DsPi])
		# model = zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_mix.create_extended(yield_bkg)])

		background_mix = zfit.pdf.SumPDF([comb_model, model_part, model_Lb, model_Bd2DsPi, model_DsK ], [frac_comb, frac_part, frac_Lb, frac_Bd2DsPi])
		# background_mix = zfit.pdf.SumPDF([comb_model, model_part, model_Bd2DsPi, model_DsK ], [frac_comb, frac_part, frac_Bd2DsPi])
		# background_mix = zfit.pdf.SumPDF([comb_model, model_part, model_Lb, model_Bd2DsPi,], [frac_comb, frac_part, frac_Lb])
		model = zfit.pdf.SumPDF([signal_model.create_extended(self._context["yield_signal"]), background_mix.create_extended(yield_bkg)])

		
		self._context.update(
			{
				"yield_bkg": yield_bkg,
				"model": model,
				"background_model": background_mix,
				"background_params": background_params,
				# "const_model": const,
				"model_part": model_part,
				# "model_Lb": model_Lb,
				# "model_Bd": model_Bd,
				# "model_DsK": model_DsK,
				"exp_yield": exp_yield,
				# "const_yield": const_yield,
				"part_yield": part_yield,
				# "model2_part": model2_part,
				# "part2_yield": part2_yield,
				# "Lb_yield": Lb_yield,
				# "Bd_yield": Bd_yield,
				# "Bs2DsK_yield": Bs2DsK_yield,
			}
		)
		return self._context
	
	def plot_mass_fit(self, ax1, x_plot, binwidth, model, params, yield_signal, yield_bkg):
		signal = model if self.simulation else model.models[0]
		signal_pdf_eval = signal.pdf(x_plot, norm_range=self.obs)
		signal_scaled = params[yield_signal]["value"] * signal_pdf_eval * binwidth

		ax1.plot(x_plot, signal_scaled, label=self.tex_decay, color="blue", linestyle="--", linewidth=2)
		if self.simulation:
			return signal_scaled

		background_pdf_evals = {}
		for name, bg_info in self._drawing_backgrounds.items():
			bg_model = bg_info['model']
			bg_frac = bg_info['frac']

			if bg_frac is None:
				# Get the fraction as 1 - all other fractions
				other_fracs = [info['frac'] for n, info in self._drawing_backgrounds.items() if n != name and info['frac'] is not None]
				if other_fracs:
					frac = 1 - sum([params[frac]["value"] for frac in other_fracs])
				else:
					frac = 1.0
			else: 
				frac = params[bg_frac]["value"]

			print(f"Background: {name}, Fraction: {frac}")

			bg_pdf_eval = bg_model.pdf(x_plot, norm_range=self.obs)
			background_pdf_evals[name] = params[yield_bkg]["value"] * frac * bg_pdf_eval * binwidth

		

		background_scaled = np.zeros_like(x_plot)
		for name, bg_scaled in background_pdf_evals.items():
			ax1.fill_between(x_plot, background_scaled, background_scaled + bg_scaled, label=self._drawing_backgrounds[name]['label'], color=self._drawing_backgrounds[name]['color'], linewidth=0, alpha=0.8)
			background_scaled += bg_scaled
		ax1.plot(x_plot, signal_scaled + background_scaled, label="Total Fit", color="black", linewidth=2)



		return signal_scaled + background_scaled