"""Base class and shared behavior for decay-specific mass models."""

from abc import ABC, abstractmethod

import numpy as np
import zfit

from .parameters import create_background_parameters, create_yield_parameter


class GenericMassModel(ABC):
    Bd_Bs_mass_shift = 87.45
    decay_type = None

    def __init__(
        self,
        obs,
        tex_decay,
        simulation,
        is_selected,
        n_events,
        shape_parameters=None,
    ):
        self.obs = obs
        self.tex_decay = tex_decay
        self.simulation = simulation
        self.is_selected = is_selected
        self.n_events = n_events
        self.shape_parameters = shape_parameters
        self._context = {}
        self._drawing_backgrounds = {}

    def construct_background_model(self, obs, is_selected, parameters=None):
        parameters = create_background_parameters(
            self.decay_type,
            is_selected,
            parameters,
        )
        if not is_selected:
            model = zfit.pdf.Chebyshev(
                obs=obs,
                coeffs=[parameters[f"c{i}"] for i in range(1, 5)],
            )
        else:
            model = zfit.pdf.Exponential(obs=obs, lambda_=parameters["lambda"])
        return model, parameters

    @classmethod
    def from_decay_type(
        cls,
        obs,
        tex_decay,
        simulation,
        is_selected,
        n_events,
        shape_parameters=None,
    ):
        # Local imports keep the base module independent of its subclasses.
        from .bd2jpsikst import Bd2JpsiKstMassModel
        from .bs2dspi import Bs2DsPiMassModel
        from .bs2jpsikst import Bs2JpsiKstMassModel
        from .bu2jpsik import Bu2JpsiKMassModel

        if r"$B^+" in tex_decay:
            model_class = Bu2JpsiKMassModel
        elif r"$B^{0}_{s} \to D_{s}^{-} \pi^+$" in tex_decay:
            model_class = Bs2DsPiMassModel
        elif r"$B^{0}_{s}" in tex_decay:
            model_class = Bs2JpsiKstMassModel
        elif r"$B^{0}" in tex_decay:
            model_class = Bd2JpsiKstMassModel
        else:
            raise ValueError(f"Unsupported decay type for model construction: {tex_decay}")
        return model_class(
            obs,
            tex_decay,
            simulation,
            is_selected,
            n_events,
            shape_parameters,
        )

    def _start_context(self):
        yield_signal = create_yield_parameter(
            self.decay_type,
            "signal",
            self.n_events,
        )
        signal_model, signal_params = self.construct_signal_model(
            self.obs,
            self.shape_parameters,
        )
        self._context = {
            "yield_signal": yield_signal,
            "signal_params": signal_params,
            "signal_model": signal_model,
        }
        self._drawing_backgrounds = {}
        if self.simulation:
            self._context["model"] = signal_model.create_extended(yield_signal)
        return signal_model, signal_params

    def _add_drawing_background(
        self,
        name,
        model,
        fraction,
        yield_parameter,
        label,
        color,
    ):
        self._drawing_backgrounds[name] = {
            "model": model,
            "frac": fraction,
            "yield": yield_parameter,
            "label": label,
            "color": color,
        }

    def _build_with_combinatorial_background(self, signal_model, yield_key):
        yield_bkg = create_yield_parameter(
            self.decay_type,
            yield_key,
            self.n_events,
        )
        background_model, background_params = self.construct_background_model(
            self.obs,
            self.is_selected,
        )
        self._add_drawing_background(
            "comb_model",
            background_model,
            None,
            yield_bkg,
            "Combinatorial",
            "lightgray",
        )
        self._context.update(
            {
                "yield_bkg": yield_bkg,
                "model": zfit.pdf.SumPDF(
                    [
                        signal_model.create_extended(self._context["yield_signal"]),
                        background_model.create_extended(yield_bkg),
                    ]
                ),
                "background_model": background_model,
                "background_params": background_params,
            }
        )
        return self._context

    @abstractmethod
    def construct_signal_model(self, obs, parameters=None):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError

    def plot_mass_fit(
        self,
        ax1,
        x_plot,
        binwidth,
        model,
        params,
        yield_signal,
        yield_bkg,
    ):
        signal = model if self.simulation else model.models[0]
        signal_pdf_eval = signal.pdf(x_plot, norm_range=self.obs)
        signal_scaled = params[yield_signal]["value"] * signal_pdf_eval * binwidth
        ax1.plot(
            x_plot,
            signal_scaled,
            label=self.tex_decay,
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        if self.simulation:
            return signal_scaled

        background_pdf_evals = {}
        for name, bg_info in self._drawing_backgrounds.items():
            bg_fraction = bg_info["frac"]
            if bg_fraction is None:
                other_fractions = [
                    info["frac"]
                    for other_name, info in self._drawing_backgrounds.items()
                    if other_name != name and info["frac"] is not None
                ]
                fraction = (
                    1 - sum(params[item]["value"] for item in other_fractions)
                    if other_fractions
                    else 1.0
                )
            else:
                fraction = params[bg_fraction]["value"]
            pdf_eval = bg_info["model"].pdf(x_plot, norm_range=self.obs)
            background_pdf_evals[name] = (
                params[yield_bkg]["value"] * fraction * pdf_eval * binwidth
            )

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
        total_scaled = signal_scaled + background_scaled
        ax1.plot(x_plot, total_scaled, label="Total Fit", color="black", linewidth=2)
        return total_scaled
