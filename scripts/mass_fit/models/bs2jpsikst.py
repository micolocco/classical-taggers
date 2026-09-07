"""Mass model for the Bs -> J/psi K* decay."""

import zfit

from .generic_mass_model import GenericMassModel
from .parameters import create_component_parameter, create_yield_parameter
from .shapes import build_cb_with_gauss


class Bs2JpsiKstMassModel(GenericMassModel):
    decay_type = "Bs2JpsiKst"

    def construct_signal_model(self, obs, parameters=None):
        return build_cb_with_gauss(obs, self.decay_type, parameters)

    def construct_secondary_model(self, obs, parameters=None):
        signal_model, signal_params = self.construct_signal_model(obs, parameters)
        mean_bd = zfit.ComposedParameter(
            "mean_bd_from_bs",
            lambda mean: mean - self.Bd_Bs_mass_shift,
            params=signal_params["mean"],
        )
        shifted_parameters = {**signal_params, "mean": mean_bd}
        secondary_model, _ = build_cb_with_gauss(
            obs,
            self.decay_type,
            shifted_parameters,
        )
        return signal_model, secondary_model, signal_params

    def build(self):
        signal_model, signal_params = self._start_context()
        if self.simulation:
            return self._context

        mean_bd = zfit.ComposedParameter(
            "mean_bd_from_bs",
            lambda mean: mean - self.Bd_Bs_mass_shift,
            params=signal_params["mean"],
        )
        shifted_parameters = {**signal_params, "mean": mean_bd}
        secondary_model, _ = build_cb_with_gauss(
            self.obs,
            self.decay_type,
            shifted_parameters,
        )
        if not self.is_selected:
            return self._build_with_combinatorial_background(
                signal_model,
                "background_unselected",
            )

        yield_bkg = create_yield_parameter(
            self.decay_type,
            "background_selected",
            self.n_events,
        )
        frac_secondary = create_component_parameter(
            self.decay_type,
            "secondary_fraction",
        )
        background_model, background_params = self.construct_background_model(
            self.obs,
            True,
        )
        background_mix = zfit.pdf.SumPDF(
            [secondary_model, background_model],
            [frac_secondary],
        )
        self._add_drawing_background(
            "model_bd",
            secondary_model,
            frac_secondary,
            yield_bkg,
            r"$B^{0} \to J/\psi K^*$",
            "goldenrod",
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
                        background_mix.create_extended(yield_bkg),
                    ]
                ),
                "background_model": background_mix,
                "background_params": background_params,
                "secondary_model": secondary_model,
                "frac_secondary": frac_secondary,
                "use_secondary": True,
            }
        )
        return self._context

    def build_bs_region_context(
        self,
        signal_params,
        yield_signal,
        yield_bkg,
        mass_range,
        lower_bound=5300,
    ):
        obs_restricted = zfit.Space("mass", limits=(lower_bound, mass_range[1]))
        model_bs, signal_params = build_cb_with_gauss(
            obs_restricted,
            self.decay_type,
            signal_params,
        )
        mean_bd = zfit.ComposedParameter(
            "mean_bd_region",
            lambda mean: mean - self.Bd_Bs_mass_shift,
            params=signal_params["mean"],
        )
        model_bd, _ = build_cb_with_gauss(
            obs_restricted,
            self.decay_type,
            {**signal_params, "mean": mean_bd},
        )
        background_model, background_params = self.construct_background_model(
            obs_restricted,
            True,
        )
        for key in ["mean", "sigmaL", "sigmaR", "g_sigma", "sig_frac"]:
            signal_params[key].floating = False
        background_params["lambda"].floating = False

        frac_secondary = create_component_parameter(
            self.decay_type,
            "secondary_fraction",
        )
        background_mix = zfit.pdf.SumPDF(
            [model_bd, background_model],
            [frac_secondary],
        )
        model = zfit.pdf.SumPDF(
            [
                model_bs.create_extended(yield_signal),
                background_mix.create_extended(yield_bkg),
            ]
        )
        return {
            "obs": obs_restricted,
            "model": model,
            "signal_params": signal_params,
            "background_model": background_mix,
            "background_params": background_params,
            "frac_secondary": frac_secondary,
            "use_secondary": True,
        }
