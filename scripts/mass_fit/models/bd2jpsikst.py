"""Mass model for the Bd -> J/psi K* decay."""

import zfit

from .generic_mass_model import GenericMassModel
from .parameters import create_component_parameter, create_yield_parameter
from .shapes import build_cb_with_gauss


class Bd2JpsiKstMassModel(GenericMassModel):
    decay_type = "Bd2JpsiKst"

    def construct_signal_model(self, obs, parameters=None):
        return build_cb_with_gauss(obs, self.decay_type, parameters)

    def build(self):
        signal_model, signal_params = self._start_context()
        if self.simulation:
            return self._context
        if not self.is_selected:
            return self._build_with_combinatorial_background(
                signal_model,
                "background_unselected",
            )

        mean_bs = zfit.ComposedParameter(
            "mean_bs",
            lambda mean: mean + self.Bd_Bs_mass_shift,
            params=signal_params["mean"],
        )
        shifted_parameters = {**signal_params, "mean": mean_bs}
        secondary_model, _ = build_cb_with_gauss(
            self.obs,
            self.decay_type,
            shifted_parameters,
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
            "comb_model",
            background_model,
            None,
            yield_bkg,
            "Combinatorial",
            "lightgray",
        )
        self._add_drawing_background(
            "model_bs",
            secondary_model,
            frac_secondary,
            yield_bkg,
            r"$B^{0}_{s} \to J/\psi K^*$",
            "goldenrod",
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
