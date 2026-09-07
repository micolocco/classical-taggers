"""Mass model for Bs -> Ds pi, preserving the existing component model."""

import zfit

from .generic_mass_model import GenericMassModel
from .parameters import create_component_parameter, create_yield_parameter
from .shapes import build_cb_with_gauss


class Bs2DsPiMassModel(GenericMassModel):
    decay_type = "Bs2DsPi"

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

        comb_model, background_params = self.construct_background_model(self.obs, True)
        exp_yield = create_yield_parameter(
            self.decay_type,
            "exponential_component",
            self.n_events,
        )
        frac_comb = create_component_parameter(self.decay_type, "comb_fraction")
        self._add_drawing_background(
            "comb_model",
            comb_model,
            frac_comb,
            exp_yield,
            "Combinatorial",
            "lightgray",
        )

        part_yield = create_yield_parameter(
            self.decay_type,
            "partial_component",
            self.n_events,
        )
        frac_part = create_component_parameter(self.decay_type, "part_fraction")
        part_gauss1 = zfit.pdf.Gauss(
            obs=self.obs,
            mu=create_component_parameter(self.decay_type, "part_mean1"),
            sigma=create_component_parameter(self.decay_type, "part_sigma1"),
        )
        part_gauss2 = zfit.pdf.Gauss(
            obs=self.obs,
            mu=create_component_parameter(self.decay_type, "part_mean2"),
            sigma=create_component_parameter(self.decay_type, "part_sigma2"),
        )
        model_part = zfit.pdf.SumPDF(
            [part_gauss1, part_gauss2],
            [create_component_parameter(self.decay_type, "part_ratio")],
        )
        self._add_drawing_background(
            "model2_part",
            model_part,
            frac_part,
            part_yield,
            "Partially Reconstructed",
            "mediumorchid",
        )

        lb_yield = create_yield_parameter(
            self.decay_type,
            "lb_component",
            self.n_events,
        )
        model_lb = zfit.pdf.Gauss(
            obs=self.obs,
            mu=create_component_parameter(self.decay_type, "lb_mean"),
            sigma=create_component_parameter(self.decay_type, "lb_sigma"),
        )
        frac_lb = create_component_parameter(self.decay_type, "lb_fraction")
        self._add_drawing_background(
            "model_Lb",
            model_lb,
            frac_lb,
            lb_yield,
            r"$\Lambda_b^0 \rightarrow \bar{\Lambda_c} \pi^+$",
            "mediumseagreen",
        )

        bd_yield = create_yield_parameter(
            self.decay_type,
            "bd_component",
            self.n_events,
        )
        mean_bd = zfit.ComposedParameter(
            "mu_Bd2DsPi",
            lambda mean: mean - self.Bd_Bs_mass_shift,
            params=signal_params["mean"],
        )
        model_bd, _ = build_cb_with_gauss(
            self.obs,
            self.decay_type,
            {**signal_params, "mean": mean_bd},
        )
        frac_bd = create_component_parameter(self.decay_type, "bd_fraction")
        self._add_drawing_background(
            "model_Bd2DsPi",
            model_bd,
            frac_bd,
            bd_yield,
            r"$B^0 \rightarrow D^+_s \pi^-$",
            "steelblue",
        )

        dsk_yield = create_yield_parameter(
            self.decay_type,
            "dsk_component",
            self.n_events,
        )
        model_dsk = zfit.pdf.Gauss(
            obs=self.obs,
            mu=create_component_parameter(self.decay_type, "dsk_mean"),
            sigma=create_component_parameter(self.decay_type, "dsk_sigma"),
        )
        self._add_drawing_background(
            "model_DsK",
            model_dsk,
            None,
            dsk_yield,
            r"$B_s \rightarrow D_s^- K^+$",
            "goldenrod",
        )

        yield_bkg = create_yield_parameter(
            self.decay_type,
            "background_selected",
            self.n_events,
        )
        background_mix = zfit.pdf.SumPDF(
            [comb_model, model_part, model_lb, model_bd, model_dsk],
            [frac_comb, frac_part, frac_lb, frac_bd],
        )
        model = zfit.pdf.SumPDF(
            [
                signal_model.create_extended(self._context["yield_signal"]),
                background_mix.create_extended(yield_bkg),
            ]
        )
        self._context.update(
            {
                "yield_bkg": yield_bkg,
                "model": model,
                "background_model": background_mix,
                "background_params": background_params,
                "model_part": model_part,
                "exp_yield": exp_yield,
                "part_yield": part_yield,
            }
        )
        return self._context
