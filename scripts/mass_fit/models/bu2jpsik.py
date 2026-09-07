"""Mass model for the Bu -> J/psi K decay."""

from .generic_mass_model import GenericMassModel
from .shapes import build_cb_with_double_gauss


class Bu2JpsiKMassModel(GenericMassModel):
    decay_type = "Bu2JpsiK"

    def construct_signal_model(self, obs, parameters=None):
        return build_cb_with_double_gauss(obs, self.decay_type, parameters)

    def build(self):
        signal_model, _ = self._start_context()
        if self.simulation:
            return self._context
        return self._build_with_combinatorial_background(
            signal_model,
            "background",
        )
