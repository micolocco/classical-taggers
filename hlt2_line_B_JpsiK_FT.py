import Functors as F
from Moore.config import register_line_builder
from Moore.lines import Hlt2Line
from RecoConf.reconstruction_objects import (
    make_pvs,
    upfront_reconstruction,
)
from RecoConf.event_filters import require_pvs
from Hlt2Conf.lines.b_to_charmonia.b_to_jpsix import (
    make_BuToJpsimumuKplus_detached_line,
    )
from Hlt2Conf.standard_particles import (
    make_has_rich_long_pions,
    make_has_rich_up_pions
  
)


all_lines = {}

@register_line_builder(all_lines)
def Bu2JpsiK_Jpsi2MuMu_line(name="Hlt2_Bu2JpsiK_Jpsi2MuMu",
                      prescale=1):
    pvs = make_pvs()
    jpsi, kplus, b2jpsikplus = make_BuToJpsimumuKplus_detached_line('hlt2')
    longTaggingParticles = make_has_rich_long_pions()
    upstreamTaggingParticles = make_has_rich_up_pions()


    return Hlt2Line(
        name=name,
        algs=upfront_reconstruction() + [require_pvs(pvs), b2jpsikplus],
        extra_outputs = [('LongTaggingParticles', longTaggingParticles),
        ('UpstreamTaggingParticles', upstreamTaggingParticles)],
        # Passing `prescale` to `Hlt2Line` is optional; The default value is 1
        prescale=prescale,
    )

