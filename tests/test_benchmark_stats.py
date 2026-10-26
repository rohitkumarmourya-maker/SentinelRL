import numpy as np
import pytest
from src.benchmark_stats import bootstrap_families


def test_unequal_seed_axes_exact_constant_effect_and_family_count():
 arrays={'champion':np.full((5,7),3.),'old':np.full((3,7),1.),'rule':np.full((1,7),2.)}
 summaries,paired=bootstrap_families(arrays,'champion',repetitions=100)
 assert all(r['bonferroni_comparators']==2 for r in paired)
 assert next(r for r in paired if r['comparator']=='old')['comparator_seed_rows']==3
 for r in paired:assert r['familywise95_low']==pytest.approx(r['mean_difference']) and r['familywise95_high']==pytest.approx(r['mean_difference'])


def test_common_scenario_pairing_cancels_shared_scenario_noise():
 base=np.arange(20)[None,:]
 _,paired=bootstrap_families({'champion':base+1,'control':base},'champion',repetitions=500)
 assert paired[0]['familywise95_low']==pytest.approx(1.)
 assert paired[0]['familywise95_high']==pytest.approx(1.)
