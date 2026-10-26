#!/usr/bin/env python3
"""Audit scenario-ID disjointness only; never instantiate a final environment."""
from pathlib import Path
from datetime import datetime,timezone
import json

ROOT=Path(__file__).resolve().parent
FINAL_BASE=900000000
FINAL_STRIDE=103
FINAL_COUNT=10000


def audit():
 old=(17,29,43);fresh=(17,29,43,71,89)
 known={
  'original_training280':{s*10000+i for s in old for i in range(280)},
  'original_expert25':{s*50000+i for s in old for i in range(25)},
  'original_test30':{300000+97*i for i in range(30)},
  'original_sensitivity20x3':{400000+10000*j+53*i for j in range(3) for i in range(20)},
  'previous_corrected_training2500':{1000000*s+i for s in old for i in range(2500)},
  'previous_corrected_test100':{300000+97*i for i in range(100)},
  'previous_corrected_sensor100':{400000+97*i for i in range(100)},
  'current_development_training20k':{1000000*s+i for s in (101,202) for i in range(20000)},
  'current_validation300_superset':{600000+101*i for i in range(300)},
  'current_final_training20k':{1000000*s+i for s in fresh for i in range(20000)},
  'original_and_new_constructor_seeds':set(old+fresh+(101,202)),
 }
 final={FINAL_BASE+FINAL_STRIDE*i for i in range(FINAL_COUNT)}
 intersections={name:sorted(final&values) for name,values in known.items()}
 if any(intersections.values()):raise ValueError('reserved test collides with known exposure')
 retired={900000+103*i for i in range(10000)}
 return {'status':'DISJOINT','audited_utc':datetime.now(timezone.utc).isoformat(),
  'final_seed_base':FINAL_BASE,'final_seed_stride':FINAL_STRIDE,'final_scenarios':FINAL_COUNT,
  'final_first':min(final),'final_last':max(final),'final_outcomes_accessed':False,
  'known_exposures':{name:{'count':len(values),'minimum':min(values),'maximum':max(values),'intersection':intersections[name]} for name,values in known.items()},
  'prospective_amendment':{'retired_base':900000,'retired_count':10000,'retired_was_not_executed':True,
  'reason':'expanded old proposal overlapped original expert-demonstration scenario1450020; new base chosen before any final outcome',
  'retired_collision_with_original_experts':sorted(retired&known['original_expert25'])},
  'power_amendment':{'count':10000,'bootstrap_replicates':50000,'chosen_before_test':True,
   'development_paired_sd_approx':4.442,'development_gap_approx':.2103,'purpose':'resolve small paired gaps with conservative familywise inference'}}


if __name__=='__main__':
 result=audit();path=ROOT/'protocol/SCENARIO_SEED_AUDIT.json';path.parent.mkdir(exist_ok=True)
 path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
