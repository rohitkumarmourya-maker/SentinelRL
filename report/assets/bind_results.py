"""Bind verified completed CSV summaries into report macros; never runs models."""
from pathlib import Path
import csv,json
P=Path(__file__).resolve().parents[2]
R=P/'results/final_benchmark'
manifest=json.loads((R/'manifest.json').read_text())
assert manifest['status']=='COMPLETE' and manifest['rows']==520000 and manifest['episodes']==10000
with (R/'summary_intervals.csv').open() as f: rows=list(csv.DictReader(f))
with (R/'paired_comparisons.csv').open() as f: paired=list(csv.DictReader(f))
lookup={(r['policy'],r['metric']):r for r in rows}
def val(policy,metric):return float(lookup[policy,metric]['mean'])
def iv(policy,metric):return [float(lookup[policy,metric][k]) for k in ['ci95_low','ci95_high']]
def contrast(policy):return next(r for r in paired if r['metric']=='return' and r['comparator']==policy)
primary='champion_temporal_ensemble'
ret=[r for r in paired if r['metric']=='return']
assert len(ret)==55 and sum(r['familywise_lower_positive']=='True' for r in ret)==54
rule=contrast('tuned_belief_threshold');graph=contrast('graph_pure_20k_ensemble');temporal=contrast('graph_pure_temporal_ensemble')
assert [r['comparator'] for r in ret if r['familywise_lower_positive']!='True']==['tuned_belief_threshold']
lo,hi=iv(primary,'return')
coverage=val(primary,'confirmed_fraction')*100
clo,chi=[x*100 for x in iv(primary,'confirmed_fraction')]
family=json.loads((P/'results/CHAMPION_TRAINING_FAMILY.json').read_text())
worker_seconds=sum(json.loads((P/p).read_text())['elapsed_seconds'] for p in family['worker_manifests'].values())
macros={
'AbstractResult':r'On 10,000 paired test scenarios, 54 of 55 Bonferroni-adjusted return contrasts favor the ensemble; the tuned diagnostic rule remains an unresolved rival with a better point estimate.',
'TrainingCompute':f'Logged training totals {worker_seconds/60:.1f} worker-minutes on CPU (five parallel workers, one Torch thread each).',
'InferenceMethod':r'The 55 primary contrasts cover 22 comparator methods plus 33 fixed baseline checkpoints. A 50,000-resample bootstrap shares scenario draws while independently resampling actual comparator seed axes; the ensemble has one fixed-policy row. Bonferroni intervals use $\alpha=.05/55$. Secondary intervals use 2,000 resamples and are exploratory.',
'PrimaryResults':(
 f'The deployment obtains mean return ${val(primary,"return"):.3f}$ (nominal 95\\% CI $[{lo:.3f},{hi:.3f}]$). '
 f'The strongest learned comparator, the final-checkpoint graph ensemble (${val("graph_pure_20k_ensemble","return"):.3f}$), has paired deficit ${float(graph["mean_difference"]):.3f}$ '
 f'(simultaneous CI $[{float(graph["familywise95_low"]):.3f},{float(graph["familywise95_high"]):.3f}]$). '
 f'The matched temporal graph ensemble gives ${val("graph_pure_temporal_ensemble","return"):.3f}$; the paired gain is ${float(temporal["mean_difference"]):.3f}$ '
 f'($[{float(temporal["familywise95_low"]):.3f},{float(temporal["familywise95_high"]):.3f}]$). '
 r'Fifty-four contrasts have positive adjusted lower bounds. '
 f'\n\nThe exception is the tuned diagnostic rule (${val("tuned_belief_threshold","return"):.3f}$): ensemble minus rule is ${float(rule["mean_difference"]):.3f}$ '
 f'(adjusted CI $[{float(rule["familywise95_low"]):.3f},{float(rule["familywise95_high"]):.3f}]$). '
 r'This comparison is inconclusive; universal superiority and equivalence are not established. '
 f'The five individual final models average ${val("champion_individual_20k","return"):.3f}$ with seed SD ${float(lookup["champion_individual_20k","return"]["seed_mean_sd"]):.3f}$; '
 r'they are separate from the deployed ensemble.'),
'SecondaryResults':(
 f'Uncontained hosts average ${val(primary,"uncontained_at_end"):.4f}$ per episode. Explicit-confirmation coverage is only ${coverage:.2f}\\%$ '
 f'(nominal CI $[{clo:.2f},{chi:.2f}]\\%$), versus ${val("previous_alert_rule","confirmed_fraction")*100:.2f}\\%$ for the alert rule. '
 f'Censored confirmation delay is ${val(primary,"censored_confirmation_delay"):.2f}$ versus ${val("previous_alert_rule","censored_confirmation_delay"):.2f}$ steps. '
 r'Low successful confirmation is a substantive weakness despite the return gains. '
 f'The tuned rule has fewer uncontained hosts (${val("tuned_belief_threshold","uncontained_at_end"):.4f}$) but more clean-host isolations '
 f'(${val("tuned_belief_threshold","false_isolations"):.4f}$ versus ${val(primary,"false_isolations"):.4f}$); these are exploratory mean comparisons.'),
}
table=[(primary,'Temporal DDQN'),('tuned_belief_threshold','Tuned diagnostic'),('graph_pure_20k_ensemble','Graph final ens.'),('original_threshold','Original threshold'),('original_full_transfer','Original transfer'),('previous_alert_rule','Alert rule')]
macros['MetricTable']='\n'.join(f'{name} & ${val(p,"return"):.3f}$ & ${val(p,"critical_compromises"):.4f}$ & ${val(p,"false_isolations"):.4f}$ & ${val(p,"confirmed_fraction")*100:.2f}$ \\\\' for p,name in table)
(P/'report/verified_results.tex').write_text('% Generated from completed final evidence by assets/bind_results.py.\n'+'\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in macros.items())+'\n')
(P/'report/RESULT_BINDINGS.json').write_text(json.dumps({'primary':primary,'primary_return':val(primary,'return'),'return_nominal95':iv(primary,'return'),'adjusted_positive':54,'primary_contrasts':55,'rule_exception':rule,'strongest_learned_comparison':graph,'training_worker_seconds':worker_seconds,'table_policy_rows':[p for p,_ in table]},indent=2)+'\n')
print('Bound actual final values:54positive contrasts; tuned-rule comparison unresolved.')
