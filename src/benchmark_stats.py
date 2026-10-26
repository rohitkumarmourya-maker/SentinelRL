"""Crossed seed/scenario bootstrap and conservative familywise return intervals."""
from __future__ import annotations
import numpy as np


def bootstrap_families(arrays, primary, repetitions=20000, alpha=.05, seed=20261026, comparators=None):
    """Shared scenario draws, independent unequal seed axes for every family.

    Arrays are [independent trained-model seed, common scenario]. A deterministic
    comparator has one row, not replicated rows presented as new training runs.
    Intervals for primary-minus-comparator differences use Bonferroni alpha/K.
    """
    arrays={name:np.asarray(value,dtype=float) for name,value in arrays.items()}
    if primary not in arrays or len(arrays)<2:
        raise ValueError('primary and at least one comparator required')
    scenarios={a.shape[1] for a in arrays.values() if a.ndim==2}
    if len(scenarios)!=1 or any(a.ndim!=2 or not a.size or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('finite rectangular arrays with common scenario columns required')
    count=scenarios.pop();rng=np.random.default_rng(seed)
    draws={name:np.empty(repetitions) for name in arrays}
    for first in range(0,repetitions,200):
        size=min(200,repetitions-first)
        scenario_weight=rng.multinomial(count,np.full(count,1/count),size=size)/count
        for name,array in arrays.items():
            nseed=array.shape[0]
            seed_weight=rng.multinomial(nseed,np.full(nseed,1/nseed),size=size)/nseed
            draws[name][first:first+size]=np.einsum('bi,ij,bj->b',seed_weight,array,scenario_weight,optimize=True)
    names=list(comparators) if comparators is not None else [name for name in arrays if name!=primary]
    if len(set(names))!=len(names) or primary in names or any(name not in arrays for name in names) or not names:
        raise ValueError('invalid prespecified comparator set')
    summaries=[];differences=[];family=len(names)
    for name,array in arrays.items():
        lo,hi=np.quantile(draws[name],[alpha/2,1-alpha/2])
        summaries.append({'policy':name,'mean':float(array.mean()),'ci95_low':float(lo),'ci95_high':float(hi),
                          'independent_seed_rows':array.shape[0],'scenarios':count,
                          'seed_mean_sd':float(array.mean(axis=1).std(ddof=1)) if array.shape[0]>1 else 0.})
        if name in names:
            delta=draws[primary]-draws[name]
            low,high=np.quantile(delta,[alpha/(2*family),1-alpha/(2*family)])
            raw_low,raw_high=np.quantile(delta,[alpha/2,1-alpha/2])
            differences.append({'comparison':primary+'-minus-'+name,'comparator':name,
                'mean_difference':float(arrays[primary].mean()-array.mean()),
                'ci95_low':float(raw_low),'ci95_high':float(raw_high),
                'familywise95_low':float(low),'familywise95_high':float(high),
                'bonferroni_comparators':family,'per_comparison_alpha':alpha/family,
                'primary_seed_rows':arrays[primary].shape[0],'comparator_seed_rows':array.shape[0],
                'paired_scenarios':count,'familywise_lower_positive':bool(low>0)})
    return summaries,differences
