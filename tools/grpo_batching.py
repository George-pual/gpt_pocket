"""Validate fresh multi-question G4 batches and mask only uncertain groups."""
from collections import Counter

def fresh_group_indices(row_ids,questions_per_generation):
    counts=Counter(row_ids)
    if len(counts)!=questions_per_generation or any(n!=4 for n in counts.values()):
        raise ValueError('Expected exactly four responses for every distinct question')
    groups={rid:[i for i,x in enumerate(row_ids) if x==rid] for rid in counts}
    for indices in groups.values():
        if indices!=list(range(indices[0],indices[0]+4)):
            raise ValueError('G4 rows must be contiguous before TRL shuffling')
    return groups

def mask_uncertain_groups(batch,row_ids,records,questions_per_generation):
    if len(row_ids)!=len(records):raise ValueError('Grading batch size mismatch')
    groups=fresh_group_indices(row_ids,questions_per_generation)
    dropped={}
    for rid,indices in groups.items():
        dropped[rid]=any(records[i]['group_exclusion'] for i in indices)
        if dropped[rid]:
            batch['completion_mask'][indices]=0
            batch['advantages'][indices]=0
    batch['num_items_in_batch']=batch['completion_mask'].sum()
    return groups,dropped
