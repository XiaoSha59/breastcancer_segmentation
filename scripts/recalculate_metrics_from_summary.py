import json
import numpy as np
import os
import sys

def check(summary_path):
    with open(summary_path, 'r') as f:
        data = json.load(f)
    cases = data.get('patient_metrics', {})
    if not cases and 'cases' in data:
        cases = data['cases']
    
    # Handle official nnUNet summary.json format
    if not cases and 'metric_per_case' in data:
        cases = {}
        for item in data['metric_per_case']:
            ref = os.path.basename(item['reference_file']).replace('.nii.gz', '')
            m = item['metrics']['1']
            tp = m.get('TP', 0)
            fp = m.get('FP', 0)
            fn = m.get('FN', 0)
            rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            cases[ref] = {
                'Dice': m.get('Dice', 0.0),
                'IoU': m.get('IoU', 0.0),
                'Recall': rec,
                'Precision': prec,
            }
            
    print(f"File: {os.path.basename(summary_path)} | Total cases: {len(cases)}")
    
    dices, ious, recs, precs = [], [], [], []
    zero_cases = []
    for cid, m in cases.items():
        d = m['Dice']
        r = m['Recall']
        p = m['Precision']
        u = m['IoU']
        if r == 0.0 and p == 1.0:
            zero_cases.append(cid)
            p = 0.0  # corrected
        dices.append(d)
        ious.append(u)
        recs.append(r)
        precs.append(p)
        
    print(f"Cases with empty predictions (Rec=0, Prec=1 bug): {len(zero_cases)}")
    if zero_cases:
        print(f"Empty prediction cases: {zero_cases}")
    print(f"Original / Corrected Mean Dice:      {np.mean(dices):.4f}")
    print(f"Original / Corrected Median Dice:    {np.median(dices):.4f}")
    print(f"Original / Corrected Mean IoU:       {np.mean(ious):.4f}")
    print(f"Original / Corrected Mean Recall:    {np.mean(recs):.4f}")
    print(f"Corrected Mean Precision:            {np.mean(precs):.4f}")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'summary.json'
    check(path)
