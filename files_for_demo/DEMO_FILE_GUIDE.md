# Demo File Guide

## General governance demo
- `coffee_quality_demo.csv`: small retail dataset with duplicates, missing counts, inconsistent neighbourhood labels, mixed date quality, and a sensitive-looking email field.
- `coffee_quality_context.txt`: governance note for sensitive fields, date formatting, standardization, and traceability.
- Suggested question: `Assess this dataset for quality and prepare it for analytics. Explain the quality issues found, governance risks, and what requires human review.`
- Suggested semantic config: `{"time_col":"last_update","primary_metric":"visit_count","dimensions":["neighbourhood"]}`

## Finance demo
- `finance_market_demo.csv`: two-ticker market sample with close, volume, benchmark, and signal fields.
- `finance_market_context.txt`: context for price, volume, benchmark, and backtest interpretation.
- Suggested question: `Assess this dataset for quality, then calculate returns, risk, drawdown, and comment on unusual volume.`
- Suggested semantic config: `{"time_col":"date","entity_col":"ticker","price_col":"close","volume_col":"volume","benchmark_col":"benchmark_close","signal_col":"signal"}`

## Healthcare demo
- `healthcare_admissions_demo.csv`: admission and discharge records with repeated patients, cohorts, duration, event, and treatment fields.
- `healthcare_context.txt`: context for readmission, cohort comparison, and outcome interpretation.
- Suggested question: `Assess this dataset for quality, compare cohorts, compute readmission risk, and highlight governance concerns.`
- Suggested semantic config: `{"patient_id_col":"patient_id","admission_date_col":"admission_date","discharge_date_col":"discharge_date","cohort_col":"cohort","outcome_col":"outcome_score","duration_col":"duration_days","event_col":"event","treatment_col":"treatment"}`

## Pipeline demo
- `pipeline_multi_defect_demo.xyz`: richer synthetic pipeline scan with multiple dents, broad ovality, and mild surface variation. This is the best presentation sample.
- `pipeline_multi_defect_context.md`: interpretation notes for the stronger pipeline visuals and engineering framing.
- `sample_pipe.xyz`: simpler single-dent cylindrical pipe segment if you want a minimal smoke test.
- `sample_pipeline_context.txt`: short dent and ovality context.
- `pipeline_inspection_context.md`: richer explanation of how to interpret the 3D, unwrapped, cross-section, axial-profile, and ovality visuals.
- Suggested question: `Find dents and deformation in this pipeline scan, compute the deviation map, measure ovality, assess fit quality, and explain which findings should be treated as engineering-review items.`
- Suggested semantic config: `{"units":"m","voxel_size":0.03}`
- Suggested analysis params: `{"deviation_threshold":0.05,"min_cluster_points":10,"slice_spacing":0.35}`
