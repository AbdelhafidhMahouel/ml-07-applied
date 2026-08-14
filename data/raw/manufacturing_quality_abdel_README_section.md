## manufacturing_quality_abdel.csv (Phase 5 Custom Project Dataset)

A synthetic but realistically engineered injection molding process dataset,
created for the Phase 5 custom project by Abdelhafidh Mahouel. It simulates
3,000 production batches from a plastics injection molding line.

- 3,000 rows, 19 columns
- No missing values, no duplicate rows or batch IDs

### Columns

- `batch_id` - unique identifier for each production batch
- `machine_id` - one of 5 machines (M1-M5)
- `shift` - Day, Evening, or Night
- `operator_experience_years` - years of operator experience
- `melt_temperature_c` - plastic melt temperature (Celsius)
- `mold_temperature_c` - mold surface temperature (Celsius)
- `injection_pressure_bar` - injection pressure (bar)
- `injection_speed_mm_s` - injection speed (mm/s)
- `screw_speed_rpm` - screw rotation speed (RPM)
- `back_pressure_bar` - back pressure (bar)
- `hold_pressure_bar` - hold/packing pressure (bar)
- `hold_time_s` - hold/packing time (seconds)
- `cooling_time_s` - cooling time (seconds)
- `cycle_time_s` - total cycle time (seconds)
- `material_moisture_pct` - raw material moisture content (%)
- `ambient_humidity_pct` - ambient shop-floor humidity (%)
- `part_weight_g` - final part weight (grams); nominal target is 50.0g
- `defect` - binary target: 1 = batch predicted/observed defective, 0 = good
- `defect_type` - one of none, warpage, flash, short_shot, burn_mark (only set when defect=1)

### How this dataset was generated

The data was generated with `numpy`, using realistic process engineering
relationships rather than pure random noise. For example: defect risk rises
when melt temperature deviates from the 235C setpoint (in either direction),
when cooling time is too short relative to a 14 second minimum, when
material moisture is high, when injection speed is very fast (flash risk),
and is reduced by higher operator experience. Part weight responds to
injection pressure, hold pressure, melt temperature, and injection speed.
Random noise was added on top of these relationships so the data behaves
like real sensor data, not a perfectly deterministic formula.

### Example Use

- Binary classification: predict `defect` from the 13 raw process
  parameters (plus 2 engineered features) using RandomForestClassifier
  and GradientBoostingClassifier.
- Regression: predict `part_weight_g` from the same features using
  RandomForestRegressor.
- Multiclass extension (not required, a natural next step): predict
  `defect_type` for batches where `defect == 1`.

### License

Synthetic data created for coursework. No external license or attribution
required; free to reuse for educational purposes.
