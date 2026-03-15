# Multi-defect pipeline demo context

This local demo pack simulates a more realistic pipeline inspection scene than the simple single-dent sample.

Interpretation guidance:
- The 3D dent-intensity view should look mostly pale, with a few coherent warm patches where inward deformation is strongest.
- The unwrapped dent map should behave like a flattened pipe wall: a mostly quiet background with hotspot islands or short streaks where dents exist.
- The axial dent profile should show clear peaks at the axial positions of the strongest defects.
- The cross-section should show a local inward notch or flattened arc instead of a perfectly circular section.
- The ovality profile captures broader out-of-round behaviour; it should be interpreted separately from localized dents.
- The dent risk matrix should separate small cosmetic-looking features from deeper, longer, and wider review items.

Engineering framing:
- Localized hotspots with matching axial peaks and cross-section notch evidence should be treated as strong dent candidates.
- Speckled heatmap noise without coherent hotspots should be treated as fit/noise review rather than a defect conclusion.
- If fit RMSE is elevated, treat defect outputs as engineering-review inputs rather than final integrity decisions.
