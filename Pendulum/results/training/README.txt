Generated from saved training-evaluation CSVs, without retraining.
Figure: evaluation mean reward with +/- one episode-reward SD, capture rate, and balanced time.
No smoothing or interpolation; lines connect logged evaluations.
These are evaluations of one training run per phase, not averages across training seeds.
Base history includes recovery/restoration and exploration-boost interventions recorded in the source CSV.
Base logging starts partway through training. Mild has a separate phase-local step counter.
Highest mean reward is descriptive and is not the deployment selection rule, which prioritizes sustained balance.
The table lists the first, highest-mean-reward and final logged evaluations for each phase.
Capture means at least one second of continuous balance. Mean longest balance averages each episode's longest streak.
CSV rates remain fractions; plots and LaTeX convert them to percentages.
Include the PDF figure with includegraphics and the .tex table with input in your report.
