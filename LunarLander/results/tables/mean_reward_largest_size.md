% Requires: \usepackage{pdflscape}, \usepackage{booktabs}, \usepackage{adjustbox}, \usepackage{colortbl}
% Define in the preamble: \newcommand{\best}[1]{\cellcolor{green!25}\textbf{#1}}
% Define in the preamble: \newcommand{\secondbest}[1]{\cellcolor{blue!15}#1}
\begin{landscape}
\begin{table}[htbp]
\centering
\small
\caption{Mean reward comparison across model sizes. Green and blue indicate the best and second-best values within each model size, respectively.}
\label{tab:mean-reward-comparison-largest}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llrrrrrrr}
\toprule
Size & Model & Mean reward $\uparrow$ & Reward SD $\downarrow$ & Best reward $\uparrow$ & Worst reward $\uparrow$ & Solve (\%) $\uparrow$ & Success (\%) $\uparrow$ & Failure (\%) $\downarrow$ \\
\midrule
256x256 & FP32 (Baseline) & 272.22 & 8.36 & \best{280.11} & 259.60 & 94.20 & 94.20 & \secondbest{2.00} \\
256x256 & W8 exported & \secondbest{272.55} & \best{4.08} & 277.93 & \best{267.60} & \secondbest{95.40} & \secondbest{94.80} & 2.20 \\
256x256 & W8A8 exported & \best{275.23} & \secondbest{4.47} & \secondbest{278.37} & \secondbest{266.38} & \best{96.80} & \best{96.80} & \best{1.40} \\
256x256 & W4 simulated & 160.95 & 84.36 & 243.31 & 2.11 & 58.00 & 70.40 & 12.80 \\
256x256 & W4A4 simulated & -164.29 & 212.63 & 123.66 & -504.86 & 11.60 & 17.00 & 50.40 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
\end{landscape}
