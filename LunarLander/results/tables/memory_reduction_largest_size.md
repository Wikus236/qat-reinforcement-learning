% Requires: \usepackage{pdflscape}, \usepackage{booktabs}, \usepackage{adjustbox}, \usepackage{colortbl}
% Define in the preamble: \newcommand{\best}[1]{\cellcolor{green!25}\textbf{#1}}
% Define in the preamble: \newcommand{\secondbest}[1]{\cellcolor{blue!15}#1}
\begin{landscape}
\begin{table}[htbp]
\centering
\small
\caption{Memory comparison across model sizes. Actual stored parameter memory reflects the current tensor representation, while theoretical values assume efficient packed quantized deployment. Green and blue indicate the best and second-best reductions within each model size, respectively.}
\label{tab:memory-comparison-largest}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llrrrrrrrr}
\toprule
Size & Model & Actual params $\downarrow$ & Actual red. $\uparrow$ & Theor. params $\downarrow$ & Theor. red. $\uparrow$ & Theor. acts. $\downarrow$ & Act. red. $\uparrow$ & Theor. runtime $\downarrow$ & Runtime red. $\uparrow$ \\
 &  & (MB) & (\%) & (MB) & (\%) & (MB) & (\%) & (MB) & (\%) \\
\midrule
256x256 & FP32 (Baseline) & 0.263687 & 0.00 & 0.263687 & 0.00 & 0.001953 & 0.00 & 0.265640 & 0.00 \\
256x256 & W8 exported & 0.067410 & \best{74.44} & 0.067410 & \secondbest{74.44} & 0.001953 & 0.00 & 0.069363 & 73.89 \\
256x256 & W8A8 exported & 0.067417 & \secondbest{74.43} & 0.067417 & 74.43 & 0.000488 & 75.00 & 0.067905 & 74.44 \\
256x256 & W4 simulated & 0.067410 & \best{74.44} & 0.034695 & \best{86.84} & 0.001953 & 0.00 & 0.036648 & \secondbest{86.20} \\
256x256 & W4A4 simulated & 0.067417 & \secondbest{74.43} & 0.034702 & \best{86.84} & 0.000244 & 87.50 & 0.034946 & \best{86.84} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
\end{landscape}

\noindent
The simulated INT4 models restrict values to a 4-bit range, but the current implementation stores these values using INT8 tensors. Therefore, actual stored parameter memory reflects the current INT8-backed representation, while theoretical packed parameter and runtime memory assume an efficient packed INT4 deployment.
