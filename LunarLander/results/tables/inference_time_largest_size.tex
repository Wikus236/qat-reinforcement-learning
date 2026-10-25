% Requires: \usepackage{pdflscape}, \usepackage{booktabs}, \usepackage{adjustbox}, \usepackage{colortbl}
% Define in the preamble: \newcommand{\best}[1]{\cellcolor{green!25}\textbf{#1}}
% Define in the preamble: \newcommand{\secondbest}[1]{\cellcolor{blue!15}#1}
\begin{landscape}
\begin{table}[htbp]
\centering
\small
\caption{Deployment-style NumPy inference time comparison across model sizes. Green and blue indicate the best and second-best values within each model size, respectively.}
\label{tab:inference-time-comparison-largest}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llrrrrl}
\toprule
Size & Model & Mean time $\downarrow$ & SD across seeds $\downarrow$ & Mean timing SD $\downarrow$ & Reduction $\uparrow$ & Note \\
 &  & ($\mu$s) & ($\mu$s) & ($\mu$s) & (\%) &  \\
\midrule
256x256 & FP32 (Baseline) & \best{8.481060} & 0.031837 & 0.081578 & \best{0.00} & Custom FP32 \\
256x256 & W8 exported & 38.626130 & 0.105376 & 0.104554 & -355.44 & INT8 weights \\
256x256 & W8A8 exported & 43.981246 & 0.270989 & 0.209604 & -418.58 & INT8 weights, INT8 activations \\
256x256 & W4 simulated & \secondbest{38.524395} & 0.241568 & 0.083804 & \secondbest{-354.24} & Simulated W4, stored as INT8 \\
256x256 & W4A4 simulated & 44.478662 & 1.177463 & 1.895246 & -424.45 & Simulated W4A4, stored as INT8 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
\end{landscape}

\noindent
All displayed models use the same custom layer-by-layer NumPy inference structure. Quantized weights are dequantized for floating-point matrix operations because native low-precision kernels are not used. Simulated INT4 values remain stored in INT8 arrays.
