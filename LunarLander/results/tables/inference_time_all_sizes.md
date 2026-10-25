% Requires: \usepackage{pdflscape}, \usepackage{booktabs}, \usepackage{adjustbox}, \usepackage{colortbl}
% Define in the preamble: \newcommand{\best}[1]{\cellcolor{green!25}\textbf{#1}}
% Define in the preamble: \newcommand{\secondbest}[1]{\cellcolor{blue!15}#1}
\begin{landscape}
\begin{table}[htbp]
\centering
\small
\caption{Deployment-style NumPy inference time comparison across model sizes. Green and blue indicate the best and second-best values within each model size, respectively.}
\label{tab:inference-time-comparison-all}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llrrrrl}
\toprule
Size & Model & Mean time $\downarrow$ & SD across seeds $\downarrow$ & Mean timing SD $\downarrow$ & Reduction $\uparrow$ & Note \\
 &  & ($\mu$s) & ($\mu$s) & ($\mu$s) & (\%) &  \\
\midrule
16x16 & FP32 (Baseline) & \best{5.146062} & 0.012782 & 0.073337 & \best{0.00} & Custom FP32 \\
16x16 & W8 exported & \secondbest{7.014362} & 0.083296 & 0.080277 & \secondbest{-36.31} & INT8 weights \\
16x16 & W8A8 exported & 11.470190 & 0.147502 & 0.115758 & -122.89 & INT8 weights, INT8 activations \\
16x16 & W4 simulated & 7.213576 & 0.211671 & 0.409897 & -40.18 & Simulated W4, stored as INT8 \\
16x16 & W4A4 simulated & 11.524098 & 0.295468 & 0.261241 & -123.94 & Simulated W4A4, stored as INT8 \\
\midrule
32x32 & FP32 (Baseline) & \best{5.263652} & 0.028413 & 0.055663 & \best{0.00} & Custom FP32 \\
32x32 & W8 exported & 7.996485 & 0.977758 & 1.334092 & -51.92 & INT8 weights \\
32x32 & W8A8 exported & 12.047715 & 0.210418 & 0.442987 & -128.89 & INT8 weights, INT8 activations \\
32x32 & W4 simulated & \secondbest{7.496125} & 0.043936 & 0.079481 & \secondbest{-42.41} & Simulated W4, stored as INT8 \\
32x32 & W4A4 simulated & 11.888529 & 0.160986 & 0.143809 & -125.86 & Simulated W4A4, stored as INT8 \\
\midrule
64x64 & FP32 (Baseline) & \best{5.433384} & 0.139813 & 0.076996 & \best{0.00} & Custom FP32 \\
64x64 & W8 exported & 9.300865 & 0.534876 & 0.081991 & -71.18 & INT8 weights \\
64x64 & W8A8 exported & 14.258600 & 0.821211 & 0.329589 & -162.43 & INT8 weights, INT8 activations \\
64x64 & W4 simulated & \secondbest{9.152335} & 0.283995 & 0.082902 & \secondbest{-68.45} & Simulated W4, stored as INT8 \\
64x64 & W4A4 simulated & 14.435472 & 1.283723 & 0.518823 & -165.68 & Simulated W4A4, stored as INT8 \\
\midrule
128x128 & FP32 (Baseline) & \best{7.096793} & 0.227429 & 0.265162 & \best{0.00} & Custom FP32 \\
128x128 & W8 exported & 16.559478 & 0.769237 & 0.718895 & -133.34 & INT8 weights \\
128x128 & W8A8 exported & 21.856390 & 0.135632 & 0.181295 & -207.98 & INT8 weights, INT8 activations \\
128x128 & W4 simulated & \secondbest{16.523698} & 0.505132 & 0.439443 & \secondbest{-132.83} & Simulated W4, stored as INT8 \\
128x128 & W4A4 simulated & 22.494589 & 0.767611 & 0.481152 & -216.97 & Simulated W4A4, stored as INT8 \\
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
