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
16x16 & FP32 (Baseline) & \best{5.265574} & 0.154837 & 0.184765 & \best{0.00} & Custom FP32 \\
16x16 & W8 exported & 7.288539 & 0.363042 & 0.365155 & -38.42 & INT8 weights \\
16x16 & W8A8 exported & 11.733529 & 0.103223 & 0.125649 & -122.83 & INT8 weights, INT8 activations \\
16x16 & W4 simulated & \secondbest{7.120090} & 0.168150 & 0.128364 & \secondbest{-35.22} & Simulated W4, stored as INT8 \\
16x16 & W4A4 simulated & 11.773844 & 0.202582 & 0.086654 & -123.60 & Simulated W4A4, stored as INT8 \\
\midrule
32x32 & FP32 (Baseline) & \best{5.266173} & 0.032221 & 0.053805 & \best{0.00} & Custom FP32 \\
32x32 & W8 exported & 7.933231 & 0.366455 & 0.467169 & -50.65 & INT8 weights \\
32x32 & W8A8 exported & 12.195417 & 0.059917 & 0.090518 & -131.58 & INT8 weights, INT8 activations \\
32x32 & W4 simulated & \secondbest{7.642602} & 0.234361 & 0.328855 & \secondbest{-45.13} & Simulated W4, stored as INT8 \\
32x32 & W4A4 simulated & 12.297827 & 0.148067 & 0.096177 & -133.52 & Simulated W4A4, stored as INT8 \\
\midrule
64x64 & FP32 (Baseline) & \best{5.434947} & 0.064176 & 0.057511 & \best{0.00} & Custom FP32 \\
64x64 & W8 exported & 9.368801 & 0.247711 & 0.536765 & -72.38 & INT8 weights \\
64x64 & W8A8 exported & 13.951352 & 0.155624 & 0.102796 & -156.70 & INT8 weights, INT8 activations \\
64x64 & W4 simulated & \secondbest{9.155039} & 0.289317 & 0.310954 & \secondbest{-68.45} & Simulated W4, stored as INT8 \\
64x64 & W4A4 simulated & 13.803288 & 0.071927 & 0.113866 & -153.97 & Simulated W4A4, stored as INT8 \\
\midrule
128x128 & FP32 (Baseline) & \best{7.048160} & 0.073127 & 0.046204 & \best{0.00} & Custom FP32 \\
128x128 & W8 exported & 16.008109 & 0.150453 & 0.384266 & -127.12 & INT8 weights \\
128x128 & W8A8 exported & 22.047728 & 0.078107 & 0.137974 & -212.82 & INT8 weights, INT8 activations \\
128x128 & W4 simulated & \secondbest{15.821126} & 0.136666 & 0.146854 & \secondbest{-124.47} & Simulated W4, stored as INT8 \\
128x128 & W4A4 simulated & 22.085493 & 0.212822 & 0.163029 & -213.35 & Simulated W4A4, stored as INT8 \\
\midrule
256x256 & FP32 (Baseline) & \best{8.645236} & 0.055926 & 0.103969 & \best{0.00} & Custom FP32 \\
256x256 & W8 exported & 38.625354 & 0.141650 & 0.228209 & -346.78 & INT8 weights \\
256x256 & W8A8 exported & 44.200771 & 0.137596 & 0.385820 & -411.27 & INT8 weights, INT8 activations \\
256x256 & W4 simulated & \secondbest{38.564348} & 0.202968 & 0.208180 & \secondbest{-346.08} & Simulated W4, stored as INT8 \\
256x256 & W4A4 simulated & 44.313286 & 0.177997 & 0.233910 & -412.57 & Simulated W4A4, stored as INT8 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
\end{landscape}

\noindent
All displayed models use the same custom layer-by-layer NumPy inference structure. Quantized weights are dequantized for floating-point matrix operations because native low-precision kernels are not used. Simulated INT4 values remain stored in INT8 arrays.
