# Random initialization distributions

30 seeds per architecture, fresh weights from the trained runs’ configurations. Each row averages next-token distributions for one fixed context. No pretrained weights loaded.

Uniform probability is 1/50,257 = 0.0000198977. Ratio columns multiply probability by vocabulary size; uniform is 1. KL and total variation are zero for uniform predictions.

| Architecture | Context | Mean KL to uniform (nats) | Mean total variation | Mean maximum probability | Last-input token ratio | Left ratio | Right ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3 | bos_only | 0.204819 | 0.251084 | 0.000246319 | 1.15338 | 0.872906 | 1.31987 |
| qwen3 | single_ball | 0.204845 | 0.251123 | 0.000243734 | 1.27358 | 0.834243 | 1.4585 |
| qwen3 | single_left | 0.205487 | 0.251478 | 0.000261645 | 1.04867 | 1.04867 | 1.21879 |
| qwen3 | single_right | 0.205021 | 0.251212 | 0.000235857 | 1.65406 | 0.950084 | 1.65406 |
| qwen3 | neutral | 0.20507 | 0.251163 | 0.000251481 | 1.08322 | 1.02132 | 1.19937 |
| qwen3 | left_scene | 0.204947 | 0.251219 | 0.00025428 | 1.2598 | 1.03758 | 0.935642 |
| qwen3 | right_scene | 0.204976 | 0.251099 | 0.000249017 | 1.32872 | 1.00176 | 1.11168 |
| qwen3 | summary_bridge | 0.204841 | 0.251061 | 0.000267055 | 1.26511 | 1.07145 | 1.00255 |
| llama | bos_only | 0.204554 | 0.250866 | 0.000240119 | 1.056 | 1.18682 | 1.21743 |
| llama | single_ball | 0.204694 | 0.25098 | 0.000238066 | 1.21257 | 0.99954 | 1.08294 |
| llama | single_left | 0.205101 | 0.251213 | 0.000246355 | 1.29965 | 1.29965 | 1.11572 |
| llama | single_right | 0.204472 | 0.250851 | 0.00024156 | 1.22762 | 1.18886 | 1.22762 |
| llama | neutral | 0.204791 | 0.250967 | 0.000247623 | 1.08294 | 1.14359 | 0.943973 |
| llama | left_scene | 0.205506 | 0.251303 | 0.000280398 | 1.20349 | 1.12223 | 1.08708 |
| llama | right_scene | 0.20534 | 0.251234 | 0.000262872 | 1.19756 | 1.09457 | 1.02897 |
| llama | summary_bridge | 0.205462 | 0.251257 | 0.000272095 | 1.17509 | 1.09844 | 1.0578 |

summary.json includes across-seed standard deviations and approximate 95% t intervals for means (30 seeds). These can be unstable for skewed probabilities and are not simultaneous confidence intervals. Finite-seed averaged distributions retain sampling noise; deviation from uniform in that average alone does not establish a population asymmetry.

This estimates the initialization procedure in the installed library with matched architectural settings, not the exact original training-start RNG state. Token ratios refer to leading-space single tokens. This is a next-token diagnostic, not full-answer probe accuracy.
