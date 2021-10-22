# DDDM: a brain-inspired framework for robust classification

This is the source code of our dropout-bayes-based classifier framework.

## Workflow to plot accuracy heatmaps from scratch
1. run `utils.mkdir_dataset()` to set up data-storing directories.
2. run `traintest.train_model()` to train a neural network classifier (NN).
3. run `traintest.pipeline_multidropout()` to attack the NN, save the NN predictions, estimate likelihood of NN predictions and perform bayesian inference.
4. run `analysis.plot_heatmap()` to plot the accuracy heatmaps.

## Workflow to plot ϵ-Acc/RT from scratch
1. run `utils.mkdir_dataset()` to set up data-storing directories.
2. run `traintest.train_model()` to train a neural network classifier (NN).
3. run `traintest.pipeline_multiepsilon()` to attack the NN, save the NN predictions, estimate likelihood of NN predictions and perform bayesian inference.
4. run `analysis.plot_epsilon_accrt()` to plot the relationship between ϵ and Acc/RT.