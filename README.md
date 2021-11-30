# DDDM: a brain-inspired framework for robust classification

This is the source code of our dropout-bayes-based classifier framework.

## Workflow to plot accuracy heatmaps from scratch
- run `traintest.train_model()` to train a neural network classifier (NN).
- run `traintest.pipeline()` to attack the NN, save the NN predictions, estimate likelihood of NN predictions and perform cumsum & bayesian inference.
- run `analysis.plot_heatmap()` to plot the accuracy heatmaps.

## Workflow to plot ϵ-Acc/RT from scratch
- run `traintest.train_model()` to train a neural network classifier (NN).
- run `traintest.pipeline()` to attack the NN, save the NN predictions, estimate likelihood of NN predictions and perform cumsum & bayesian inference.
- run `analysis.plot_epsilon_accrt()` to plot the relationship between ϵ and Acc/RT.

## Notes to attack NLP model with `textattack.attack_recipes.textbugger_li_2018` in China
- install textattack 

    - in my case, I install version==0.3.0 from source codes to avoid some unnecessary dependence
    - further more, in `requirements.txt`, `scipy==1.4.1` is unnecessary and could raise error while during installation, you can replace it with `scipy>=1.4.1`
    - again, `numpy<1.19.0` is unnecessary, `numpy==1.21.4` works fine in my case

    
- run `nlp.attack_model_textattack()`
- In case of a download timeout of `universal-sentence-encoder_4`, here is a work-around:

    - open `/home/yourusername/.local/lib/python3.9/site-packages/textattack/constraints/semantics/sentence_encoders/universal_sentence_encoder/universal_sentence_encoder.py`
    - replace `tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"` 
    - with `tfhub_url = './data/universal-sentence-encoder_4/'`
    - download `universal-sentence-encoder_4.tar.gz` from [https://tfhub.dev/google/universal-sentence-encoder/4](https://tfhub.dev/google/universal-sentence-encoder/4)
    - extract the downloaded file in `./data/`, so there will be a `DDDM/data/universal-sentence-encoder_4/saved_model.pb` after the extraction
    - now try running `nlp.attack_model_textattack()`
    