# TrustyAI-IG
Algorithm and tools to explain the relationship between input text to an LLM and its output according to integrated gradients.

## Integrated Gradients
IG-Enabler extends Integrated Gradients (IG), an Explainable AI technique introduced in the paper [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365), to Large Language Models (LLMs).  

## Supported ML Frameworks and Model Archeitectures
**HuggingFace**: All encoder-decoder sentiment classification models that implement the HF `AutoModelForSequenceClassification` library and masked-language models that implement the HF `AutoModelForMaskedLM` library.

## Quick Start
1. Clone this repo

    ```
    git clone https://github.com/trustyai-explainability/trustyai-ig.git
    ```

2. Install required packages and libraries

    ```
    pip install -r requirements.txt
    ```

3. Import required libraries

    ```
    # common
    import os
    import random
    import numpy as np

    # viz
    from IPython.display import display

    # local
    from ig_enabler import (
        ig,
        visualize
    )

    # third party 
    from transformers import (
        AutoModelForSequenceClassification, 
        AutoTokenizer
    )
    from datasets import load_dataset
    ```
3. Load model and tokenizer 

    ```
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    ```

4. Load and tokenize text

    ```
    text = "Review: Wow. What a wonderful film. The script is nearly perfect it appears this is the only film written by Minglun Wei,I hope he has more stories in him.<br /><br />The acting is sublime. Renying Zhou as Doggie was amazing -- very natural talent, and Xu Zhu was a delight - very believable as the jaded old traditionalist. <br /><br />The soundtrack was very effective, guiding without being overwhelming. <br /><br />If only more movies like this were made whether in Hollywood or Hong Kong- a family friendly, well acted, well written, well directed, near perfect gem."

    model_inputs = tokenizer(review, return_tensors="pt", truncation=True)
    ```
5. Calculate scores based on integrated gradients

    ```
    # enable model to calculate integrated gradients
    ig_enabled_model = ig.IGWrapper(model)

    # pass in input ids and attention mask to integrated gradients enabled model, as well as baseline embedding and number of steps
    scores = ig_enabled_model(
        model_inputs['input_ids'],
        model_inputs['attention_mask'],
        baseline=None, 
        num_steps=5
    )

    scores = scores.tolist()[0]
    ```

6. Visualize words in text that contribute the most to model prediction

    ```
    tokens = tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])

    # visualize words in the input text according to their scores
    display(visualize.visualize_token_scores(tokens, scores))

    # plot tokens associated with the top 10 scores
    visualize.plot_topk_scores(tokens, scores, 10)
    ```