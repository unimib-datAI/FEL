# Fairness Encoding in Logical Tensor Networks (FEL)

debiasing tool for feature-based binary classification task

### How it works:
- based on LTN
- axioms for classification: fit training data wrt target variable
- axioms for fairness: statistical parity


### Features:
- Declarative representation of constraints: no need to modify the loss function
- Compositionality: can be generalised to different fairness definitions
- Actively control the level of fairness
- No need to transpose the definition into a metric



### Parameters:

Datasets:
- Adult: gender, income
- German: gender, credit
- Compass: ethnicity, recidivism

Implication interpretation:
- Godel
- Lucasiewicx
- Reicchenbach

Axom weights: gives control over calssification task Vs. fairness task

Quantifiers interpretation:



### Evaluation metrics:
- Fairness: Statistical Parity Distance (SPD), Disparate Impact (DI)
- Performance: Accuracy



### Deployment:

$ docker build -t my_env .
$ docker run --gpus all -it -v $(pwd)/src:/home/developer/src -w /home/developer greta_env bash
inside the container
$ wandb login
$ python src/wan.py
