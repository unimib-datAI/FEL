# Fairness Encoding in Logic Tensor Networks (FEL)

FEL is a debiasing framework for feature-based binary classification tasks.

## **General Description**
- FEL integrates fairness and accuracy objectives within the Logic Tensor Networks (LTN) framework, where learning is guided by logical axioms rather than direct loss modifications.
- Through classification axioms, the model learns to correctly represent the relationship between features and target labels, ensuring predictive accuracy.
- Fairness axioms introduce constraints that promote statistical parity between sensitive groups, embedding fairness directly into the training process.
- Evaluation combines both perspectives:
    - Fairness is assessed using Statistical Parity Distance (SPD) and Disparate Impact (DI).
    - Performance is measured through overall classification accuracy.


## **Top Features**
- Declarative representation of constraints: no need to modify the loss function
- Compositionality: can be generalised to different fairness definitions
- Actively control the level of fairness
- No need to transpose the definition into a metric




## **Setup Instructions**

**1. Prerequisites**
1. Install Docker
2. Create a YAML config file (example: `./config.yaml`)
3. Prepare a CSV dataset with headers

**2. Build the Docker image**
```bash
$ docker build -t my_env .
```

**3. Run the container**
```bash
$ docker run --gpus all -it -v $(pwd):/home/developer/ -w /home/developer my_env bash
```

**4. Inside the container**

**Train a fair model**:
```bash
$ python src/training.py --config ./config.yaml --dataset ./datasets/compas.csv
```
**Run inference**:
```bash
$ python src/inference.py --config ./config.yaml --dataset ./datasets/compas.csv --model ./models/kb.npz
```


## **Configuration Guide**
The required YAML file need to specify the following parameters:
### Data Configuration
- **target_variable**: Column name containing binary outcomes
- **sensitive_feature**: Column name of the protected attribute
- **labels**:
  - **positive**: The outcome value to balance fairly across groups
  - **negative**: The other possible outcome value
- **protected_values**:
  - **privileged**: Value(s) marking the privileged group
  - **unprivileged**: Value(s) marking the unprivileged group

### Model Configuration
- **hidden_layer_sizes**: Neural network architecture [list of integers]
- **implies**: LTN fuzzy implication operator:
  - Options: KleeneDienes, Godel, Reichenbach, Goguen, Luk
- **p_mean**: Quantifier aggregation parameter:
  - Options: [1,3,5,7]
- **aggregator_deviation**: Deviation control in aggregator 
- **fairness_weight**: Weight of fairness axioms wrt classification axioms.
  - Options: [1...3]: A value of 1 gives fairness and classification axioms equal weight, while 3 assigns 3x times importance to fairness satisfiability.

### Training Configuration
- **epochs**: Number of training iterations

## Understanding "Positive" and "Negative" Labels

The choice of positive/negative labels is crucial:
- **Positive**: The outcome you want to balance fairly across groups
  - Example: loan_approved=1 means "approved"
  - Fairness metrics measure this outcome's distribution
- **Negative**: The alternative outcome
  - Not directly balanced by fairness constraints


Examples:

- Loan approval: positive = 1 (approved), negative = 0 (denied) → approval rates should be similar across groups.

- Disease detection: positive = “has_disease”, negative = “no_disease” → detection rates should be comparable across groups.

## **License**

FEL is open-source and released under the [Apache-2.0](./LICENSE) License.


## **Cite this Work**

Greco, G., Alberici, F., Palmonari, M., & Cosentini, A. (2023).
Declarative Encoding of Fairness in Logic Tensor Networks.
Frontiers in Artificial Intelligence and Applications, 372, 908–915.
IOS Press BV.
https://boa.unimib.it/bitstream/10281/462523/1/Greco-2023-ECAI-VoR.pdf