# Negative log likelihood (NLL) and cross-entropy 

## 1. Negative Log Likelihood (NLL):
Suppose we have a probabilistic model with parameters θ, and we want to estimate the optimal values for θ given some observed data x. The likelihood function L(θ) represents the probability of observing the data x given the model parameters. The negative log likelihood is then calculated as:

NLL(θ) = -log L(θ)

In practice, we often work with the log-likelihood rather than the likelihood directly because it simplifies calculations and avoids numerical underflow issues.

## 2. Cross-Entropy:
In the context of classification tasks, suppose we have a predicted probability distribution y_pred over C classes and a true probability distribution y_true (often one-hot encoded) representing the ground truth. The cross-entropy loss between the predicted and true distributions is given by:

Cross-Entropy = - ∑[i=1 to C] y_true[i] * log y_pred[i]

In this formula, y_true[i] represents the true probability of class i, and y_pred[i] represents the predicted probability of class i according to the model. The summation is taken over all C classes. Note that the logarithm used is typically the natural logarithm (base e).

By minimizing the cross-entropy loss, we aim to make the predicted class probabilities (y_pred) as close as possible to the true class probabilities (y_true), thereby improving the model's classification performance.

It's important to note that while negative log likelihood (NLL) and cross-entropy have similar forms, their applications and interpretations differ. NLL is used for estimating model parameters and assessing the goodness of fit, while cross-entropy is employed as a loss function for classification tasks.

| Data  | Class | Class A Probability (y_pred) | Class B Probability (y_pred) | Class A True Probability (y_true) | Class B True Probability (y_true) | Cross-Entropy |
|-------|-------|-----------------------------|-----------------------------|----------------------------------|----------------------------------|---------------|
| Data 1 | A     | 0.9                         | 0.1                         | 1.0                              | 0.0                              | 0.105         |
| Data 2 | A     | 0.7                         | 0.3                         | 1.0                              | 0.0                              | 0.356         |
| Data 3 | A     | 0.8                         | 0.2                         | 1.0                              | 0.0                              | 0.223         |
| Data 4 | B     | 0.3                         | 0.7                         | 0.0                              | 1.0                              | 0.357         |
| Data 5 | B     | 0.2                         | 0.8                         | 0.0                              | 1.0                              | 0.223         |
| Data 6 | B     | 0.4                         | 0.6                         | 0.0                              | 1.0                              | 0.510         |

In the Cross-Entropy column, you can see the calculated cross-entropy loss for each datapoint using the formula mentioned earlier. Each value represents the dissimilarity between the predicted probabilities (y_pred) and the true probabilities (y_true) for that particular datapoint. The cross-entropy values provide an indication of how well the model's predicted probabilities align with the true probabilities.

Note that the cross-entropy values may vary depending on the specific predicted and true probabilities for each datapoint. In this example, we calculated the cross-entropy loss using the negative logarithm of the predicted probabilities for the true class.
