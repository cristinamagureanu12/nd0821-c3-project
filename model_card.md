# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model uses KNN. It uses n=10 as a hyperparameter. The implementation is done using scikit-learn.

## Intended Use

This model classifies the a person salary intwo two categories (>50k and <=50k) based on the social background of a person.

## Training Data

Dataset source: https://archive.ics.uci.edu/ml/datasets/census+income. We use 80% of the dataset for training.

## Evaluation Data

Dataset source: https://archive.ics.uci.edu/ml/datasets/census+income. We use 20% for the evaluation.

## Metrics

We report around 0.78 accuracy.

## Ethical Considerations

Dataset contains data related race, gender and origin country. This will drive to a model that may potentially discriminate people;

## Caveats and Recommendations

Explore different learning algorithms.

