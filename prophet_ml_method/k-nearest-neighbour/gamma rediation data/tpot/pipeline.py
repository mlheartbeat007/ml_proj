import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.878794294606
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=0.01, dual=True, penalty="l2")),
    StackingEstimator(estimator=LogisticRegression(C=0.01, dual=False, penalty="l1")),
    RobustScaler(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=0.95, min_samples_leaf=13, min_samples_split=10, n_estimators=100, subsample=0.65)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
