from catboost import CatBoostClassifier
from obrabotka import exec_data, save_predicted_result
from time import time
from tqdm import tqdm

t = time()
print('Loading Data')
y, X, X_answer = exec_data()
print('Data successfully loaded. Time taken:', time() - t)
t = time()

model = CatBoostClassifier(
    random_seed=42,
    iterations=1300,
    bagging_temperature=1.6,
    boosting_type='Plain',
    bootstrap_type='Bayesian'
)

model.fit(X, y, verbose=False)

save_predicted_result(model.predict(X_answer))

print('Finished. Time taken:', time() - t)
