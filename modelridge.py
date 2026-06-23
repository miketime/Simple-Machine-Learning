import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, kendalltau


date_train = pd.read_csv('train.csv')
date_test = pd.read_csv('test.csv')
date_val = pd.read_csv('val.csv')

X_train= date_train['text']
y_train = date_train['score']

X_val = date_val['text']
y_val = date_val['score']

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(date_test['text'])

# Regresie ridge cu regularizare, pentru a preveni oferfit pe setul de date
ridge = Ridge(alpha = 40)
ridge.fit(X_train_tfidf, y_train)

# evaluare model pe setul de val
y_val_pred = ridge.predict(X_val_tfidf)

mae = mean_absolute_error(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)
corelatie_spearman, _ = spearmanr(y_val, y_val_pred)
corelatie_kendall, _ = kendalltau(y_val, y_val_pred)

print(f"MAE val: {mae}")
print(f"MSE val: {mse}")
print(f"Spearman: {corelatie_spearman}")
print(f"Kendall: {corelatie_kendall}")

test_predictii = ridge.predict(X_test_tfidf)
date_test['scor_predictii'] = test_predictii
date_test[['id', 'scor_predictii']].to_csv('IncarcareFinalv1.csv', index=False)

