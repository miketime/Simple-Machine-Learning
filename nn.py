import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Incarca seturile de date
date_train = pd.read_csv('train.csv')
date_val = pd.read_csv('val.csv')
date_test = pd.read_csv('test.csv')

# Extrage caracteristicile si tintele
X_train_text = date_train['text']
y_train = date_train['score']

X_val_text = date_val['text']
y_val = date_val['score']

X_test_text = date_test['text']

tfidf_vectorizator = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizator.fit_transform(X_train_text).toarray()
X_val_tfidf = tfidf_vectorizator.transform(X_val_text).toarray()
X_test_tfidf = tfidf_vectorizator.transform(X_test_text).toarray()

scaler = StandardScaler()
y_train_standardizat = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_standardizat = scaler.transform(y_val.values.reshape(-1, 1)).flatten()

model = Sequential([
    Input(shape=(X_train_tfidf.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='linear')  
])

model.compile(optimizer=Adam(learning_rate=0.00005), loss='mean_squared_error', metrics=['mean_absolute_error'])

EarlyStopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
nn = model.fit(
    X_train_tfidf, y_train_standardizat,
    validation_data=(X_val_tfidf, y_val_standardizat),
    epochs=50,
    batch_size=16,
    callbacks=[EarlyStopping],
    verbose=1
)

predictii_val_standardizat = model.predict(X_val_tfidf)
predictii_val = scaler.inverse_transform(predictii_val_standardizat).flatten()

spearman, _ = spearmanr(y_val, predictii_val)
kendall, _ = kendalltau(y_val, predictii_val)
mse = mean_squared_error(y_val, predictii_val)
mae = mean_absolute_error(y_val, predictii_val)

print(f"Mean Absolute Error (MAE): {mae:}")
print(f"Mean Squared Error (MSE): {mse:}")
print(f"Corelatia Spearman pe setul de val: {spearman:}")
print(f"Corelatia Kendall pe setul de val: {kendall:}")

predictii_test_standardizat = model.predict(X_test_tfidf)
predictii_test = scaler.inverse_transform(predictii_test_standardizat).flatten()

rezultate = pd.DataFrame({'id': date_test['id'], 'score': predictii_test})
rezultate.to_csv('PredictiiFinaleV3.csv', index=False)
