# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


data = {
    'text': ['Отличный продукт', 'Плохое качество', 'Нормально работает',
             'Великолепно!', 'Ужасно разочарован', 'Хорошо подходит'],
    'label': [1, 2, 0, 1, 2, 0]  # 0 - нейтральный, 1 - позитивный, 2 - негативный
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

joblib.dump(model, 'text_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Модель успешно обучена и сохранена!")