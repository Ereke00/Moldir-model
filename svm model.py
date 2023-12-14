from sklearn.svm import SVC
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def read_messages(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    messages = data.split('-------------------------\n')
    return messages

def preprocess_text(text):
    stop_words = set(stopwords.words('russian'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    processed_text = ' '.join(processed_tokens)
    return processed_text

file1_messages = read_messages('ИЗМЕНЕННЫЕ_СООБЩЕНИЯ1.txt')
file2_messages = read_messages('text.txt')

texts = []
categories = []

for message in file1_messages + file2_messages:
    lines = message.split('\n')
    for line in lines:
        if line.startswith('Категория'):
            category = line.split(': ')[1]
            categories.append(category)
            text = '\n'.join(lines[3:-5])
            preprocessed_text = preprocess_text(text)
            texts.append(preprocessed_text)

# Инициализация и обучение TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)  # Обучение и трансформация текстов

y = categories

# Обучение модели SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

# Функция классификации сообщения
def classify_message_with_svm(message):
    processed_message = preprocess_text(message)
    message_vectorized = vectorizer.transform([processed_message])
    prediction = svm_model.predict(message_vectorized)
    return prediction[0]
# Предсказания для тренировочных данных
y_pred = svm_model.predict(X)

# Оценка точности модели
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Построение матрицы ошибок
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Пример использования
sample_message = "Топ вакансии для вас. Успейте откликнуться на них"
svm_prediction = classify_message_with_svm(sample_message)
print("Predicted category using SVM:", svm_prediction)
