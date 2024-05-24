import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pymorphy2
import spacy
from nltk.probability import FreqDist

# Подгрузить текст из файла
with open('Family.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Токенизация с использованием регулярных выражений
words_regex = re.findall(r'\b\w+\b', text)

# Токенизация с использованием NLTK
nltk.download('punkt')
words_nltk = word_tokenize(text)

# Посчитать количество слов в тексте при помощи len
word_count_len = len(words_nltk)

# Посчитать количество слов в тексте при помощи Counter
word_count_counter = Counter(words_nltk)

# Очистить от пунктуации при помощи isalpha
words_alpha = [word for word in words_nltk if word.isalpha()]

# Очистить текст от стоп-слов при помощи библиотеки NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
words_no_stopwords = [word for word in words_alpha if word not in stop_words]

# Провести стемминг при помощи библиотеки NLTK
stemmer = SnowballStemmer(language='russian')
stemmed_words = [stemmer.stem(word) for word in words_no_stopwords]

# Очистка от пунктуации и приведение к нижнему регистру
words = [word.lower() for word in words_nltk if word.isalpha()]

# Очистка от стоп-слов
filtered_words = [word for word in words if word not in stop_words]

# Стемминг слов
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Подсчет частотности слов
word_freq = Counter(stemmed_words)

# Построение графика наиболее частотных слов
most_common_words = word_freq.most_common(10)
words, freq = zip(*most_common_words)

plt.figure(figsize=(10, 6))
plt.bar(words, freq)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words')
plt.show()

# Визуализация облака слов
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Анализ средней длины слова в тексте
word_lengths = [len(word) for word in words]
avg_word_length = sum(word_lengths) / len(word_lengths)
print(f'Средняя длина слова: {avg_word_length}')

# Анализ уникальных слов в тексте
unique_words = set(words)
num_unique_words = len(unique_words)
print(f'Количество уникальных слов: {num_unique_words}')

# Анализ количества предложений в тексте
sentences = re.split(r'[.!?]', text)
num_sentences = len(sentences)
print(f'Количество предложений: {num_sentences}')

print("Токенизация с использованием регулярных выражений:", words_regex)
print("Токенизация с использованием NLTK:", words_nltk)
print("Количество слов в тексте:", word_count_len)
print("Частота слов в тексте:", word_count_counter)
print("Текст без пунктуации:", words_alpha)
print("Текст без стоп-слов:", words_no_stopwords)
print("Стемминг:", stemmed_words)


#Уровень 2

# Загрузка модели SpaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

# Обработка текста с помощью SpaCy
doc = nlp(text)

# Лемматизация и вывод лемматизированного текста
lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])
print("Лемматизированный текст: ", lemmatized_text)

# Вывод результатов морфоанализа
for token in doc:
    print(f"Токен: {token.text}, Лемма: {token.lemma_}, Часть речи: {token.pos_}, Тэг: {token.tag_}")

# Удаление слов короче трех букв
filtered_tokens = [word.text for word in doc if len(word.text) >= 3]

# Вычисление частоты встречаемости слов
freq_dist = FreqDist(filtered_tokens)

# Построение графика частотных слов
freq_dist.plot(20, cumulative=False)