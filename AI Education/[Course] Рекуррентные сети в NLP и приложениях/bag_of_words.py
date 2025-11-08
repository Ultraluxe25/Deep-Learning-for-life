from sklearn.feature_extraction.text import CountVectorizer

text = ['Имеется корпус из нескольких текстов.',
        'Из них формируется частотный словарь', 
        'где ключи - слова из текста, присутствующие в корпусе,',
        'а значения - количество их вхождений в текст.']


bag_words = CountVectorizer(stop_words=["на", "и", "из"])
X = bag_words.fit_transform(text)
print(f'Слова текста: \n{bag_words.get_feature_names_out()}')
print(f'Частотный словарь мешка слов: \n{X.toarray()}')
