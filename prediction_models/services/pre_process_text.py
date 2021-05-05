import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as stpwrds

bad_chars = "أإآئءة"
punct = "\n\r!\''\".“”،:؟"+string.punctuation

def rejoin_text(text):
    st=""
    st=st.join(text)
    return st

def remove_punctuations(text):
    for punctuation in punct:
        text = text.replace(punctuation, '')
    return text

def remove_stop_words(text):
    stopwords = stpwrds.words('arabic')
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word+' ' for word in text_tokens if not word in stopwords]
    return tokens_without_sw

def replace_special_chars(text):
    for i in bad_chars :
        if i=='ة':
            text = text.replace(i, 'ه')
        else:
            text = text.replace(i, 'ا')
    return text

def pre_process_txt(text):
    text = replace_special_chars(text)
    text = remove_punctuations(text)
    text = remove_stop_words(text)
    text = rejoin_text(text)
    return text
