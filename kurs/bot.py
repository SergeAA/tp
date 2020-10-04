import sys
import os
import re
import annoy
import string
import dialogflow
import numpy as np
import pandas as pd
from gensim.models import FastText
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
from telegram.ext  import Updater, CommandHandler, MessageHandler, Filters

# ================================================

from config import DIALOGFLOW_PROJECT_ID, DIALOGFLOW_LANGUAGE_CODE
from config import DIALOGFLOW_PROJECT_JSON, SESSION_ID, TELEGRAM_TOKEN

ROOT = '../data/'
DNU = ['Моя твоя не понимать', 'Попробуйте выразиться по другому', 'что?']
MORE = ['хотя кому я это говорю', 'эх, никто меня не слушает', 'хотя все это ерунда']
MSG_ERROR = 'Произошла ужасная ошибка, попробуйте позже'
EMBEDING_COUNT = 50
MINDESTINATION = 0.3
MAXDESTINATION = 0.7

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ROOT + DIALOGFLOW_PROJECT_JSON

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)

def preprocess_txt(line):
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    spls = list(filter(lambda x: len(x)>2, spls))
    return spls

def clean_str(line):
    spls = re.sub(re.compile('<.*?>'), '', line)
    spls = re.sub(r'\W+',' ', spls)
    spls = re.sub(r'\s+',' ', spls)
    spls = spls.split()
    spls = " ".join(i for i in spls if i not in sw and i != "").strip()
    return spls

def getVector(question, model):
    n_ft = 0
    vector_ft = np.zeros(EMBEDING_COUNT)
    for word in question:
        if word in model.wv:
            vector_ft += model.wv[word]
            n_ft += 1
    if n_ft > 0:
        vector_ft = vector_ft / n_ft
    return vector_ft

def getResponse(question, model, index, answers):
    question = preprocess_txt(question)
    vector = getVector(question, model)
    idx, dst = index.get_nns_by_vector(vector, 5, include_distances=True)
    res = []
    for i, j in enumerate(idx):
        a = answers[j].lower()
        a = re.sub(r'\W',' ', a)
        res.extend(a.split('.'))
    i = len(res)
    if not i: return '', 100
    while len(". ".join(res[:i])) > 100 and i > 1: i -= 1
    res = ". ".join(i.capitalize() for i in res[:i])
    if len(res)>100:
        res = res.split(' ')
        i = len(res)
        while len(" ".join(res[:i])) > 100 and i > 1: i -= 1
        res = " ".join(res[:i]) + ' ... ' + np.random.choice(MORE, 1)[0]
    return res, dst[0]

def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Добрый день')

def textMessage(bot, update):
    text, dst = getResponse(update.message.text, model, index, answers)
    print(f'{update.message.text}\n\t{dst:0.3f}: {text}')

    if dst > MINDESTINATION:
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
        text_input = dialogflow.types.TextInput(text=update.message.text , language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.types.QueryInput(text=text_input)

        try:
            response = session_client.detect_intent(session=session, query_input=query_input)
        except InvalidArgument:
            bot.send_message(chat_id=update.message.chat_id, text=MSG_ERROR)
            raise

        if response.query_result.action != 'input.unknown' or dst > MAXDESTINATION:
            text = response.query_result.fulfillment_text
            print('\tGoogle response:', text)

    if not text:
        text = np.random.choice(DNU, 1)[0].capitalize()

    bot.send_message(chat_id=update.message.chat_id, text=text)




if __name__ == "__main__":
    if not os.path.exists(f'{ROOT}df.pkl'):
        print('не могу загрузить ответы')
        sys.exit()

    if not os.path.exists(f'{ROOT}ft_model'):
        print('не могу загрузить модель')
        sys.exit()

    if not os.path.exists(f'{ROOT}speaker.ann'):
        print('не могу загрузить индекс')
        sys.exit()

    print('загружаем ответы ... ', end='')
    df = pd.read_pickle(f'{ROOT}df.pkl')
    answers = df.answer
    print('готово')

    print('загружаем модель ... ', end='')
    model = FastText.load(f'{ROOT}ft_model')
    print('готово')

    print('загружаем индекс ... ', end='')
    index = annoy.AnnoyIndex(EMBEDING_COUNT ,'angular')
    index.load(f'{ROOT}speaker.ann')
    print('готово')

    print('Стартуем бота ... ')
    updater = Updater(token=TELEGRAM_TOKEN)
    dispatcher = updater.dispatcher

    # Хендлеры
    start_command_handler = CommandHandler('start', startCommand)
    text_message_handler = MessageHandler(Filters.text, textMessage)

    # Добавляем хендлеры в диспетчер
    dispatcher.add_handler(start_command_handler)
    dispatcher.add_handler(text_message_handler)

    # Начинаем поиск обновлений
    updater.start_polling(clean=True)

    # Останавливаем бота, если были нажаты Ctrl + C
    updater.idle()