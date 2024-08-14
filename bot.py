import telebot
from telebot import types
import PyPDF2
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Токен бота
bot = telebot.TeleBot("7454075170:AAFYXXYYlOwhxDfUgkkx2k7ORvenYNotDJQ", parse_mode=None)

command = "start"

# Вставка ключа Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyASDLRRhBg-8ySqEzbrzk1uOWQg395Ckd4"

# Загрузка документа
text = ""
file = open("c:/tmp/python/danet_text.pdf", "rb")
pdf = PyPDF2.PdfReader(file)
for page in pdf.pages:
    text += page.extract_text()

# Разбиение документа на части
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
    length_function=len
)
texts = text_splitter.split_text(text)

# Создание векторного хранилища
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_texts(texts, embedding=embeddings)
vector_store.as_retriever()
vector_store.save_local("faiss_local")

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.4, max_length=10000)
system_prompt = """This file contains 3 stories for the game called 'Yes or No'. All the stories are numbered. Read the text after the word 'Riddle' to the user verbatim. 
Do not read the text after 'Answer'. Then answer to the user regarding the information after the word 'Answer' and before the next 'Riddle'. If user tells you he knows the answer, tell him the answer to that riddle. Then ask if wishes to guess the next riddle.
Answer only in Russian\n\nContext:\n {context}?\nQuestion: \n{question}\nAnswer:"""


#@bot.message_handler(commands=['document'])
#def get_document(message):
#    global command
#    command = 'start'
#    print(text)
#    bot.send_message(message.from_user.id, text)

@bot.message_handler(commands=['start'])
def get_start(message):
    global command
    command = 'start'
    text = ('Добро пожаловать! Давай сыграем в "ДА/НЕТ"!'
            '\nПолезные команды:\n/rules - правила игры\n/story - прочитать конец истории'
            '\n/hint - подсказка - бот даст небольшую подсказку\n/answer - ответ на загадку\n/help - вывести все команды бота'
            'Вот твоя первая загадка:\nМужчина заходит в бар и просит стакан воды, бармен внезапно достает ружье и направляет на мужчину. Мужчина говорит «спасибо» и уходит.')
    print(text)
    bot.send_message(message.from_user.id, text)

@bot.message_handler(commands=['rules'])
def get_rules(message):
    global command
    command = 'rules'
    response = ('Итак, правила игры "ДА/НЕТ":\n Ты прочитаешь конец истории. '
    'Он будет максимально непонятным, а ситуация, в которой окажутся герои покажется тебе сумасшедшей. '
    'Тебе, как настоящему детективу, нужно будет понять, что привело героев к тому исходу, который ты прочитал. '
    'Задавай боту вопросы, но помни - он может ответить только ДА или НЕТ. Когда ты решишь, что угадал историю, нажми /answer и бот покажет тебе всю историю, а ты поймёшь, насколько был прав! Удачи в игре!')
    print(response)
    bot.send_message(message.from_user.id, response)

@bot.message_handler(commands=['help'])
def get_help(message):
    global command
    command = 'help'
    response = ('Полезные команды:\n/story - прочитать конец истории/hint - подсказка -- бот даст несколько слов начала ответа (пока не работает)\n/answer - ответ на загадку\n/help - вывести все команды бота')
    print(response)
    bot.send_message(message.from_user.id, response)

@bot.message_handler(commands=['story'])
def get_story(message):
    global command
    command = 'story'
    response = ('Мужчина заходит в бар и просит стакан воды, бармен внезапно достает ружье и направляет на мужчину. Мужчина говорит «спасибо» и уходит.')
    print(response)
    bot.send_message(message.from_user.id, response)

@bot.message_handler(commands=['hint'])
def get_gemini_response_rag(message):
    global command
    command = 'hint'
    response = ('Вот первые несколько слов ответа на загадку:')
    print(response)
    bot.send_message(message.from_user.id, response)
    
@bot.message_handler(commands=['answer'])
def get_help(message):
    global command
    command = 'answer'
    response = ('Человек мучился от икоты и зашел в ближайший бар выпить воды. Бармен понял, в чем его' 
'проблема, и применил испытанное средство — напугать икающего человека. Способ сработал и мужчина поблагодарил его')
    print(response)
    bot.send_message(message.from_user.id, response)    

@bot.message_handler(content_types=['text'])
def get_gemini_response(message):

    bot.send_chat_action(message.chat.id, 'typing')
    
    db = FAISS.load_local("faiss_local", embeddings, allow_dangerous_deserialization=True)
    info = db.similarity_search(message.text)
        
    prompt = PromptTemplate(template=system_prompt, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": info, "question": message.text}, return_only_outputs=True)

    print(response)
    bot.send_message(message.from_user.id, response['output_text'])

bot.polling(none_stop=True, interval=0)
