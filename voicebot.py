import os
import wave
import speech_recognition as sr
from tkinter import *
import nltk

# for downloading package files can be commented after First run
nltk.download('popular', quiet=True)
nltk.download('nps_chat',quiet=True)
nltk.download('punkt') 
nltk.download('wordnet')

posts = nltk.corpus.nps_chat.xml_posts()[:10000]

# To Recognise input type as QUES. 
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

conversation = ["hi i am ellie thanks for coming in today. i will ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential.",
"how are you doing today?",
"where are you from originally?",
"what are some things you really like about hometown?",
"what are some things you really like about livng in big city?",
"what are some things you do not really like about livng in big city?",
"what did you study at school?",
"what is your dream job?",
"do you travel a lot?",
"why you like or do not like traveling?",
"how often do you go back to your hometown?",
"do you consider yourself an introvert?",
"what do you do to relax?",
"how are you at controlling your temper?",
"when was the last time you argued with someone and what was it about?",
"how did you feel in that moment?",
"how close are you to them?",
"what are some things you like to do for fun?",
"who was someone that is been a positive influence in your life?",
"can you tell me about that?",
"how close are you to your family?",
"is there anything you regret?",
"what is one of your most memorable experiences?",
"how easy is it for you to get a good night  sleep?",
"what are you like when you do not sleep well?",
"have you been diagnosed with depression?",
"when was the last time you felt really happy?",
"okay i think i have asked everything i need to. Goodbye."]

def voicebot(txt):
    #Recording voice input using microphone 
    r = sr.Recognizer()

    # Taking voice input and processing 
    index = 0
    while True:
        txt.insert(END, "Ellie: " + conversation[index] + "\n")

        with sr.Microphone() as source:
            audio= r.listen(source, timeout=15)
        try:
            user_response = "{}".format(r.recognize_google(audio))
            txt.insert(END, "YOU SAID : " + user_response + "\n")

            if(classifier.classify(dialogue_act_features(user_response))=='Bye'):
                txt.insert(END, "Ellie: Bye! take care..\n")
                break
            
            with open("{:02d}.wav".format(index), "wb") as f:
                f.write(audio.get_wav_data())
            index += 1

            if index == len(conversation[:5]):
                break
        except sr.WaitTimeoutError:
            txt.insert(END, "Oops! Didn't catch that\n")
            pass
        except sr.UnknownValueError:
            txt.insert(END, "Oops! Didn't catch that\n")
            pass

    infiles = os.listdir()
    outfile = "sounds.wav"
    infiles = [file for file in infiles if file.endswith(".wav") and file != outfile]

    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()
        
    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()

    for infile in infiles:
        os.remove(infile)
