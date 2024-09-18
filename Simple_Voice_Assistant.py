import pyttsx3
import pywhatkit
import speech_recognition as sr
import subprocess
import datetime
import pywhatkit
import wikipedia
import webbrowser
import smtplib
import pyjokes 
engine= pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
recognizer=sr.Recognizer()
def cmd():
    with sr.Microphone() as source:
        print('cleaning background noises..Please wait')
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print('Ask me anything...')
        audiorecorded = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audiorecorded)
        print(f"Vous avez dit : {text}")
        return text
    except sr.UnknownValueError:
        print("Je n'ai pas pu comprendre votre demande.")
        return ""
    except sr.RequestError:
        print("Erreur avec le service de reconnaissance vocale.")
        return ""
def sendEmail(to, content):
    server =smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
     
    
    server.login('your email id', 'your email password')
    server.sendmail('your email id', to, content)
    server.close()
def voice_assistant():
    while True:
        print("Écoute...")
        command = cmd()
        if "bonjour" in command.lower():
            a='Bonjour! Comment puis-je vous aider?'
            engine.say(a)
            engine.runAndWait()
            
        elif "chrome" in command.lower():
            d='open chrome'
            engine.say(d)
            engine.runAndWait()
            program="C:\Program Files\Google\Chrome\Application\chrome.exe"
            subprocess.Popen([program])
        elif "wikipedia" in command.lower():
            search_term = command.lower().replace('wikipedia', '').strip()
            results = wikipedia.summary(search_term, sentences = 3)
            print(results)
            engine.say(results)
            engine.runAndWait()
        elif "jumia" in command.lower():
            engine.say('we are going to jumia')
            engine.runAndWait()
            webbrowser.open("jumia.com")
        elif "time" in command.lower():
            time= datetime.datetime.now().strftime('%I:%M %p')
            print(time)
            engine.say(time)
            engine.runAndWait()
        elif "play" in command.lower():
            search_term = command.lower().replace('play', '').strip()
            engine.say(f"Playing {search_term} on YouTube")
            engine.runAndWait()
            pywhatkit.playonyt(search_term)
        elif "email " in command.lower():
          
            engine.say('what do you want to send ')
            content = cmd()
            engine.say("whome should i send")
            person=input()
            sendEmail(person,content)
            print("i send your email succesfuly")

        elif "joke" in command.lower():
            engine.say(pyjokes.get_joke())
            engine.runAndWait()
            
        elif "au revoir" in command.lower():
            b='Au revoir!'
            engine.say(b)
            engine.runAndWait()
            break
        else:
            c=  'Je ne comprends pas cette commande'
            engine.say(c)
            engine.runAndWait()
            break
voice_assistant()

