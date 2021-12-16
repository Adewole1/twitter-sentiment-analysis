from os.path import join, dirname
from dotenv import load_dotenv

import get_username
from get_username import UserNameText

import classifier

users = []
while True:
    username = input("Enter a twitter username: (or d when done)")
    if username.capitalize() == 'D':
        break
    users.append(username)

userText = UserNameText()
data = userText.main(users)

classy = {
            'neu': 'Neutral',
            'not': "Not depressed",
            'dep': 'Depressed'
        }

for name in data.keys():
    print("\nFor {}:".format(name.capitalize()))
    for text in data[name]:
        clas, conf = classifier.sentiment(text)
        print("\n{}".format(text))
        print("This tweet is {}".format(classy[clas]))
        print("The confidence level is {}%".format(conf))
        
    



'''
for t in text:
    clas, conf = c.sentiment(t)
    print("\n", t)
    print("This tweet is", classy[clas])
    print("The confidence level is ", conf, "%")
'''