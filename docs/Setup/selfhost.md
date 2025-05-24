---
title: Selfhost
parent: Setup
nav_order: 2
---

# Selfhost

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

*We expect that you are using Visual Studio Code to selfhost, with all python extensions installed*

## Video tutorial
<iframe width="560" height="315" src="https://www.youtube.com/embed/4J0vybakKDE?si=SOuhcPw3QJypaKVE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Step 1. Install required libraries

### Virtual environment
1. Open an empty folder
2. Add a file named `bot.py` (or anything like that, as long as it's python), and set the source to:
```py
from ChatSelfbot import BotService
print("Hello world!")
```
4. Press `ctrl + shift + P`, and then select `Python: Create Environment...`
5. Select `Venv`, the download will start and might take a short while

### Terminal
6. Press `ctrl + shift + P` again and now select `Python: Create Terminal`
7. Run the command `.venv\Scripts\Activate` in the terminal, a green (.venv) tag will appear
8. Go ahead and run `pip install ChatSelfbot`, this will install the newest verion of our library
9. Now run `py bot.py` in the terminal (replace bot if you used a different name)
10. Did it print hello world without errors? Great! Now continue step 2  
    *(If there are errors, contact me!)*

---

## Step 2. Your first code!
*This example code is from `V1.1.0`, it may be outdated*  
Go to `bot.py`, and set the code to this (make sure to replace "username" with your username and "password" with your password!):
```py
from ChatSelfbot import BotService
bot = BotService.create_bot()
if bot.login("username", "password"):
    messages = bot.MessageService
    messages.create_post("Hello! I am a selfbot :D")
```
Run the command `py bot.py` again in your terminal and go to [Chat](https://chat.jonazwetsloot.nl/timeline?sort=time), did a new message by you appear? In that case you succesfully connected selfbot, read through the rest of the documentation to create the best selfbot of them all!

Remember: your ChatSelfbot won't automatically update! Keep track of announcements in Chat and download the newest update by running `pip install --upgrade ChatSelfbot` in your terminal!
