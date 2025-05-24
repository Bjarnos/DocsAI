---
title: Web Service
parent: Setup
nav_order: 1
---

# Web Service

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Video tutorial
<iframe width="560" height="315" src="https://www.youtube.com/embed/MN_pF-qhBak?si=B_V15PTZE7KI2EaM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Step 1. Create a web service

### Recommended programs
- [Koyeb](https://koyeb.com)
- [Render](https://render.com)

### Webserver variables
When deploying the webserver:
1. Make sure `Language` is `Python 3`
2. The variable `Build Command` must be `pip install -r requirements.txt`  
  *(This means that all libraries in `requirements.txt` will be installed)*
4. If there is a variable `Start Command`, set it to `gunicorn app:app & python3 -u bot.py & wait`, else you'll have to set up a [Procfile](#Content%20files)  
  *(This runs the python files `app.py` and `bot.py`)*
5. You'll have to set up 2 environment variables. Set `user` to your Chat username, and `pass` to your Chat password  
  *(Store your login info to keep it safe, as your code might be visible to others)*

*Need help with any of these steps? Contact me!*

### Content files
If you did everything correct, you should have a webserver that runs python now.
These files should be in your `main` branch:
#### `requirements.txt`
Containins all python libraries required by your bot. Gunicorn, Flask, python-dotenv and ChatSelfbot are required for the selfbot to work. Replace customlibrary1 and customlibrary2 with actual libraries you need. Format:
```text
gunicorn=23.0.0
Flask==3.0.3
python-dotenv=1.0.1
ChatSelfbot

customlibrary1
customlibrary2
```
#### `app.py`
This code will keep your service active 24/7. **Exact** content:
```py
from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def ping():
    return jsonify({"success": "true"}), 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```
#### `bot.py`
This is the customisable code of your bot! Content will be specified in [step 3](#step-3-your-first-code)
#### `Procfile`
This file contains your build command, add it even with build command already set, and use this content:
```sh
web: gunicorn app:app & python3 -u bot.py & wait
```

---

## Step 2. Setting up a Cronjob
1. Go to [https://cron-job.org/en/](https://cron-job.org/en/) and sign up/log in
2. Navigate to the [Dashboard](https://console.cron-job.org/dashboard) (should happen automatically)
3. On the right top of your screen there's a button `CREATE CRONJOB`, click it
4. Enter a title and set the `URL` to the url of your Web Service  
   *(can't find the url of your Web Service? Contact me!)*
5. Make sure the job is enabled and set the time to `Every 10 minutes` (don't use the standard 15!). If you use a different Cronjob than cron-job.org and it requires an expression, set it to `*/10 * * * *`
6. Click `CREATE`, and you're done! Keep track of the cronjob from time to time because it might stop after failing too many times.

---

## Step 3. Your first code!
*This example code is from `V1.1.0`, it may be outdated*  
Go to `bot.py`, and set the code to this:
```py
import dotenv, os
from ChatSelfbot import BotService
bot = BotService.create_bot()

dotenv.load_dotenv()
username = os.environ.get('user')
password = os.environ.get('pass')
if bot.login(username, password):
    messages = bot.MessageService
    messages.create_post("Hello! I am a selfbot :D")
```
Now redeploy your service and go to [Chat](https://chat.jonazwetsloot.nl/timeline?sort=time), did a new message by you appear? In that case you succesfully connected selfbot, read through the rest of the documentation to create the best selfbot of them all!
