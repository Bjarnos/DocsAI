---
title: Bot
parent: Classes
nav_order: 1
---

# Bot

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

The Bot class is a [class](/docs/Classes/index.md) created by the [BotService](/docs/Services/BotService) that contains other services.

## Bot.ConnectionService
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

A direct link to the [ConnectionService](/docs/Services/ConnectionService.md). For example code, check out the ConnectionService's documentation.

## Bot.MessageService
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

A direct link to the [MessageService](/docs/Services/MessageService.md). For example code, check out the MessageService's documentation.

## Bot.ProfileService
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

A direct link to the [ProfileService](/docs/Services/ProfileService.md). For example code, check out the ProfileService's documentation.

## Bot.login()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
Bot.login(username : str, password : str, params : dict) -> bool
```
This function allows you to log in to your account, so that you can begin with using your selfbot. Returns True or False, True means login succesful and False means login failed.
```py
# Example code
from ChatSelfbot import BotService
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    print("log in succesful")
else:
    print("log in failed")
```
Possible params, as of V1.0.0 (values set are the standard values):
```py
{
  'check-own': True,
  'force-first': False,
  'dm-id-v2': False
}
```

### check-own
When set to `False`, the bot will ignore all of it's own messages when checking public messages/replies.

### force-first
When you use [ConnectionService.start_checking_public()](/docs/Services/ConnectionService.html#connectionservicestart_checking_public) for the first time, the bot will run all bound functions for the last 10 minutes of messages. When `force-first` is `True`, the bot will read its own messages too (normally it won't). Use `check-own` to control the permanent reading of own messages.

### dm-id-v2
Uses a different way to send dm messages for if the normal way gets outdated, not recommended due to extra unwanted http requests.

{: .note }
You can choose which of the [services](/docs/Services/) you want to read now
