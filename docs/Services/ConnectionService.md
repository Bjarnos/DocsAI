---
title: ConnectionService
parent: Services
nav_order: 2
---

# ConnectionService

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

The ConnectionService is a [service](/docs/Services/index.md) used to run specific functions when a message is received.
The service is split in 2 parts, Public Service (for handling public posts) and DM Service (for handling DMs/group chats).

{: .note }
It is recommended to check out the [PublicMessage](/docs/Classes/PublicMessage) and [DMMessage](/docs/Classes/DMMessage) classes, as they are referenced in this article

## Public Service
### ConnectionService.bind_to_public_post()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
ConnectionService.bind_to_public_post(func : function) -> None
```
This function will run argument `func` every time that a new public post is created, and pass a [PublicMessage](/docs/Classes/PublicMessage) as the first argument.
```py
# Example code, likes every new message received
from ChatSelfbot import BotService, Classes
PublicMessage = Classes.PublicMessage
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    connections = bot.ConnectionService
    def f1(message: PublicMessage):
        message.like()
    connections.bind_to_public_post(f1)
    connections.start_checking_public() # required!
```

### ConnectionService.bind_to_message_reply()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
ConnectionService.bind_to_message_reply(message_id : str, func : function) -> None
```
This function will run argument `func` every time that there's replied to the message with id message_id, and pass a [PublicMessage](/docs/Classes/PublicMessage) (the reply) as the first argument. This function is recursive, will be fixed soon!
```py
# Example code, likes every new reply
from ChatSelfbot import BotService, Classes
PublicMessage = Classes.PublicMessage
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    connections = bot.ConnectionService
    def f1(message: PublicMessage):
        message.like()
    def f2(message: PublicMessage):
        connections.bind_to_message_reply(message.id, f1)
    connections.bind_to_public_post(f2)
    connections.start_checking_public() # required!
```
*Note that this example code can be done more effective with [PublicMessage.bind_to_reply()](https://docs.bjarnos.dev/docs/Classes/PublicMessage.html#bind_to_reply)*

### ConnectionService.start_checking_public()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
ConnectionService.start_checking_public() -> None
```
You are required to run this function if you want to use bind_to_public_post() or bind_to_message_reply(), else the bot simply won't check new messages. This function won't be automated within bot startup because it's required to keep the main thread running.

### ConnectionService.stop_checking_public()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
ConnectionService.stop_checking_public() -> None
```
Stops checking public. New received public messages won't be handled anymore.

---

## DM Service
### ConnectionService.bind_to_any_dm()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
ConnectionService.bind_to_any_dm(func : function) -> None
```
This function will run argument `func` every time that a new DM is received (including group chats!), and pass a [DMMessage](/docs/Classes/DMMessage) as the first argument.
```py
# Example code, responds with "Okay!" to every DM
from ChatSelfbot import BotService, Classes
DMMessage = Classes.DMMessage
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    connections = bot.ConnectionService
    def f1(message: DMMessage):
        message.reply("Okay!")
    connections.bind_to_any_dm(f1)
    connections.start_checking_dms() # required!
```
*(In group messages, message.reply() will send a reply to the group, not the sender of the message)*

### ConnectionService.bind_to_user_dm()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
ConnectionService.bind_to_user_dm(username : str, func : function) -> None
```
*This function DOES NOT work for group DMs!*
This function will run argument `func` every time that a new DM is received from username, and pass a [DMMessage](/docs/Classes/DMMessage) as the first argument.
```py
# Example code, responds with "Okay!" to every DM from user 'Bjarnos'
from ChatSelfbot import BotService, Classes
DMMessage = Classes.DMMessage
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    connections = bot.ConnectionService
    def f1(message: DMMessage):
        message.reply("Okay!")
    connections.bind_to_user_dm("Bjarnos", f1)
    connections.start_checking_dms() # required!
```

### ConnectionService.start_checking_dms()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
ConnectionService.start_checking_dms() -> None
```
You are required to run this function if you want to use bind_to_any_dm() or bind_to_user_dm(), else the bot simply won't check new dm messages. This function won't be automated within bot startup because it's required to keep the main thread running.

### ConnectionService.stop_checking_dms()
<p style="font-size: 0.9rem; color: #6c757d;">V1.0.0+</p>

```py
ConnectionService.stop_checking_dms() -> None
```
Stops checking dms. New received DM messages won't be handled anymore.
