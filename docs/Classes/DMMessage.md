---
title: DMMessage
parent: Classes
nav_order: 3
---

# DMMessage

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

The DMMessage class is a [class](/docs/Classes/index.md) that contains data and useful functions for a sent message in DMs.

## Variables
### time
<p style="font-size: 0.9rem; color: #6c757d;">V1.1.0+</p>

The epoch time that the dmmessage was sent, as float. 0 or None if an error occures.
```py
# Example code, prints the time of a new dm
from ChatSelfbot import BotService, Classes
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    messages = bot.MessageService
    success, message = messages.direct_message("ChatSelfbot", "Hello!")
    print(message.time)
```

### text
<p style="font-size: 0.9rem; color: #6c757d;">V1.1.0+</p>
<p style="font-size: 0.8rem; color: #6c757d;"><b>Aliases:</b> content</p>

The content text of a message, without markdown.
```py
# Example code, prints the text of a new dm
from ChatSelfbot import BotService, Classes
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    messages = bot.MessageService
    success, message = messages.direct_message("ChatSelfbot", "Hello!")
    print(message.text)
```

### markdowntext
<p style="font-size: 0.9rem; color: #6c757d;">V1.1.0+</p>

The original text of a message, including markdown.
```py
# Example code, prints the original text of a new dm
from ChatSelfbot import BotService, Classes
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    messages = bot.MessageService
    success, message = messages.direct_message("ChatSelfbot", "Hello!")
    print(message.markdowntext)
```

### sender
<p style="font-size: 0.9rem; color: #6c757d;">V1.1.0+</p>

The name of the sender of the message.
```py
# Example code, prints the sender name of a new dm
from ChatSelfbot import BotService, Classes
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    messages = bot.MessageService
    success, message = messages.direct_message("ChatSelfbot", "Hello!")
    print(message.sender)
```

### id
<p style="font-size: 0.9rem; color: #6c757d;">V1.1.0+</p>

The id of a message, this can not be used for anything except for identifying incoming messages.
```py
# Example code, prints the id of a new dm
from ChatSelfbot import BotService, Classes
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    messages = bot.MessageService
    success, message = messages.direct_message("ChatSelfbot", "Hello!")
    print(message.id) # prints "0" because the dm was just sent
```

### groupname
<p style="font-size: 0.9rem; color: #6c757d;">V1.1.0+</p>

The name of the group the message was sent in, if it was in one. "" or None if no groupname can be found.
```py
# Example code, prints the groupname of incoming dms
from ChatSelfbot import BotService
BotService.toggle_show_http()
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    connections = bot.ConnectionService
    def f1(message):
        print(message.groupname)

    connections.bind_to_any_dm(f1)
    connections.start_checking_dms()
```

## reply()
<p style="font-size: 0.9rem; color: #6c757d;">V1.1.0+</p>

```py
PublicMessage.reply(message : str) -> Tuple[bool, PublicMessage]
```
Reply to a dm, this works different than [MessageService.reply()](https://docs.bjarnos.dev/docs/Services/MessageService.html#messageservicereply), which is only meant for public posts!.
```py
# Example code, replies to incoming dms
from ChatSelfbot import BotService, Classes
DMMessage = Classes.DMMessage
bot = BotService.create_bot()
if bot.login("USERNAME HERE", "PASSWORD HERE"):
    connections = bot.ConnectionService
    def f1(message: DMMessage):
        message.reply("Thanks for your dm!")

    connections.bind_to_any_dm(f1)
    connections.start_checking_dms()
```
