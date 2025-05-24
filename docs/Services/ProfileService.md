---
title: ProfileService
parent: Services
nav_order: 4
---

# ProfileService

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

{: .warning }
This page isn't done yet and may contain incorrect info!

The ProfileService is a [service](/docs/Services/index.md) used view profiles and modify your own.

{: .note }
It is recommended to check out the [Profile](/docs/Classes/Profile) class, as it is referenced in this article

## ProfileService.get_profile()
<p style="font-size: 0.9rem; color: #6c757d;">V1.1.0+</p>

```py
ProfileService.get_profile(username : str) -> Profile
```
This function returns info about a profile, returns `(True, Profile)` if the request succeeded and `(False, None)` if the request failed.
More about the [Profile](/docs/Classes/Profile) class.
```py
# Example code, empty
```
