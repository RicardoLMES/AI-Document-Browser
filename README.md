﻿# AI-Document-Browser
In this project I attempted to create an AI Chat where you can upload your own PDF and query that information

Although functional, there are a few bugs that still need resolving:
- Although not a bug, there is code and imports not being used, this is still to be fixed but I need to check exacly what I can and cannot delete.
- There is a weird loop happening where it will attempt to print the first query more than once, still investigating.
- When you first ask a question it gives you an answer alongside an error. Once you upload a document and then ask a question this does not happen, still investigating.

What are the next steps to improve this app?
- FAISS Stores the informartion localy, perhaps a way to know what PDFs have been ingested already as to not get repeats.
- Add a way to catch errors when uploading the same file twice.
- Webscraping, a way to automatically ingest PDFs from the web.
- Wikipedia Ingestor, as well as PDFs I believe it could be of use to be able to add information from Wikipedia directly for an added layer of context. (Use case for this particular project would be to maintain an updated list of UK officials)

Sasha's comments (to be deleted):
- Its great that you included a readme page and mentioned the issues you have and the future state, however you could have linked to the function where those issues are raised for example [function](/app.py::main) so developers know exactly where to go rather than finding it themselves, especially since you dont have a lot of documentation. Doc strings """ """ are great to describe the functions, when you create a comment to describe a function, include its purpose, the parameters and what data type it is, and same for what variables it returns if it does return any variables :)
- Another point is to screenshot your streamlit user interface, its also good for alex to see how it works even if your current demo has not run as you'd like it to. When you develop the program and it does work, screenshot immediately in case it won't when you add or change the code. You could put this into your readme page as well
