# Multilingual-RAG-chatbot
Fully open source multilingual RAG chatbot. Built on models from hugging face

In Client-support-chatbot , you'll find the code for running the bot on opensource LLM models along with the code for hosting a frontend with streamlit.

Add pdfs to data folder.
## Here is a detailed discription of the project.

# WHALE AI- A discovery chatbot

Idea and disclaimer:
Use case- Client query Q&A bot with Language translation and automated mailing capabilities.
All tools and models used in the project are open-source, meaning they can be used with in-house infra or cloud gpu and hence guarantee privacy and security.
The main features of the bot:
1. Advanced Document retrieval.
2. Language translation for over 120 languages.
3. Lead generation by storing the mail and name of client in CSV.
4. Automated mail forwarding through Gmail, users chat is analysed and relevant information is sent to them for future reference.
5. Cybersecurity measures to battle attacks like prompt injection has been taken.

Apart from adding these extremely sophisticated features, much research and effort has been made to choose the Best Models, tools and resources so that the chatbot can perform fast and effectively without error and hallucination.
The prototype app has been made where the Chatbot can be tested on a streamlit app, in the local environment.
However many alternative tools and methodologies have been submitted (Code provided) So That we can understand and choose what would work best for us.
Good things are good at somethings but the cost must not outweigh the value gained while using them. Keeping this is mind ..... 
A basic explanation of the working of the chatbot.

Model inference:
For this task we have used the latest "Intel/neural-chat-7b-v3-1" open source mistral based model. We have not used a fine-tuned model in the Entire project, as We believe the System or framework that the model runs on must be first brought to it's peak performance and should do most of the work and not the AI(LLM).

Fine-tuning is the last step of the process and is very delicate as the requirements and wants must be extremely clear so that the process turns fruitful.
Hence, Fine-tuning .. an expensive and vital process must only be done after proper discussion with the Client(MPEDA).
For model inference many a useful techniques have been used. So as to get the fastest possible response from the LLM.
With good infrastructure the following methods can reduce inference time to almost 3sec! 

Techniques used to reduce Inference speed:
1. Quantization-  4-bit and 8-bit quantization increase the speed of the model by a lot with only a slight decrease in quality.  Such loss is okay since it would hardly make any difference with powerful models are used in production.
Larger models can give faster results without sacrificing quality with proper quantization methods.
2. Optimization with Cuda- GPU performance is greatly increased with Cuda optimization.
3. Flash Attention- FlashAttention can be combined with other optimization techniques like quantization to further speedup inference.
4. Enabling BetterTransformer - Transformers provide a lot of tools to improve your inference.
Another method is using vllm, however it is still not a production ready resource.(more tests needed)
This is a RAG pipeline. 
We have focused on the documents and formatted them in a way the RAG can be more effective. For example the exporters list and website doc.
We have made a new doc altering that, adding a product column that mentions what products the exports sell and from where. This makes for a very useful and effective RAG pipeline.
 
We have used a RAG pipeline. The model used to create the embeddings is . (open source)
"sentence-transformers/all-MiniLM-L6-v2", it is highly effective and lightweight to use. Reliable in production. Vector store used is FAISS library. (open source), it is highly effective and lightweight to use. There is another methodology that can lead to better RAG for Tables and text. however it will require some extra power. 
Even so this is a viable option as we have many tables to retrieve data from. As it is a bit much for my device to handle. It is provided as a separate method in CODE. 

Language Translation:
There is a very simple yet effective code written by us to run these language translations on the user query.
Langdetect has been used to detect foreign language. And 2 models "Helsinki-NLP/opus-mt-mul-en" and "Helsinki-NLP/opus-mt-en-mul" has been used as a language translating model. (open-source).
These are the best open source model available for language translation translating over 120 languages.
The system is as follows.
If user enters non-english language, then it is detected and converted to English before passing into the LLM by "Helsinki-NLP/opus-mt-mul-en". Then the output that comes back in English is converted back to the original language by "Helsinki-NLP/opus-mt-en-mul”.
This works as the language translation system.


Lead generation:
There is a Feature added to the bot which can store email and name of the person asked in an input box to a CSV file that updates automatically. This way we can keep a track on the interested buyers.

Automatic mail forwarding:
This is a very interesting feature of the chatbot. The Idea is that the user can be mailed a processed and tailored version of their detailed queries and answers that they can reference and use for the future. 
A langchain agent is used for this job.
First it is fed the user chat data. Then it is asked to summarize it in the required way. Then using the Gmail Tool kit, we can directly make our agent work on the Email and draft it to the senders email. We have a choice to save the draft so that we can get some human supervision, or directly send it to the user.
This is a relatively new feature hence its integration into the app will need some time.


Cybersecurity measures:
The chatbot follows all compliance and regulations. It is armed for cyber attacks, the features added are.
Input text limiter limits the number of characters which can protect us from prompt injection attacks.
A strong prompt that maintains consistency in the output despite direct or indirect prompt hacking.
The application is end to end encrypted.
 

