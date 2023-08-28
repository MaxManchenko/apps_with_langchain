#Intelligent YouTube Assistant

##Overview
The Intelligent YouTube Assistant is designed to answer questions about any specific YouTube video. Powered by the LangChain framework, this assistant leverages Language Models via OpenAI and HuggingFace's API to deliver accurate and context-specific answers. Its modular design allows easy adaptability to different types of user-specific data or content platforms beyond YouTube.

###Features
- **Video-Specific Query Handling**: Can answer questions related to the content, context, and more for specific YouTube videos.
- **Multi-model Support**: Seamlessly integrates with various Language Models available via OpenAI and HuggingFace's API.
- **Modular Structure**: Easily adaptable for different data sources and types of content.

##Core Components
###Initializer
The Initializer serves as the backbone of the assistant, handling all the static initialization tasks. It is designed to be data-source agnostic, allowing you to easily adapt it to other platforms or text sources. The Initializer takes care of:

- Setting up the environment
- Preloading common queries or data
- Other initialization tasks required before querying a Language Model

###YouTubeAgent
YouTubeAgent is the component responsible for understanding and answering queries about specific YouTube videos. It leverages the initialized environment from the Initializer to:

*Fetch video-specific data
*Run the language models
*Return the most accurate and context-specific answers  

##Installation
The project is Dockerized for easy setup and execution. Here's how to get it up and running:

1. **Clone the Repository**

``git clone https://github.com/YourRepo/IntelligentYouTubeAssistant.git``

2. **Navigate to the Project Folder**

``cd IntelligentYouTubeAssistant``

3. **Build the Docker Container**

``docker-compose up --build``

Your Intelligent YouTube Assistant should now be up and running on http://localhost:8000.


##How to Use

1. **Health Check**
Open your browser and go to http://localhost:8000 to check if the application is running. You should see a JSON response like {"health_check":"OK","model_version":"0.2.0"}.

2.**Post a Query**

You can use Postman or similar tools to post a query.

- **First Query (With Video URL)**:

``POST http://localhost:8000/get_answer``

JSON Payload:
```
{
  "video_url": "https://www.youtube.com/watch?v=example",
  "question": "What is the video about?"
}
```

This will return an answer along with an **"interaction_id""**.

- **Subsequent Queries (Same Video)**:

Use the **"interaction_id"** received in the previous response as a cookie to ask more questions about the same video.

Cookie: **interaction_id=<your_interaction_id_here>**

3. **Logs and Debugging**

Docker logs can provide debugging information if needed.

``docker-compose logs``