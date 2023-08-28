# Intelligent YouTube Assistant

## Overview
The Intelligent YouTube Assistant is designed to answer questions about any specific YouTube video. Powered by the LangChain framework, this assistant leverages Language Models via OpenAI and HuggingFace's API to deliver accurate and context-specific answers. Its modular design allows easy adaptability to different types of user-specific data or content platforms beyond YouTube.

### Features
- **Video-Specific Query Handling:** Can answer questions related to the content, context, and more for specific YouTube videos.
- **Multi-model Support:** Seamlessly integrates with various Language Models available via OpenAI and HuggingFace's API.
- **Modular Structure:** Easily adaptable for different data sources and types of content.

## Core Components
### Initializer
The Initializer serves as the backbone of the assistant, handling all the static initialization tasks. It is designed to be data-source agnostic, allowing you to easily adapt it to other platforms or text sources. The Initializer takes care of:

- Setting up the environment
- Preloading common queries or data
- Other initialization tasks required before querying a Language Model

### YouTubeAgent
YouTubeAgent is the component responsible for understanding and answering queries about specific YouTube videos. It leverages the initialized environment from the Initializer to:

- Fetch video-specific data
- Run the language models
- Return the most accurate and context-specific answers  

## Installation
The project is Dockerized for easy setup and execution. Here's how to get it up and running:

1. **Clone the Repository**

``git clone https://github.com/YourRepo/IntelligentYouTubeAssistant.git``

2. **Navigate to the Project Folder**

``cd IntelligentYouTubeAssistant``

3. **Build the Docker Container**

``docker-compose up --build``

Your Intelligent YouTube Assistant should now be up and running on http://localhost:8000.


## How to Use

### Health Check

Open your browser and go to http://localhost:8000 to check if the application is running. You should see a JSON response like **{"health_check":"OK","model_version":"0.2.0"}**.

### Sending a Request with a New Video URL

1. **Open Postman**: Launch the Postman application on your computer.
2. **Set Request Type**: Select `POST` as the request type.
3. **Enter URL**: In the URL field, enter `http://localhost:8000/get_answer`.
4. **Add Request Body**: In the "Body" section, set the format to `JSON` and add the following JSON object:

    ```json
    {
      "video_url": "https://www.youtube.com/watch?v=example",
      "question": "Your question here"
    }
    ```

5. **Send Request**: Click on the "Send" button. You will receive an answer along with an `interaction_id`.

### Sending a Follow-up Request without a New Video URL

1. **Reuse `interaction_id`**: Use the same `interaction_id` from the previous response as a cookie to ask more questions about the same video.
2. **Set Request Type**: Select `POST` as the request type.
3. **Enter URL**: In the URL field, enter `http://localhost:8000/get_answer`.
4. **Add Request Body**: In the "Body" section, set the format to `JSON` and add the following JSON object:

    ```json
    {
      "question": "Your follow-up question here"
    }
    ```

5. **Add Cookie**: Add a cookie with the name `interaction_id` and the value from the previous response.

    Cookie: **interaction_id=<your_interaction_id_here>**

6. **Send Request**: Click on the "Send" button to get an answer to your follow-up question.

3. **Logs and Debugging**
Docker logs can provide debugging information if needed.

``docker-compose logs``