
# Voice Assistant

## Overview

This project is a voice agent designed to interact with users through voice commands. It leverages advanced text-to-speech and natural language processing to provide a seamless conversational experience.

## Features


- **Natural Language Processing**: Understands and processes user words.
- **Text-to-Speech**: Converts text responses back into spoken language.
- **Customizable Responses**: Easily modify responses to suit different use cases.
- **Integration with APIs**: Connects with LLM APIs to  perform actions .
-  **RAG**: using of documents to provide answers .

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/hatemhellal/voice-assistant.git
    cd voice-agent
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Create a `.env` file in the root directory and add your API keys and other configuration settings and code the env exportation.
    ```env
    gemini=your_api_key
    ```
    or you can add it to the code in the gemini api key

## Usage

1. **Run the voice agent**:
    ```bash
    python main.py
    ```

2. **Interact with the agent**:
    Speak into your microphone and the agent will respond based on your commands.





