import warnings ; warnings.warn = lambda *args,**kwargs: None
import google.generativeai as genai
import pyaudio
import time
import wave
import queue
import struct
import threading
import subprocess
import pyttsx3
import time
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
# Initialize the TTS engine
engine = pyttsx3.init()
import pyaudio
import librosa
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings

import chromadb
DATA_PATH="./data"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""
def load_documents():
  """
  Load PDF documents from the specified directory using PyPDFDirectoryLoader.
  Returns:
  List of Document objects: Loaded PDF documents represented as Langchain
                                                          Document objects.
  """
  # Initialize PDF loader with specified directory
  document_loader = PyPDFDirectoryLoader(DATA_PATH) 
  # Load PDF documents and return them as a list of Document objects
  return document_loader.load() 
documents = load_documents()
#print(documents[0])

def split_text(documents: list[Document]):
  """
  Split the text content of the given list of Document objects into smaller chunks.
  Args:
    documents (list[Document]): List of Document objects containing text content to split.
  Returns:
    list[Document]: List of Document objects representing the split text chunks.
  """
  # Initialize text splitter with specified parameters
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # Size of each chunk in characters
    chunk_overlap=100, # Overlap between consecutive chunks
    length_function=len, # Function to compute the length of the text
    add_start_index=True, # Flag to add start index to each chunk
  )

  # Split documents into smaller chunks using text splitter
  chunks = text_splitter.split_documents(documents)
  print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

  # Print example of page content and metadata for a chunk
  document = chunks[0]
  #print(document.page_content)
  #print(document.metadata)

  return chunks # Return the list of split text chunks

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"
def save_to_chroma(chunks: list[Document]):
  """
  Save the given list of Document objects to a Chroma database.
  Args:
  chunks (list[Document]): List of Document objects representing text chunks to save.
  Returns:
  None
  """

  # Clear out the existing database directory if it exists
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
  
  # Create a new Chroma database from the documents
  # Assuming chunks is a list of text documents
  chunks = split_text(documents)

  db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=CHROMA_PATH
  )

  # Persist the database to disk
  db.persist()
  #print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
  """
  Function to generate vector database in chroma from documents.
  """
  documents = load_documents() # Load documents from a source
  chunks = split_text(documents) # Split documents into manageable chunks
  save_to_chroma(chunks)
generate_data_store()
LANG = "EN" # EN for English
DEBUG = True

# Model Configuration
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")
gemini=""
genai.configure(api_key=gemini)
generation_cofig={
    "temperature":0.7,
    "top_p":1,
    "top_k":1,
    "max_output_tokens":4096,

}
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel('gemini-pro', generation_config=generation_cofig, safety_settings=safety_settings)

def query_rag(query_text,model=model):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
  Args:
    - query_text (str): The text to query the RAG system with.
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
  # YOU MUST - Use same embedding function as before
  embedding_function = embeddings

  # Prepare the database
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  
  # Retrieving the context from the DB using similarity search
  results = db.similarity_search_with_relevance_scores(query_text, k=3)

  # Check if there are any matching results or if the relevance score is too low
  #if len(results) == 0 or results[0][1] < 0.6:
    #print(f"Unable to find matching results.")

  # Combine context from matching documents
  context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 
  # Create prompt template using context and query text
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  
  # Initialize OpenAI chat model
  model = model

  # Generate response text based on the prompt
  response_text = model.generate_content(prompt)
 
   # Get sources of the matching documents
  sources = [doc.metadata.get("source", None) for doc, _score in results]
 
  # Format and return response including generated text and sources
  formatted_response = f"Response: {response_text}\nSources: {sources}"
  return formatted_response, response_text


# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500
SILENT_CHUNKS = 2 * RATE / CHUNK  # two seconds of silence marks the end of user voice input
MIC_IDX = 0 # Set microphone id. Use tools/list_microphones.py to see a device list.

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)
        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    wf = wave.open('recordings/output.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


class VoiceOutputCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.generated_text = ""
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False

    def on_llm_new_token(self, token, **kwargs):
        # Append the token to the generated text
        with self.lock:
            self.generated_text += token

        # Check if the token is the end of a sentence
        if token in ['.', '。', '!', '！', '?', '？']:
            with self.lock:
                # Put the complete sentence in the queue
                self.speech_queue.put(self.generated_text)
                self.generated_text = ""

    def process_queue(self):
        while True:
            # Wait for the next sentence
            text = self.speech_queue.get()
            if text is None:
                self.tts_busy = False
                continue
            self.tts_busy = True
            self.text_to_speech(text)
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False

    def text_to_speech(self, text):
        try:
            subprocess.call(["say", "-r", "180", "-v", "Karen", text])
        except Exception as e:
            print(f"Error in text-to-speech: {e}")


if __name__ == '__main__':
    prompt_path = "prompts/en.txt" 
    with open(prompt_path, 'r', encoding='utf-8') as file:
        template = file.read().strip() # {dialogue}
    prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])

    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler()

    # Create a callback manager with the voice output handler
    callback_manager = BaseCallbackManager(handlers=[voice_output_handler])


    dialogue = ""
    try:
        while True:
            if voice_output_handler.tts_busy:  # Check if TTS is busy
                continue  # Skip to the next iteration if TTS is busy 
            try:
                print("Listening...")
                record_audio()
                print("Transcribing...")
                time_ckpt = time.time()
                audio_array, sampling_rate = librosa.load("recordings/output.wav", sr=16000)

# Process the audio
                input_features = whisper_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features

# Generate token ids
                predicted_ids = whisper_model.generate(input_features)

# Decode token ids to text
                transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
                
                print("%s: %s (Time %d ms)" % ("Guest", transcription[0], (time.time() - time_ckpt) * 1000))
            
            except subprocess.CalledProcessError:
                print("voice recognition failed, please try again")
                continue
            time_ckpt = time.time()
            print("Generating...")
            dialogue += "*Q* {}\n".format(transcription[0])

            prompt = prompt_template.format(dialogue=dialogue)
            
            # Let's call our function we have defined
            formatted_response, response_text = query_rag(prompt)
# and finally, inspect our final response!

            if response_text is not None:
                voice_output_handler.speech_queue.put(None)
                dialogue += "*A* {}\n".format(response_text.text)
                print("%s: %s (Time %d ms)" % ("Server", response_text.text, (time.time() - time_ckpt) * 1000))
                engine.say(response_text.text)
                engine.runAndWait()
    except KeyboardInterrupt:
        pass

