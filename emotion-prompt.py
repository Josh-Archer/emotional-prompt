try:
    import requests
except ImportError:
    print("Error:  module not found. Install it with 'pip install requests'.")
    exit(1)

import re
import json
import time

import utils

# Define the model to use
MODEL = "llama3.2"

# Dictionary mapping emotions to lists of topics
emotion_map = {
    "happiness": ["love", "peace", "food"],
    "sadness": ["death", "loss"],
    "anger": ["war", "conflict", "betrayal", "heartbreak"],
    "fear": ["supernatural", "paranormal", "mystery"],
    "surprise": ["gift"],
    "disgust": ["rotten"],
    "love": ["creator", "family", "electricity"]
}

OLLAMA_PORT = 11434

def topic_query_ollama(prompt):
    """
    Query Ollama with a prompt.
    """

    all_topics = [topic for topics in emotion_map.values() for topic in topics]
    ollama_prompt = (f"You have these topics to pick from: {', '.join(all_topics)}. If none of them fit, respond with 'indifference'. "
                     f"Respond with a single topic and the keyword from the prompt that made you pick it. "
                     f"Be concise, do not give any extra context other than the 2 words. If you respond with more or less than that we will break. The first word is the topic and the second is the keyword in the prompt that made you pick it: {prompt}")

    result = query_ollama(ollama_prompt)
    response_dict = utils.split_to_dict(result.lower())
    print(f'{response_dict["topic"]}: {response_dict["keyword"]}')
    return response_dict

def emotional_query_ollama(emotion, prompt):
    prompt = f"{prompt} Respond with the Emotion: {emotion}. But don't let on that you are being told to respond with this emotion."
    return query_ollama(prompt)


def query_ollama(prompt):
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        print(f'Prompt: {prompt}')
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        result = json.loads(response.text)
        print(f'Response: {result["response"]}')
        return result['response']
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama on 127.0.0.1:{OLLAMA_PORT}. Is it running?")
        print(f"Run 'ollama run {MODEL}' in a terminal and keep it open.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def extract_topic_option(topic_response):
    """Extract the first word from the topic response as the topic option."""
    if not topic_response:
        return "indifference"
    words = topic_response.split()
    return words[0] if words else "indifference"


def update_emotion_map(emotion, keyword):
    """
    Updates the emotion_map by adding the keyword to the topic list for the given emotion.

    Args:
        emotion (str): The emotion determined from the topic option.
        keyword (str): The keyword to add to the emotion_map.
    """
    if emotion in emotion_map and keyword not in emotion_map[emotion]:
        emotion_map[emotion].append(keyword)
        print(f"Added new topic '{keyword}' to emotion '{emotion}'")

def get_emotion(topic_option):
    """Return the emotion associated with the topic option, or 'indifference' if not found."""
    for emotion, topics in emotion_map.items():
        if topic_option in topics:
            return emotion
    return "indifference"

def main():
    print("Emotion Prompt System")
    print("Type your prompt (or 'quit' to exit)")
    time.sleep(2)  # Give Ollama time to start if just launched

    while True:
        original_prompt = input("> ").strip()

        if original_prompt.lower() == "quit":
            print("Goodbye!")
            break

        if not original_prompt:
            print("Please enter something!")
            continue

        # Step 1: Get the topic response from Ollama
        topic_response = topic_query_ollama(original_prompt)

        if not topic_response:
            print("Could not determine the topic. Is Ollama running?")
            continue

        # Step 2: Get the emotion based on the topic option and/or add keyword to emotion map
        topic = topic_response["topic"]
        emotion = get_emotion(topic)

        keyword = topic_response["keyword"]
        if keyword and emotion != "indifference":
            update_emotion_map(emotion, topic)

        # Step 3: Get the final response from Ollama using the new prompt
        final_response = emotional_query_ollama(emotion, original_prompt)

        if final_response:
            print(f"Topic: {topic}")
            print(f"Emotion: {emotion}")
            print(f"Response: {final_response}")
        else:
            print("Could not generate a response.")

if __name__ == "__main__":
    main()