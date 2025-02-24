try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Install it with 'pip install requests'.")
    exit(1)

import json
import time

# Define the model to use
MODEL = "llama3.2"

# Dictionary mapping emotions to lists of topics
emotion_map = {
    "happiness": ["love", "peace", "food"],
    "sadness": ["death", "loss"],
    "anger": ["war", "conflict", "betrayal", "heartbreak"],
    "fear": ["supernatural", "paranormal", "mystery"],
    "surprise": ["gift"],
    "disgust": ["rotten"]
}

OLLAMA_PORT = 11434

def query_ollama(prompt, purpose="topic"):
    """
    Query Ollama with a prompt.
    If purpose is 'topic', it asks for a topic and keyword.
    If purpose is 'response', it asks for a response incorporating the emotion.
    """
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"

    if purpose == "topic":
        all_topics = [topic for topics in emotion_map.values() for topic in topics]
        ollama_prompt = (f"You have these topics to pick from: {', '.join(all_topics)}. If none of them fit, respond with 'indifference'. "
                         f"Respond with a single topic without quotes and the keyword that made you pick it. "
                         f"Be concise, do not give any extra context and only give max 2 words. The first one is the topic and the second is the keyword that made you pick it: {prompt}")
    elif purpose == "response":
        ollama_prompt = prompt  # Use the prompt as is for generating the response

    payload = {
        "model": MODEL,
        "prompt": ollama_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        result = json.loads(response.text)
        if purpose == "topic":
            # For topic, strip, lowercase, and remove trailing periods
            sanitized_result = result["response"].strip().lower().rstrip('.')
            print(sanitized_result)
            return sanitized_result
        else:
            # For response, only strip whitespace to preserve formatting
            return result["response"].strip()
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
        topic_response = query_ollama(original_prompt, purpose="topic")

        if not topic_response:
            print("Could not determine the topic. Is Ollama running?")
            continue

        # Step 2: Extract the topic option (first word)
        topic_option = extract_topic_option(topic_response)

        # Step 3: Get the emotion based on the topic option
        emotion = get_emotion(topic_option)

        # Step 4: Create a new prompt by appending the emotion
        new_prompt = f"{original_prompt} Respond with the Emotion: {emotion}. But don't let on that you are being told to respond with this emotion."

        # Step 5: Get the final response from Ollama using the new prompt
        final_response = query_ollama(new_prompt, purpose="response")

        if final_response:
            print(f"Topic: {topic_option}")
            print(f"Emotion: {emotion}")
            print(f"Response: {final_response}")
        else:
            print("Could not generate a response.")

if __name__ == "__main__":
    main()