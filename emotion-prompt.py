try:
    import requests
except ImportError:
    print("Error:  module not found. Install it with 'pip install requests'.")
    exit(1)

import json
import utils
import threading

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
    "love": ["creator", "family", "electricity"],
    "indifference": []
}

OLLAMA_PORT = 11434

def topic_keyword_query_ollama(input_prompt):
    """
    Query Ollama with a prompt.
    """

    ollama_prompt = (f"Respond with a single topic and the keyword from the prompt that made you pick it. "
                     f"Be concise, do not give any extra context other than the 2 words. "
                     f"If you respond with more or less than that we will break. "
                     f"The first word is the topic and the second is the keyword in the prompt that made you pick it. "
                     f"The keyword has to be a word that is in the prompt."
                     f"Here is the prompt: {input_prompt}")

    result = query_ollama(ollama_prompt)
    print(result)
    response_dict = utils.split_to_dict(result.lower())
    response_dict["keyword"] =  response_dict["keyword"] if "keyword" in response_dict else None
    print(f'{response_dict["topic"]}: {response_dict["keyword"] if "keyword" in response_dict else ""}')
    return response_dict

def emotion_encoding(emotion, input_prompt):
    input_prompt = f"{input_prompt}. Respond with you having the Emotion of {emotion}. But don't let on that you are being told to respond with this emotion."
    return query_ollama(input_prompt)


def query_ollama(input_prompt):
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"

    payload = {
        "model": MODEL,
        "prompt": input_prompt,
        "stream": False
    }

    try:
        print(f'Prompt: {input_prompt}')
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


def update_emotion_map(emotion, topic, keyword):
    """
    Updates the emotion_map by adding the keyword to the topic list for the given emotion.

    Args:
        emotion (str): The emotion determined from the topic option.
        topic (str): The topic determined from the topic option.
        keyword (str): The keyword to add to the emotion_map.
    """
    print("Checking if we need to update the emotion map.")
    if keyword and emotion != "indifference":
        if emotion in emotion_map and topic not in emotion_map[emotion]:
            emotion_map[emotion].append(topic)
            print(f"Added new topic '{topic}' to emotion '{emotion}'")
    print("Updated emotion map.")

def get_emotion(topic_option):
    """Return the emotion associated with the topic option, or 'indifference' if not found."""
    for emotion, topics in emotion_map.items():
        if topic_option in topics:
            return emotion
    return "indifference"

def prompt():
    while True:
        original_prompt = input("> ").strip()

        if original_prompt.lower() == "quit":
            print("Goodbye!")
            break

        if not original_prompt:
            print("Please enter something!")
            continue

        # Step 1: Get the topic response from Ollama
        topic_response = topic_keyword_query_ollama(original_prompt)

        if not topic_response:
            print("Could not determine the topic. Is Ollama running?")
            continue

        # Step 2: Get the emotion based on the topic option and/or add keyword to emotion map
        topic = topic_response["topic"]
        emotion = get_emotion(topic)

        update_emotion_map_thread = threading.Thread(target=update_emotion_map, args=(emotion, topic, topic_response["keyword"]))
        update_emotion_map_thread.start()

        # Step 3: Get the final response from Ollama using the new prompt
        final_response = emotion_encoding(emotion, original_prompt)

        if final_response:
            print(f"Topic: {topic}")
            print(f"Emotion: {emotion}")
            print(f"Response: {final_response}")
        else:
            print("Could not generate a response.")
        update_emotion_map_thread.join()

def main():
    print("Emotion Prompt System")
    print("Type your prompt (or 'quit' to exit)")

    prompt()

if __name__ == "__main__":
    main()