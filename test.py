import requests

def test_detect_endpoint(model_name, input_text):
    """
    Sends a GET request to the /detect endpoint and prints the returned logits.

    Args:
        model_name (str): The name of the model to load (e.g., meta-llama/Llama-3.2-1B).
        input_text (str): The input text to get logits for.
    """
    url = "http://127.0.0.1:5000/detect"
    params = {
        "model_name": model_name,
        "text": input_text
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        logits = response.json()["logits"]
        print(f"Logits for the input text: {logits}")
    else:
        print(f"Error: {response.status_code} - {response.json()}")

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    input_text = "During photosynthesis in green plants"
    
    test_detect_endpoint(model_name, input_text)
