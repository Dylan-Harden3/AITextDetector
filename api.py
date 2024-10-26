import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)

current_model_name = None
current_model = None
current_tokenizer = None

def load_model_and_tokenizer(model_name):
    """
    Loads both the model and the tokenizer for the specified model. If the model is
    different from the currently loaded one, it clears the current model and tokenizer
    from memory and loads the new ones with 4-bit quantization.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    global current_model_name, current_model, current_tokenizer

    if model_name != current_model_name:
        if current_model is not None:
            del current_model
            del current_tokenizer
            torch.cuda.empty_cache()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
        HF_CACHE_DIR = os.getenv("HUGGING_FACE_CACHE_DIR")

        current_tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=HF_API_TOKEN, cache_dir=HF_CACHE_DIR
        )

        current_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            output_attentions=True,
            use_auth_token=HF_API_TOKEN,
            cache_dir=HF_CACHE_DIR
        )

        current_model_name = model_name

    return current_model, current_tokenizer


@app.route("/detect", methods=["GET"])
def detect_ai_text():
    model_name = request.args.get("model_name")
    input_text = request.args.get("text")

    if not model_name or not input_text:
        return jsonify({"error": "Missing model_name or text parameter"}), 400

    model, tokenizer = load_model_and_tokenizer('meta-llama/Llama-3.1-8B-Instruct')


    return jsonify(
        {
        }
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)