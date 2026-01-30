from pathlib import Path
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

class SentimentAnalyzer:
    def __init__(self, model_path: str = "finbert"):
        """
        Initialize tokenizer and TF model once.
        model_path: local folder containing finbert files:
            - config.json
            - tf_model.h5
            - vocab.txt
            - tokenizer_config.json
            - special_tokens_map.json
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ValueError(f"Model path {model_path} does not exist.")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        # Load TF model
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            from_pt=False  # ensure TF only
        )

    def predict(self, texts):
        """
        texts: list of str
        returns: list of dicts with class probabilities
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="tf"
        )
        # Forward pass
        logits = self.model(**encoded).logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()

        # Map output to label names
        # For FinBERT-tone, labels are usually: ["positive", "neutral", "negative"]
        labels = ["positive", "neutral", "negative"]
        results = []
        for p in probs:
            results.append({labels[i]: float(p[i]) for i in range(len(labels))})
        return results
