from transformers import T5ForConditionalGeneration, T5Tokenizer


class Summarizer:
    """
    Lightweight abstractive summariser based on `t5-small`.
    Keeps everything on CPU and respects the 1 GB image cap (model ≈231 MB).
    """

    def __init__(self, model_name: str = "t5-small", device: str = "cpu"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def summarize(self, text: str, max_len: int = 64, min_len: int = 16) -> str:
        text = "summarize: " + text.replace("\n", " ")
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(self.device)

        ids = self.model.generate(
            inputs,
            num_beams=4,
            length_penalty=2.0,
            max_length=max_len,
            min_length=min_len,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)
