import requests
from typing import *


class Translator:
    """
    Class for handling text translation using fourie's translation API.
    """

    def __init__(self):
        self.url = "https://v2.fourie.ai/api/translatetext/batch"
        self.headers = {"staging": "beta"}

    def translate_batch(
        self, text_batch: List[str], source_lang: str, target_lang: str
    ) -> Dict:
        """
        Translate a batch of text from the source language to the target language.

        Args:
            text_batch (List[str]): List of texts to be translated.
            source_lang (str): Source language code (e.g., "en" for English).
            target_lang (str): Target language code (e.g., "fr" for French).

        Returns:
            Dict: Translated text and related information.
        """
        data = {
            "text_batch": [text_batch],
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        response = requests.post(self.url, headers=self.headers, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise requests.exceptions.RequestException("No text supplied")
