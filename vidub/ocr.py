import easyocr
from typing import *

import vidub.config as config


class OCR:
    """
    Class for performing Optical Character Recognition (OCR) on images.
    """

    def __init__(self, lang_id: str = config.gpu, gpu: bool = config.gpu):
        self.reader = easyocr.Reader([lang_id], gpu=gpu)
        self.THRESH = config.conf_threshold

    def process(self, frame) -> Dict:
        """
        Perform OCR on an image frame.

        Args:
            frame: Input image frame.

        Returns:
            Dict: Detected text and related information.
        """
        response = self.reader.readtext(frame, paragraph=True)
        ocr_output = {}
        for detection in response:
            ocr_output[detection[1]] = {
                "coord": detection[0],
            }
        return ocr_output
