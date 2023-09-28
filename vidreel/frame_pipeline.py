from .frame_processor import FrameProcessor
from .ocr import OCR
from translator import Translator


class FramePipeline:
    """
    Class for orchestrating the text translation pipeline.
    """

    def __init__(self, source: str, target: str, lang_id: str):
        self.translator = Translator()
        self.ocr_processor = OCR(lang_id)
        self.image_processor = FrameProcessor()
        self.source = source
        self.target = target

    def __call__(self, frame):
        """
        Process an image frame with text translation.

        Args:
            frame: Input image frame.

        Returns:
            Image: Translated and processed image.
        """
        ocr_output = self.ocr_processor.process(frame)
        if ocr_output:
            translations = self.translator.translate_batch(
                text_batch=list(ocr_output.keys()),
                source_lang=self.source,
                target_lang=self.target,
            )

            for i, text in enumerate(ocr_output.keys()):
                ocr_output[text]["translation"] = translations["translated_sentences"][
                    i
                ]

        inpainted_frame = self.image_processor.inpaint(frame, ocr_output)
        final_image = self.image_processor.text_overlay(inpainted_frame, ocr_output)
        return final_image
