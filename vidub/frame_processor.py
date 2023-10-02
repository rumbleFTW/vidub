import os
import textwrap

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

import vidub.config as config


class FrameProcessor:
    """
    Class for image processing, including inpainting and text overlay.
    """

    def inpaint(self, frame, ocr_output):
        """
        Inpaint the image based on OCR-detected text regions.

        Args:
            frame: Input image frame.
            ocr_output: Detected text and related information.

        Returns:
            Image: Inpainted image.
        """
        if not ocr_output:
            return frame
        for key in ocr_output.keys():
            bbox = ocr_output[key]["coord"]
            points = np.array(bbox).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(frame, [points], (0, 0, 0))

        inpainted_frame = cv2.inpaint(
            frame,
            (frame == 0).all(axis=2).astype(np.uint8),
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )
        return inpainted_frame

    def text_overlay(self, frame, ocr_output):
        """
        Overlay translated text on the image.

        Args:
            frame: Input image frame.
            ocr_output: Detected text and related information.

        Returns:
            Image: Image with overlaid text.
        """
        current_dir = os.path.dirname(__file__)
        static = os.path.join(current_dir, "static")
        fontpath = os.path.join(static, "SakalBharati.ttf")
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        for key in ocr_output.keys():
            bbox = ocr_output[key]["coord"]
            text = ocr_output[key]["translation"]
            (x1, y1), (x2, y2) = bbox[0], bbox[2]
            lines = 1
            while True:
                size = (y2 - y1 - ((y2 - y1) / 3)) // lines
                if draw.textlength(
                    text=textwrap.wrap(
                        text=text, width=(len(text) // lines), fix_sentence_endings=True
                    )[0],
                    font=ImageFont.truetype(font=fontpath, size=size),
                ) > (x2 - x1):
                    lines += 1
                else:
                    break

            font = ImageFont.truetype(font=fontpath, size=size)
            draw.multiline_text(
                (x1, y1),
                "\n".join(
                    textwrap.wrap(
                        text, width=(len(text) // lines), break_long_words=False
                    )
                ),
                font=font,
                fill=config.text_color,
                stroke_width=config.text_stroke_width,
                stroke_fill=config.text_stroke_color,
                align="center",
            )
        img = np.array(img_pil)
        return img
