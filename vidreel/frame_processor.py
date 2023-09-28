import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

import .config


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
        fontpath = config.fontpath
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        for key in ocr_output.keys():
            bbox = ocr_output[key]["coord"]
            text = ocr_output[key]["translation"]
            (x1, y1), (x2, y2) = bbox[0], bbox[2]
            font_size = int(y2 - y1)
            font = ImageFont.truetype(fontpath, font_size - (font_size // 3))
            draw.text(
                (x1, y1),
                text,
                font=font,
                fill=config.text_color,
                stroke_width=config.text_stroke_width,
                stroke_fill=config.text_stroke_color,
            )
        img = np.array(img_pil)
        return img
