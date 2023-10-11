import os
import json
import textwrap
import requests

import cv2
import easyocr
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from tqdm import tqdm


class SegmentProcessor:
    def __init__(
        self,
        src_path,
        dest_path,
        src_lang,
        target_lang,
        lang_id: str = "en",
        gpu: bool = True,
    ):
        self.texts = {}
        self.reader = easyocr.Reader([lang_id], gpu=gpu)
        current_dir = os.path.dirname(__file__)
        static = os.path.join(current_dir, "static")
        self.fontpath = os.path.join(static, "SakalBharati.ttf")

        self.src_lang = src_lang
        self.target_lang = target_lang

        self.cap = cv2.VideoCapture(src_path)
        self.vid_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.out = cv2.VideoWriter(
            dest_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.cap.get(cv2.CAP_PROP_FPS),
            (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

    def inpaint(self):
        for key in self.texts.keys():
            bbox = self.texts[key]["coord"]
            self.texts[key]["point"] = np.array(bbox).reshape(-1, 2).astype(np.int32)

    def text_overlay(self, frame):
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        for key in self.texts.keys():
            bbox = self.texts[key]["coord"]
            text = self.texts[key]["translation"]
            (x1, y1), (x2, y2) = bbox[0], bbox[2]
            lines = 1
            while True:
                size = (y2 - y1 - ((y2 - y1) / 3)) // lines
                if draw.textlength(
                    text=textwrap.wrap(
                        text=text, width=(len(text) // lines), fix_sentence_endings=True
                    )[0],
                    font=ImageFont.truetype(font=self.fontpath, size=size),
                ) > (x2 - x1):
                    lines += 1
                else:
                    break
            self.texts[key]["lines"] = lines
            self.texts[key]["font_size"] = size

    def translate_batch(self, text_batch, source_lang: str, target_lang: str):
        url = "https://v2.fourie.ai/api/translatetext/batch"
        headers = {"staging": "beta"}
        data = {
            "text_batch": [text_batch],
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        response = requests.post(url, headers=headers, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            raise requests.exceptions.RequestException("No text supplied")

    def apply_ocr(self, frame):
        response = self.reader.readtext(frame, paragraph=True)
        self.texts = {}
        for detection in response:
            self.texts[detection[1]] = {
                "coord": detection[0],
            }
        return self.texts

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()

    def __call__(self, ocr_frame, start_frame, end_frame, align="left") -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, ocr_frame)
        ret, o_frame = self.cap.read()
        if not ret:
            raise Exception("Could not read `ocr_frame`")
        # self.texts = self.apply_ocr(o_frame)
        # if self.texts:
        #     translations = self.translate_batch(
        #         text_batch=list(self.texts.keys()),
        #         source_lang=self.src_lang,
        #         target_lang=self.target_lang,
        #     )

        #     for i, text in enumerate(self.texts.keys()):
        #         self.texts[text]["translation"] = translations["translated_sentences"][
        #             i
        #         ]
        # # self.inpaint()
        # self.text_overlay(o_frame)
        # print(self.texts)
        # # del self.texts["Skocd Berq]"]
        # with open("./static/config3.json", "w") as json_file:
        #     json.dump(self.texts, json_file, ensure_ascii=False, indent=4)
        with open("./static/config3.json", "r") as json_file:
            self.texts = json.load(json_file)
        # self.texts["Brain"]["translation"] = "ಮೆದುಳು"
        for frame_id in tqdm(range(start_frame, end_frame + 1)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.cap.read()
            for key in self.texts.keys():
                cv2.fillPoly(frame, [np.array(self.texts[key]["coord"])], (0, 0, 0))

            inpainted_frame = cv2.inpaint(
                frame,
                (frame == 0).all(axis=2).astype(np.uint8),
                inpaintRadius=4,
                flags=cv2.INPAINT_TELEA,
            )

            img_pil = Image.fromarray(inpainted_frame)
            draw = ImageDraw.Draw(img_pil)
            for key in self.texts.keys():
                text = self.texts[key]["translation"]
                bbox = self.texts[key]["coord"]
                (x1, y1), (x2, y2) = bbox[0], bbox[2]
                font = ImageFont.truetype(
                    font=self.fontpath, size=self.texts[key]["font_size"]
                )
                draw.multiline_text(
                    (x1, y1),
                    "\n".join(
                        textwrap.wrap(
                            text,
                            width=(len(text) // self.texts[key]["lines"]),
                            break_long_words=False,
                        )
                    ),
                    font=font,
                    fill=(0, 0, 0),
                    stroke_width=2,
                    stroke_fill=(255, 255, 255),
                    align=align,
                )

            self.out.write(np.array(img_pil))
        self.release()


if __name__ == "__main__":
    pipe = SegmentProcessor(
        src_path="../data/SkoolBeepSample1_en_kn.mp4",
        dest_path=f"../data/SkoolBeep2/{2}-e.mp4",
        src_lang="english",
        target_lang="kannada",
    )
    start_frame = 160
    end_frame = 375
    ocr_frame = 190
    pipe(
        ocr_frame=ocr_frame,
        start_frame=start_frame,
        end_frame=end_frame,
        align="center",
    )
