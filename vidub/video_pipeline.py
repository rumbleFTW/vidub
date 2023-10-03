import cv2
from tqdm import tqdm
from typing import *

from vidub.frame_processor import FrameProcessor
from vidub.ocr import OCR
from vidub.translator import Translator


class VideoPipeline:
    """
    Class for orchestrating the text translation pipeline on videos.
    """

    def __init__(
        self, src_path: str, dest_path: str, source: str, target: str, lang_id: str
    ) -> None:
        self.translator = Translator()
        self.ocr_processor = OCR(lang_id)
        self.image_processor = FrameProcessor()
        self.source = source
        self.target = target

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

    def init_pipeline(self, frame):
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
        return ocr_output, final_image

    def continue_pipeline(self, conf, frame):
        inpainted_frame = self.image_processor.inpaint(frame, conf)
        final_image = self.image_processor.text_overlay(inpainted_frame, conf)
        return final_image

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()

    def __call__(self, video_timestamps: List[Tuple[Tuple[float, int]]]):
        """
        Process a video with text translation.

        Args:
            video_timestamps: Timestamps of frames where texts appear and disappear in the video.

        Returns:
            None.
        """

        text_flag = False
        sc_ptr = 0
        frame_id = 1
        with tqdm(total=self.vid_len) as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                curr_timestamp = video_timestamps[sc_ptr]
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = self.cap.read()

                if frame_id == curr_timestamp[0][1]:
                    det, frame = self.init_pipeline(frame)
                    text_flag = True

                elif frame_id == curr_timestamp[1][1] - 1:
                    det = {}
                    sc_ptr += 1
                    text_flag = False

                elif text_flag:
                    frame = self.continue_pipeline(det, frame)
                self.out.write(frame)
                frame_id += 1
                pbar.update(1)

        self.release()
