import time
from typing import *

from scenedetect import detect, ContentDetector, ThresholdDetector, AdaptiveDetector
import vidub.config as config


class SceneChangeDetection:
    """
    SceneChangeDetection is a class for detecting scene changes in a video using python scenedetect library.

    Attributes:
        base (str): The detection algorithm to use ('ContentDetector', 'ThresholdDetector', or 'AdaptiveDetector').
        Fore more info on each of the algorithms mentioned, visit: https://www.scenedetect.com/docs/latest
        **kwargs: Additional keyword arguments specific to the chosen detection algorithm.

    Methods:
        detect(path: str) -> List[Tuple[Tuple[float, int]]]: Detect scene changes in the video at the specified path and return a list of frame timestamps where scene changes occur.

    Example usage:
        model = SceneChangeDetection(base='ContentDetector', min_scene_len=25)
        ret = model.detect("test/video.mp4", verbose=True)
    """

    def __init__(self, base, **kwargs) -> None:
        self.base = base
        self.kwargs = kwargs

        if self.base.lower() == "contentdetector":
            self._model = ContentDetector(**self.kwargs)
        elif self.base.lower() == "thresholddetector":
            self._model = ThresholdDetector(**self.kwargs)
        elif self.base.lower() == "adaptivedetector":
            self._model = AdaptiveDetector(**self.kwargs)
        else:
            raise Exception(
                "Invalid model base; Available options: ContentDetector, ThresholdDetector, or AdaptiveDetector"
            )

    def detect(
        self, path: str, verbose=True, benchmark=True
    ) -> List[Tuple[Tuple[float, int]]]:
        """
        Detect scene changes in the video at the specified path.

        Args:
            path (str): The path to the video file.
            verbose (bool): If True, display progress and log information during scene detection.
            benchmark (bool): If True, print the time taken for the scene detection process.
        Returns:
            List[Tuple[Tuple[float, int]]]: A list of tuple of boundaries of each scene. Each boundary has a `time` and `frame_number` for reference.
        """
        start_time = time.time()
        time_stamps = detect(
            video_path=path, detector=self._model, show_progress=verbose
        )
        # By default, `detect` method returns a List[Tuple[FrameTimecode, FrameTimecode]], which might be complex for decoding and testing purpose. So we will simplify it into List[Tuple[Tuple[float, int]]]
        ret = []
        for item in time_stamps:
            start = (item[0].get_seconds(), item[0].get_frames())
            end = (item[1].get_seconds(), item[1].get_frames())
            ret.append((start, end))
        end_time = time.time()
        if benchmark:
            print(f"Took {end_time-start_time} seconds")
        return ret
