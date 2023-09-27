**How to use**

**Setup**

#### Install the necessary libraries using pip:

```bash
pip install -r requirements.txt
```

**Examples**

1. The following example shows how to translate text in an single frame(image) from English to French:

```python
from frame_pipeline import FramePipeline

# Create a FramePipeline object with the source and target languages.
pipeline = FramePipeline(source="english", target="french", lang_id="en")

# Load the image frame.
frame = cv2.imread("image.jpg")

# Process the frame with the pipeline.
translated_frame = pipeline.process(frame)

# Save the translated frame.
cv2.imwrite("translated_image.jpg", translated_frame)
```

2. The following example shows how to translate text in an video from English to Hindi:

```python

from scd import SceneChangeDetection
from video_pipeline import VideoPipeline

# Create a SceneChangeDetection object.
detector = SceneChangeDetection(base="ContentDetector", min_scene_len=25)

# Detect timestamps from videos using the SceneChangeDetection object
video_timestamps = detector.detect(path="/path/to/source.mp4")

# Create a VideoPipeline object with the video paths, source and target languages.
source_path = "/path/to/source.mp4"
dest_path = "/path/to/dest.mp4"
source_lang = "english"
target_lang = "hindi"
lang_id = "en"
pipe = VideoPipeline(
    source_path, dest_path, source_lang, target_lang, lang_id
)

# Call the pipeline object to apply the translation
pipe(video_timestamps)
```
