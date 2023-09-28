# SceneChangeDetector Base model
import os

scd_model_name = "ContentDetector"
min_scene_len = 15

# OCR parameters
conf_threshold = 0.1

lang_id = "en"
gpu = True

# Frame processor parameters
asset_dir = os.path.join(os.path.dirname(__file__), "assets")
fontpath = os.path.join(asset_dir, "SakalBharati.ttf")
text_color = (255, 255, 255)
text_stroke_width = 3
text_stroke_color = (0, 0, 0)
