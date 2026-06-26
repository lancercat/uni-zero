from ollamaocr import vlmocr

URL="http://localhost:14514/v1"



# Execute
# Make sure you have run `ollama pull llama3.2-vision` or `ollama pull llava`
MODEL = "qwen3.6:27b"

FOLDER_PATH = "/run/media/lasercat/writebuffer/tmp/synthyi_5k/imgs/sampled_images/"
OUT_PATH = "/run/media/lasercat/writebuffer/tmp/synthyi_5k/predicts/"+MODEL.replace(":","_")+""
script="modern Yi";
cset=None;
FORCE_PMPT=None

O=vlmocr(MODEL,cset,script,FORCE_PMPT,URL);
O.run_sequential_ocr(FOLDER_PATH, OUT_PATH)