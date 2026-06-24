import time

from google import genai # New package
from google.genai import types # For configuration

from typing import List, Optional
import os
import glob
import tqdm
import shutil
from ollamaocr import vlmocr
from keys import gemini


class GeminiOCR(vlmocr):
    def __init__(this, model_name: str, charset: Optional[List[str]] = None,
                 script=None, prompt=None, api_key: str = None):
        this.charset = charset
        this.script = script
        this.model_name = model_name

        # The new client-based initialization
        this.client = genai.Client(api_key=api_key)

        if prompt:
            this.prompt = prompt
        else:
            this.prompt = this.get_default_prompt()

    def ocr(this, imgpath) -> str:
        time.sleep(4.1)  # Rate limit padding

        with open(imgpath, "rb") as f:
            img_data = f.read()

        # The new SDK handles bytes directly in the content list more gracefully
        response = this.client.models.generate_content(
            model=this.model_name,
            contents=[
                this.prompt,
                types.Part.from_bytes(
                    data=img_data,
                    mime_type="image/jpeg" if imgpath.lower().endswith((".jpg", ".jpeg")) else "image/png"
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )

        return response.text if response.text else ""
    def mkmsg(this, imgpath):
        """Gemini prefers a list of parts: strings and dicts for blobs."""
        with open(imgpath, "rb") as f:
            img_data = f.read()

        # Determine mime type based on extension
        mime_type = "image/jpeg" if imgpath.lower().endswith((".jpg", ".jpeg")) else "image/png"

        return [
            this.prompt,
            {
                "mime_type": mime_type,
                "data": img_data
            }
        ]


class GeminiOCR_nocset(GeminiOCR):
    def get_default_prompt(this) ->str:
        pmpt= "Do OCR and return only transcription";
        if(this.script is not None):
            pmpt += "("+this.script+")";
        pmpt+=".\n"
        return pmpt;

class GeminiOCR_DoOCR(GeminiOCR):
    def get_default_prompt(this) ->str:
        pmpt= "Do OCR and return only transcription";
        pmpt+=".\n"
        return pmpt;




if __name__ == '__main__':
    # Execute
    # Make sure you have run `ollama pull llama3.2-vision` or `ollama pull llava`
    MODEL = "gemini-3.1-flash-lite-preview"
    from cset_yi import cset,script
    FOLDER_PATH = "/home/catknight/data/eccv_rebuttal/selected"
    FORCE_PMPT = None

    # OUT_PATH = "/home/catknight/data/eccv_rebuttal/predicts/" + MODEL.replace(":", "_") + ""
    #
    # FORCE_PMPT = None
    #
    # O = GeminiOCR(MODEL, cset, script, FORCE_PMPT,gemini());
    # O.run_sequential_ocr(FOLDER_PATH, OUT_PATH)

    # OUT_PATH = "/home/catknight/data/eccv_rebuttal/predicts_nocset/" + MODEL.replace(":", "_") + ""
    # O = GeminiOCR_nocset(MODEL, cset, script, FORCE_PMPT,gemini());
    # its carried out via my 10-feet-pole bcs i don't want to install google framework.
    OUT_PATH = "/home/catknight/data/eccv_rebuttal/predicts_plain/" + MODEL.replace(":", "_") + ""
    O = GeminiOCR_DoOCR(MODEL, cset, script, FORCE_PMPT,gemini());
    O.run_sequential_ocr(FOLDER_PATH, OUT_PATH)
