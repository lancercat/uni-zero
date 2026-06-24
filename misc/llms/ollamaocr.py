import glob
import os
import io
import base64
import shutil

import tqdm
from openai import OpenAI
from PIL import Image
from typing import List,Optional
import cv2


class vlmocr:
    def get_default_prompt(this) ->str:
        pmpt= "Do OCR and return only transcription";
        if(this.script is not None):
            pmpt += "("+this.script+")";
        pmpt+=".\n"
        if(this.charset is not None):
            pmpt+="Charset: " + str(this.charset);
        return pmpt;


    def __init__(this,model_name,charset:Optional[List[str]], script=None,prompt=None,url="http://localhost:11434/v1",key="ollama" ):
        this.charset=charset;
        this.script=script;
        this.model_name=model_name;
        this.client = OpenAI(
            base_url=url,
            api_key=key # Required but ignored by Ollama
        )

        if(prompt):
            this.prompt=prompt;
        else:
            this.prompt=this.get_default_prompt()

    def mkmsg(self, imgpath):
        # 1. Read image with OpenCV
        img = cv2.imread(imgpath,cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {imgpath}")

        # 3. Encode to JPEG memory buffer
        # Note: We use cv2.imencode instead of PIL's .save()
        success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            raise ValueError("Could not encode image to JPEG")

        # 4. Convert to Base64
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ]
    def ocr(this, imgpath) -> str:
        """Sends a single PIL image to Ollama for text extraction."""
        # Point to your local Ollama instance

        # Convert PIL to Base64

        response = this.client.chat.completions.create(
            model=this.model_name,
            messages=this.mkmsg(imgpath),
            temperature=0,
        )
        return response.choices[0].message.content


    def run_sequential_ocr(this,src_folder,dst_folder):
        # 1. Find all images
        shutil.rmtree(dst_folder,ignore_errors=True);
        os.makedirs(dst_folder);
        files = sorted(glob.glob(os.path.join(src_folder, "*.[jp][pn]g")))

        print(f"--- Starting Ollama OCR Job: {len(files)} files found ---")

        for i, file_path in tqdm.tqdm(enumerate(files, 1)):
            file_name = os.path.basename(file_path)

            try:
                text=this.ocr(file_path)

                    # 3. Save the result
                output_path = os.path.splitext(os.path.join(dst_folder,os.path.basename(file_path)))[0] + ".txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)

            except Exception as e:
                print(f"\n[!] Failed {file_name}: {e}")

        print("\n--- All files processed ---")

class deepseekocr(vlmocr):
    def get_default_prompt(this):
        return "Extract the text in the image."

if __name__ == '__main__':

    URL="http://localhost:14514/v1"

    FORCE_PMPT=None


    # Execute
    # Make sure you have run `ollama pull llama3.2-vision` or `ollama pull llava`
    # MODEL = "deepseek-ocr:latest"
    #
    # FOLDER_PATH = "/run/media/lasercat/writebuffer/tmp/synthyi_5k/imgs/sampled_images/"
    # OUT_PATH = "/run/media/lasercat/writebuffer/tmp/synthyi_5k/predicts/"+MODEL.replace(":","_")+""
    # script="modern Yi";
    # cset=["ꀀ","ꀁ","ꀂ","ꀃ","ꀄ","ꀅ","ꀆ","ꀇ","ꀈ","ꀉ","ꀊ","ꀋ","ꀌ","ꀍ","ꀎ","ꀏ","ꀐ","ꀑ","ꀒ","ꀓ","ꀔ","ꀕ","ꀖ","ꀗ","ꀘ","ꀙ","ꀚ","ꀛ","ꀜ","ꀝ","ꀞ","ꀟ","ꀠ","ꀡ","ꀢ","ꀣ","ꀤ","ꀥ","ꀦ","ꀧ","ꀨ","ꀩ","ꀪ","ꀫ","ꀬ","ꀭ","ꀮ","ꀯ","ꀰ","ꀱ","ꀲ","ꀳ","ꀴ","ꀵ","ꀶ","ꀷ","ꀸ","ꀹ","ꀺ","ꀻ","ꀼ","ꀽ","ꀾ","ꀿ","ꁀ","ꁁ","ꁂ","ꁃ","ꁄ","ꁅ","ꁆ","ꁇ","ꁈ","ꁉ","ꁊ","ꁋ","ꁌ","ꁍ","ꁎ","ꁏ","ꁐ","ꁑ","ꁒ","ꁓ","ꁔ","ꁕ","ꁖ","ꁗ","ꁘ","ꁙ","ꁚ","ꁛ","ꁜ","ꁝ","ꁞ","ꁟ","ꁠ","ꁡ","ꁢ","ꁣ","ꁤ","ꁥ","ꁦ","ꁧ","ꁨ","ꁩ","ꁪ","ꁫ","ꁬ","ꁭ","ꁮ","ꁯ","ꁰ","ꁱ","ꁲ","ꁳ","ꁴ","ꁵ","ꁶ","ꁷ","ꁸ","ꁹ","ꁺ","ꁻ","ꁼ","ꁽ","ꁾ","ꁿ","ꂀ","ꂁ","ꂂ","ꂃ","ꂄ","ꂅ","ꂆ","ꂇ","ꂈ","ꂉ","ꂊ","ꂋ","ꂌ","ꂍ","ꂎ","ꂏ","ꂐ","ꂑ","ꂒ","ꂓ","ꂔ","ꂕ","ꂖ","ꂗ","ꂘ","ꂙ","ꂚ","ꂛ","ꂜ","ꂝ","ꂞ","ꂟ","ꂠ","ꂡ","ꂢ","ꂣ","ꂤ","ꂥ","ꂦ","ꂧ","ꂨ","ꂩ","ꂪ","ꂫ","ꂬ","ꂭ","ꂮ","ꂯ","ꂰ","ꂱ","ꂲ","ꂳ","ꂴ","ꂵ","ꂶ","ꂷ","ꂸ","ꂹ","ꂺ","ꂻ","ꂼ","ꂽ","ꂾ","ꂿ","ꃀ","ꃁ","ꃂ","ꃃ","ꃄ","ꃅ","ꃆ","ꃇ","ꃈ","ꃉ","ꃊ","ꃋ","ꃌ","ꃍ","ꃎ","ꃏ","ꃐ","ꃑ","ꃒ","ꃓ","ꃔ","ꃕ","ꃖ","ꃗ","ꃘ","ꃙ","ꃚ","ꃛ","ꃜ","ꃝ","ꃞ","ꃟ","ꃠ","ꃡ","ꃢ","ꃣ","ꃤ","ꃥ","ꃦ","ꃧ","ꃨ","ꃩ","ꃪ","ꃫ","ꃬ","ꃭ","ꃮ","ꃯ","ꃰ","ꃱ","ꃲ","ꃳ","ꃴ","ꃵ","ꃶ","ꃷ","ꃸ","ꃹ","ꃺ","ꃻ","ꃼ","ꃽ","ꃾ","ꃿ","ꄀ","ꄁ","ꄂ","ꄃ","ꄄ","ꄅ","ꄆ","ꄇ","ꄈ","ꄉ","ꄊ","ꄋ","ꄌ","ꄍ","ꄎ","ꄏ","ꄐ","ꄑ","ꄒ","ꄓ","ꄔ","ꄕ","ꄖ","ꄗ","ꄘ","ꄙ","ꄚ","ꄛ","ꄜ","ꄝ","ꄞ","ꄟ","ꄠ","ꄡ","ꄢ","ꄣ","ꄤ","ꄥ","ꄦ","ꄧ","ꄨ","ꄩ","ꄪ","ꄫ","ꄬ","ꄭ","ꄮ","ꄯ","ꄰ","ꄱ","ꄲ","ꄳ","ꄴ","ꄵ","ꄶ","ꄷ","ꄸ","ꄹ","ꄺ","ꄻ","ꄼ","ꄽ","ꄾ","ꄿ","ꅀ","ꅁ","ꅂ","ꅃ","ꅄ","ꅅ","ꅆ","ꅇ","ꅈ","ꅉ","ꅊ","ꅋ","ꅌ","ꅍ","ꅎ","ꅏ","ꅐ","ꅑ","ꅒ","ꅓ","ꅔ","ꅕ","ꅖ","ꅗ","ꅘ","ꅙ","ꅚ","ꅛ","ꅜ","ꅝ","ꅞ","ꅟ","ꅠ","ꅡ","ꅢ","ꅣ","ꅤ","ꅥ","ꅦ","ꅧ","ꅨ","ꅩ","ꅪ","ꅫ","ꅬ","ꅭ","ꅮ","ꅯ","ꅰ","ꅱ","ꅲ","ꅳ","ꅴ","ꅵ","ꅶ","ꅷ","ꅸ","ꅹ","ꅺ","ꅻ","ꅼ","ꅽ","ꅾ","ꅿ","ꆀ","ꆁ","ꆂ","ꆃ","ꆄ","ꆅ","ꆆ","ꆇ","ꆈ","ꆉ","ꆊ","ꆋ","ꆌ","ꆍ","ꆎ","ꆏ","ꆐ","ꆑ","ꆒ","ꆓ","ꆔ","ꆕ","ꆖ","ꆗ","ꆘ","ꆙ","ꆚ","ꆛ","ꆜ","ꆝ","ꆞ","ꆟ","ꆠ","ꆡ","ꆢ","ꆣ","ꆤ","ꆥ","ꆦ","ꆧ","ꆨ","ꆩ","ꆪ","ꆫ","ꆬ","ꆭ","ꆮ","ꆯ","ꆰ","ꆱ","ꆲ","ꆳ","ꆴ","ꆵ","ꆶ","ꆷ","ꆸ","ꆹ","ꆺ","ꆻ","ꆼ","ꆽ","ꆾ","ꆿ","ꇀ","ꇁ","ꇂ","ꇃ","ꇄ","ꇅ","ꇆ","ꇇ","ꇈ","ꇉ","ꇊ","ꇋ","ꇌ","ꇍ","ꇎ","ꇏ","ꇐ","ꇑ","ꇒ","ꇓ","ꇔ","ꇕ","ꇖ","ꇗ","ꇘ","ꇙ","ꇚ","ꇛ","ꇜ","ꇝ","ꇞ","ꇟ","ꇠ","ꇡ","ꇢ","ꇣ","ꇤ","ꇥ","ꇦ","ꇧ","ꇨ","ꇩ","ꇪ","ꇫ","ꇬ","ꇭ","ꇮ","ꇯ","ꇰ","ꇱ","ꇲ","ꇳ","ꇴ","ꇵ","ꇶ","ꇷ","ꇸ","ꇹ","ꇺ","ꇻ","ꇼ","ꇽ","ꇾ","ꇿ","ꈀ","ꈁ","ꈂ","ꈃ","ꈄ","ꈅ","ꈆ","ꈇ","ꈈ","ꈉ","ꈊ","ꈋ","ꈌ","ꈍ","ꈎ","ꈏ","ꈐ","ꈑ","ꈒ","ꈓ","ꈔ","ꈕ","ꈖ","ꈗ","ꈘ","ꈙ","ꈚ","ꈛ","ꈜ","ꈝ","ꈞ","ꈟ","ꈠ","ꈡ","ꈢ","ꈣ","ꈤ","ꈥ","ꈦ","ꈧ","ꈨ","ꈩ","ꈪ","ꈫ","ꈬ","ꈭ","ꈮ","ꈯ","ꈰ","ꈱ","ꈲ","ꈳ","ꈴ","ꈵ","ꈶ","ꈷ","ꈸ","ꈹ","ꈺ","ꈻ","ꈼ","ꈽ","ꈾ","ꈿ","ꉀ","ꉁ","ꉂ","ꉃ","ꉄ","ꉅ","ꉆ","ꉇ","ꉈ","ꉉ","ꉊ","ꉋ","ꉌ","ꉍ","ꉎ","ꉏ","ꉐ","ꉑ","ꉒ","ꉓ","ꉔ","ꉕ","ꉖ","ꉗ","ꉘ","ꉙ","ꉚ","ꉛ","ꉜ","ꉝ","ꉞ","ꉟ","ꉠ","ꉡ","ꉢ","ꉣ","ꉤ","ꉥ","ꉦ","ꉧ","ꉨ","ꉩ","ꉪ","ꉫ","ꉬ","ꉭ","ꉮ","ꉯ","ꉰ","ꉱ","ꉲ","ꉳ","ꉴ","ꉵ","ꉶ","ꉷ","ꉸ","ꉹ","ꉺ","ꉻ","ꉼ","ꉽ","ꉾ","ꉿ","ꊀ","ꊁ","ꊂ","ꊃ","ꊄ","ꊅ","ꊆ","ꊇ","ꊈ","ꊉ","ꊊ","ꊋ","ꊌ","ꊍ","ꊎ","ꊏ","ꊐ","ꊑ","ꊒ","ꊓ","ꊔ","ꊕ","ꊖ","ꊗ","ꊘ","ꊙ","ꊚ","ꊛ","ꊜ","ꊝ","ꊞ","ꊟ","ꊠ","ꊡ","ꊢ","ꊣ","ꊤ","ꊥ","ꊦ","ꊧ","ꊨ","ꊩ","ꊪ","ꊫ","ꊬ","ꊭ","ꊮ","ꊯ","ꊰ","ꊱ","ꊲ","ꊳ","ꊴ","ꊵ","ꊶ","ꊷ","ꊸ","ꊹ","ꊺ","ꊻ","ꊼ","ꊽ","ꊾ","ꊿ","ꋀ","ꋁ","ꋂ","ꋃ","ꋄ","ꋅ","ꋆ","ꋇ","ꋈ","ꋉ","ꋊ","ꋋ","ꋌ","ꋍ","ꋎ","ꋏ","ꋐ","ꋑ","ꋒ","ꋓ","ꋔ","ꋕ","ꋖ","ꋗ","ꋘ","ꋙ","ꋚ","ꋛ","ꋜ","ꋝ","ꋞ","ꋟ","ꋠ","ꋡ","ꋢ","ꋣ","ꋤ","ꋥ","ꋦ","ꋧ","ꋨ","ꋩ","ꋪ","ꋫ","ꋬ","ꋭ","ꋮ","ꋯ","ꋰ","ꋱ","ꋲ","ꋳ","ꋴ","ꋵ","ꋶ","ꋷ","ꋸ","ꋹ","ꋺ","ꋻ","ꋼ","ꋽ","ꋾ","ꋿ","ꌀ","ꌁ","ꌂ","ꌃ","ꌄ","ꌅ","ꌆ","ꌇ","ꌈ","ꌉ","ꌊ","ꌋ","ꌌ","ꌍ","ꌎ","ꌏ","ꌐ","ꌑ","ꌒ","ꌓ","ꌔ","ꌕ","ꌖ","ꌗ","ꌘ","ꌙ","ꌚ","ꌛ","ꌜ","ꌝ","ꌞ","ꌟ","ꌠ","ꌡ","ꌢ","ꌣ","ꌤ","ꌥ","ꌦ","ꌧ","ꌨ","ꌩ","ꌪ","ꌫ","ꌬ","ꌭ","ꌮ","ꌯ","ꌰ","ꌱ","ꌲ","ꌳ","ꌴ","ꌵ","ꌶ","ꌷ","ꌸ","ꌹ","ꌺ","ꌻ","ꌼ","ꌽ","ꌾ","ꌿ","ꍀ","ꍁ","ꍂ","ꍃ","ꍄ","ꍅ","ꍆ","ꍇ","ꍈ","ꍉ","ꍊ","ꍋ","ꍌ","ꍍ","ꍎ","ꍏ","ꍐ","ꍑ","ꍒ","ꍓ","ꍔ","ꍕ","ꍖ","ꍗ","ꍘ","ꍙ","ꍚ","ꍛ","ꍜ","ꍝ","ꍞ","ꍟ","ꍠ","ꍡ","ꍢ","ꍣ","ꍤ","ꍥ","ꍦ","ꍧ","ꍨ","ꍩ","ꍪ","ꍫ","ꍬ","ꍭ","ꍮ","ꍯ","ꍰ","ꍱ","ꍲ","ꍳ","ꍴ","ꍵ","ꍶ","ꍷ","ꍸ","ꍹ","ꍺ","ꍻ","ꍼ","ꍽ","ꍾ","ꍿ","ꎀ","ꎁ","ꎂ","ꎃ","ꎄ","ꎅ","ꎆ","ꎇ","ꎈ","ꎉ","ꎊ","ꎋ","ꎌ","ꎍ","ꎎ","ꎏ","ꎐ","ꎑ","ꎒ","ꎓ","ꎔ","ꎕ","ꎖ","ꎗ","ꎘ","ꎙ","ꎚ","ꎛ","ꎜ","ꎝ","ꎞ","ꎟ","ꎠ","ꎡ","ꎢ","ꎣ","ꎤ","ꎥ","ꎦ","ꎧ","ꎨ","ꎩ","ꎪ","ꎫ","ꎬ","ꎭ","ꎮ","ꎯ","ꎰ","ꎱ","ꎲ","ꎳ","ꎴ","ꎵ","ꎶ","ꎷ","ꎸ","ꎹ","ꎺ","ꎻ","ꎼ","ꎽ","ꎾ","ꎿ","ꏀ","ꏁ","ꏂ","ꏃ","ꏄ","ꏅ","ꏆ","ꏇ","ꏈ","ꏉ","ꏊ","ꏋ","ꏌ","ꏍ","ꏎ","ꏏ","ꏐ","ꏑ","ꏒ","ꏓ","ꏔ","ꏕ","ꏖ","ꏗ","ꏘ","ꏙ","ꏚ","ꏛ","ꏜ","ꏝ","ꏞ","ꏟ","ꏠ","ꏡ","ꏢ","ꏣ","ꏤ","ꏥ","ꏦ","ꏧ","ꏨ","ꏩ","ꏪ","ꏫ","ꏬ","ꏭ","ꏮ","ꏯ","ꏰ","ꏱ","ꏲ","ꏳ","ꏴ","ꏵ","ꏶ","ꏷ","ꏸ","ꏹ","ꏺ","ꏻ","ꏼ","ꏽ","ꏾ","ꏿ","ꐀ","ꐁ","ꐂ","ꐃ","ꐄ","ꐅ","ꐆ","ꐇ","ꐈ","ꐉ","ꐊ","ꐋ","ꐌ","ꐍ","ꐎ","ꐏ","ꐐ","ꐑ","ꐒ","ꐓ","ꐔ","ꐕ","ꐖ","ꐗ","ꐘ","ꐙ","ꐚ","ꐛ","ꐜ","ꐝ","ꐞ","ꐟ","ꐠ","ꐡ","ꐢ","ꐣ","ꐤ","ꐥ","ꐦ","ꐧ","ꐨ","ꐩ","ꐪ","ꐫ","ꐬ","ꐭ","ꐮ","ꐯ","ꐰ","ꐱ","ꐲ","ꐳ","ꐴ","ꐵ","ꐶ","ꐷ","ꐸ","ꐹ","ꐺ","ꐻ","ꐼ","ꐽ","ꐾ","ꐿ","ꑀ","ꑁ","ꑂ","ꑃ","ꑄ","ꑅ","ꑆ","ꑇ","ꑈ","ꑉ","ꑊ","ꑋ","ꑌ","ꑍ","ꑎ","ꑏ","ꑐ","ꑑ","ꑒ","ꑓ","ꑔ","ꑕ","ꑖ","ꑗ","ꑘ","ꑙ","ꑚ","ꑛ","ꑜ","ꑝ","ꑞ","ꑟ","ꑠ","ꑡ","ꑢ","ꑣ","ꑤ","ꑥ","ꑦ","ꑧ","ꑨ","ꑩ","ꑪ","ꑫ","ꑬ","ꑭ","ꑮ","ꑯ","ꑰ","ꑱ","ꑲ","ꑳ","ꑴ","ꑵ","ꑶ","ꑷ","ꑸ","ꑹ","ꑺ","ꑻ","ꑼ","ꑽ","ꑾ","ꑿ","ꒀ","ꒁ","ꒂ","ꒃ","ꒄ","ꒅ","ꒆ","ꒇ","ꒈ","ꒉ","ꒊ","ꒋ","ꒌ"];
    #

    # FOLDER_PATH="/run/media/lasercat/writebuffer/tmp/synthyi_5k/apple/";
    # OUT_PATH="/run/media/lasercat/writebuffer/tmp/synthyi_5k/apple/res"+MODEL.replace(":","_")+"";
    script=None;
    cset=None;
    # FORCE_PMPT=None
    # #
    # O=deepseekocr(MODEL,cset,FORCE_PMPT);
    # O.run_sequential_ocr(FOLDER_PATH, OUT_PATH)


    MODEL = "gemma4:26b"
    FOLDER_PATH="/run/media/lasercat/writebuffer/tmp/synthyi_5k/apple/";
    OUT_PATH="/run/media/lasercat/writebuffer/tmp/synthyi_5k/apple/res"+MODEL.replace(":","_")+"";

    O=vlmocr(MODEL,cset,script,FORCE_PMPT,URL);
    O.run_sequential_ocr(FOLDER_PATH, OUT_PATH)