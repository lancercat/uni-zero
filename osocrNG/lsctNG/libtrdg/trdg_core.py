import random as rnd
import re

from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
from PIL import Image, ImageColor, ImageFont, ImageDraw
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize

# mask gen for trdg.
# blend, distort, etc comes later.
def getpaddedsize(text_width,text_height):
    return int(text_width*1.1),int(text_height*1.1);

def generate(
    text, font, font_size, orientation, space_width, character_spacing, fit, word_split
):
    if orientation == 0:
        return _generate_horizontal_text_mask(
            text, font, font_size, space_width, character_spacing, fit, word_split
        )
    elif orientation == 1:
        return _generate_vertical_text_mask(
            text, font, font_size, space_width, character_spacing, fit
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _generate_horizontal_text_mask(
    text, font, font_size, space_width, character_spacing, fit, word_split
):
    image_font = ImageFont.truetype(font=font, size=font_size)
    try:
        space_width = int(image_font.getbbox(" ")[2] * space_width)
    except:
        space_width=0;
        
    if word_split:
        splitted_text = []
        for w in text.split(' '):
            splitted_text.append(w)
            splitted_text.append(' ')
        splitted_text.pop()
    else:
        splitted_text = tokenize(text);

    piece_widths = [image_font.getbbox(p)[2] if p != " " else space_width for p in splitted_text]
    text_width = sum(piece_widths)
    if not word_split:
        text_width += character_spacing * (len(text) - 1)

    text_height = max([image_font.getbbox(p)[3] for p in splitted_text])

    txt_mask = Image.new("RGB", getpaddedsize(text_width, text_height), (0, 0, 0))

    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"
    amsks=[];
    for i, p in enumerate(splitted_text):
        ctxt_mask = Image.new("RGB", getpaddedsize(text_width, text_height), (0, 0, 0))
        ctxt_mask_draw = ImageDraw.Draw(ctxt_mask)
        ctxt_mask_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=(255, 255, 255),
            font=image_font,
        )
        txt_mask_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=(255,255,255),
            font=image_font,
        )
        amsks.append(ctxt_mask);

    if fit:
        b = txt_mask.getbbox();
        return txt_mask.crop(b), [cm.crop(b) for cm in amsks]
    else:
        return txt_mask, amsks


def _generate_vertical_text_mask(
    text, font, font_size, space_width, character_spacing, fit
):
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_height = int(image_font.getbbox(" ")[3] * space_width)

    char_heights = [
        image_font.getbbox(c)[3] if c != " " else space_height for c in tokenize(text)
    ]
    text_width = max([image_font.getbbox(c)[2] for c in tokenize(text)])
    text_height = sum(char_heights) + character_spacing * len(tokenize(text))
    amsks=[];
    txt_mask = Image.new("RGB", getpaddedsize(text_width, text_height), (0, 0, 0))

    txt_mask_draw = ImageDraw.Draw(txt_mask);
    for i, c in enumerate(tokenize(text)):
        ctxt_mask = Image.new("RGB", getpaddedsize(text_width, text_height), (0, 0, 0))
        ctxt_mask_draw = ImageDraw.Draw(ctxt_mask)
        ctxt_mask_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=(255,255,255),
            font=image_font,
        )
        txt_mask_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=(255,255,255),
            font=image_font,
        )
        amsks.append(ctxt_mask);
    if fit:
        b=txt_mask.getbbox();
        return  txt_mask.crop(b), [cm.crop(b) for cm in amsks]
    else:
        return txt_mask, amsks
