tUNK="[UNK]";
tUNKREP="⑨";

tEOS="[s]";
tMARGIN="[PAD]"
tBLANC="[-]";
tBOS="[STA]";

hUNK=" ";
tDC="❾";
ID_DC=-1; # unknown label id

tSPLIT_OCR=r"\X"; # simply a meme to keep the word unsplit, prevent making dog `d` 'o' 'g';
tSPLIT_ITEM=r"^.*$"; # just everything, duh

# this will not be honored by anything but tokenizer beyond 320.

# well nobody will use this string in data, so we will be safe.
tSPLIT_PANOPTIC="[NEP_sep_NEP]";

