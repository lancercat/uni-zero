import regex
def tokenize(s):
    return regex.findall(r'\X', s, regex.U);
