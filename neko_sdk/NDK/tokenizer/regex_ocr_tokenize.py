import regex
def tokenize(s,pattern=r'\X'):
    if(pattern==r'\X'):
        return regex.findall(pattern, s, regex.U);
    else:
        return s.split(pattern);