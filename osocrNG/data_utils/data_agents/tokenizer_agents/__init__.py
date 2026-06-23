
# why bother doing this explicitly?
# well if you have weird tokenizer like radical tokenizer or what not you can exploit it here.
# and also, injected sp tokens in a string will mess up the tokenizer.
# these forces us to explictly convert a string to a list of characters to avoid tripping our selves
# "meow" -> [m,e,o,w]
# "meow" -> [m,e,o,w,<s>]

# "cat" -> [head, body, paw,...]
# etc.