from transformers import PegasusTokenizer

tok = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
s = "adk adk adkadkadkadk adk"
x = tok.encode(s)
tok.decode(x)
