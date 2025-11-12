import tiktoken

class Tokenizer:
    def __init__(self, name='gpt2'):
        self.enc = tiktoken.get_encoding(name)
        self.enc_id = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids)