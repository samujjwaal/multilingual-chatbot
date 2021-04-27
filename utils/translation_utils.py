# import libraries
from utils.transformer_utils import mbart_load_model, opus_load_model
from googletrans import Translator

# Initialize googletrans translator instance
translator = Translator()

# options for language detection models
lang_detect_model = ["googletrans", "fasttext"]

# flag to check if mbart model is loaded
mbart_flag = False

mbart_model = None
mbart_tokenizer = None

# flag to check if opus model is loaded
opus_flag = False

opus_model_1 = None
opus_model_2 = None
opus_tokenizer_1 = None
opus_tokenizer_2 = None

# translate 'text' to 'dest' language
def translate(text, dest):
    return translator.translate(text, dest).text


# detect language of 'text
def detect_lang(text):
    lang = translator.detect(text).lang
    if type(lang) == list:
        return lang[0]
    else:
        return lang


# for selecting translation model
def select_trans_model(option="gt", variant="ROMANCE"):
    global mbart_flag, opus_flag
    # for googletrans
    if option == "gt":
        pass
    else:
        # for mbart
        if option == "mbart-50":
            if not mbart_flag:
                global mbart_model, mbart_tokenizer
                # load model and tokenizer
                mbart_model, mbart_tokenizer = mbart_load_model()
                mbart_flag = True
            else:
                pass
        # for opus
        if option == "opus":
            if not opus_flag:
                global opus_model_1, opus_tokenizer_1
                global opus_model_2, opus_tokenizer_2
                # load model and tokenizer
                name_1 = f"opus-mt-en-{variant}"
                opus_model_1, opus_tokenizer_1 = opus_load_model(name_1)
                name_2 = f"opus-mt-{variant}-en"
                opus_model_2, opus_tokenizer_2 = opus_load_model(name_2)
                opus_flag = True
            else:
                pass
