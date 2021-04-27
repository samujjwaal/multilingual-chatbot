# import libraries
import wget
import os
import fasttext

fasttext.FastText.eprint = lambda x: None


url = [
    # 917kB, compressed version of the model
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
    # 126MB, faster and slightly more accurate
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
]

model_names = list(map(lambda u: u.split("/")[-1], url))


# check whether a model is downloaded locally
def check_model_exist(option, path="models"):
    os.makedirs(path, exist_ok=True)
    if model_names[option] not in os.listdir("models"):
        return False
    else:
        return True


# to download fasttext model locally
def download_model(option=0, path="models"):
    if not check_model_exist(option):
        wget.download(url[option], out=path)


# to load fasttext model from directory
def load_pretrained_model(option=0, path="models"):
    download_model(option)
    PRETRAINED_MODEL_PATH = f"{path}/{model_names[option]}"
    lang_model = fasttext.load_model(PRETRAINED_MODEL_PATH)
    return lang_model


# predict language of input text using fasttext
def predict_lang(prompt, option=0):
    model = load_pretrained_model(option)
    lang = model.predict(prompt)[0][0].split("__")[2]
    return lang
