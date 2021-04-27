# import libraries
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# check whether a model is downloaded locally
def check_model_exist(model_name, path="models"):
    model_path = f"./{path}/{model_name}/"
    os.makedirs(model_path, exist_ok=True)
    if len(os.listdir(model_path)) == 0:
        return False
    else:
        return True


# to download mbart-50 model locally
def mbart_load_model(path="models"):
    # load tokenizer from transformers
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    if check_model_exist("mbart-large-50"):
        # load model from local directory
        model = MBartForConditionalGeneration.from_pretrained(
            f"./{path}/mbart-large-50/"
        )
    else:
        # load model from transformers
        model = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        # save model to local directory
        model.save_pretrained(f"./{path}/mbart-large-50/")
    return model, tokenizer


# translate text using mbart-50 model
def mbart_translate(text, model, tokenizer, source_lang, target_lang):
    # set source language
    tokenizer.src_lang = source_lang
    # tokenize input text
    encoded_hi = tokenizer(text, return_tensors="pt")
    # generate translated text vector
    generated_tokens = model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
    )
    # decode translated text vector to text
    translation = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]
    return translation


# to download opus-mt model locally
def opus_load_model(model_variant, path="models"):
    model_name = f"Helsinki-NLP/{model_variant}"
    # load tokenizer from transformers
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if check_model_exist(model_variant):
        # load model from local directory
        model = AutoModelForSeq2SeqLM.from_pretrained(
            f"./{path}/{model_variant}/"
        )
    else:
        # load model from transformers
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # save model to local directory
        model.save_pretrained(f"./{path}/{model_variant}/")

    return model, tokenizer


# translate text using opus-mt model
def opus_translate(text, model, tokenizer):
    # tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    # generate translated text vector
    outputs = model.generate(**inputs)
    # decode translated text vector to text
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded
