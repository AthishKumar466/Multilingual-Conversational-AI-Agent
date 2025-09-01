from transformers import pipeline

_translation_pipelines = {}

LANGUAGE_MODELS = {
    'en->hi': 'Helsinki-NLP/opus-mt-en-hi',
    'hi->en': 'Helsinki-NLP/opus-mt-hi-en',
    'en->ja': 'Helsinki-NLP/opus-mt-en-jap',
    'ja->en': 'Helsinki-NLP/opus-mt-ja-en',
}

def _get_pipeline(key: str):
    if key in _translation_pipelines:
        return _translation_pipelines[key]
    model_id = LANGUAGE_MODELS.get(key)
    if not model_id:
        raise ValueError(f"No translation model configured for {key}")
    pipe = pipeline('translation', model=model_id)
    _translation_pipelines[key] = pipe
    return pipe

def translate_text(text: str, src: str, tgt: str) -> str:
    key = f"{src}->{tgt}"
    pipe = _get_pipeline(key)
    out = pipe(text)
    return out[0]["translation_text"]