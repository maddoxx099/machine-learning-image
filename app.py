from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")


def translate_text(text):
    inputs = tokenizer(">>en<<" + text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translated_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        text_to_translate = request.form['text_to_translate']
        translated_text = translate_text(text_to_translate)
        return render_template('index.html', text_to_translate=text_to_translate, translated_text=translated_text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
