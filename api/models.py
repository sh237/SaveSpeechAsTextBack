from django.db import models
from google.cloud import speech
from google.cloud import storage
from pydub import AudioSegment
import re
import os
import unicodedata
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

CODE_PATTERN = re.compile(r"```.*?```", re.MULTILINE | re.DOTALL)
LINK_PATTERN = re.compile(r"!?\[([^\]\)]+)\]\([^\)]+\)")
IMG_PATTERN = re.compile(r"<img[^>]*>")
URL_PATTERN = re.compile(r"(http|ftp)s?://[^\s]+")
NEWLINES_PATTERN = re.compile(r"(\s*\n\s*)+")

# 環境変数の設定
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/nagashimashunya/Django/SpeechSaveAsTextBack/SaveSpeechAsTextBack/api/stone-chariot-371804-8d0f0ecfcb5d.json'
GCS_BASE = "gs://ssat_bucket/"


class AudioModel(models.Model):
    file = models.FileField(blank=False, null=False, upload_to='audio/')

    def __str__(self):
        return self.file.name

    def predicted_text(self):
        client = storage.Client()  # GCSのクライアントを作成
        bucket = client.get_bucket('ssat_bucket')  # バケットを取得
        transcribe_file = self.file.name  # ファイル名を取得
        sound = AudioSegment.from_wav(self.file.path)  # 音声ファイルを読み込み
        sound.set_channels(1)  # チャンネル数を1に変換
        sound.set_frame_rate(44100)  # サンプリングレートを44100に変換
        sound.export(self.file.path, format="wav")  # 変換した音声ファイルを上書き保存

        length = sound.duration_seconds  # 音声ファイルの長さを取得
        length += 1  # 音声ファイルの長さに1秒を足す
        if length > 60:
            blob = bucket.blob(transcribe_file)  # バケットにファイルをアップロード
            blob.upload_from_filename(filename=self.file.path)  # ファイルをアップロード
            result = self.transcribe_model_selection_gcs(
                gcs_uri=GCS_BASE + transcribe_file,
                length=length)  # 1分以上の音声ファイルをテキストに変換する
        else:
            result = self.transcribe_model_selection(
                length=length)  # 1分未満の音声ファイルをテキストに変換する
        return result  # テキストを返す

    def transcribe_model_selection(self, length):
        """1分未満のファイルをテキストに変換する。
        この場合は、音声ファイルをローカルに保存してから変換する。"""
        client = speech.SpeechClient()  # Speech-to-Textのクライアントを作成

        # 音声ファイルを読み込み
        with open(self.file.path, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)  # 音声ファイルを読み込み

        # 音声ファイルの設定
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="ja-JP",
            audio_channel_count=1,
        )

        # 音声ファイルをテキストに変換
        response = client.recognize(config=config, audio=audio)

        # テキストを返す
        results = []
        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]
            results.append(alternative.transcript)
        return results

    def transcribe_model_selection_gcs(self, gcs_uri, length):
        """1分以上の音声ファイルをテキストに変換する。
        この場合は、音声ファイルをGCSに保存してから変換する。"""

        client = speech.SpeechClient()  # Speech-to-Textのクライアントを作成

        audio = speech.RecognitionAudio(uri=gcs_uri)  # 音声ファイルを読み込み

        # 音声ファイルの設定
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            sample_rate_hertz=44100,
            language_code="ja-JP",
            model="default",
        )

        # 音声ファイルをテキストに変換
        operation = client.long_running_recognize(config=config, audio=audio)

        # テキストを返す
        results = []
        results_str = ""
        response = operation.result(timeout=length*2)
        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]
            print("lens: {}".format(len(result.alternatives)))
            for alternative in result.alternatives:
                print("Transcript: {}", alternative.transcript)
            print("-" * 20)
            print("First alternative of result {}".format(i))
            print(u"Transcript: {}".format(alternative.transcript))
            results_str += alternative.transcript
        results.append(results_str)
        return results


class TextModel(models.Model):
    text = models.TextField()

    MODEL_PATH = 'api/ml_models/'
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, is_fast=True)
    trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    def predict(self):
        USE_GPU = torch.cuda.is_available()
        if USE_GPU:
            self.trained_model.cuda()

        self.preprocess_questionnaire_body(self.text)
        MAX_SOURCE_LENGTH = 512   # 入力文の最大トークン数
        MAX_TARGET_LENGTH = 126   # 生成文の最大トークン数

        # モデルを推論モードに設定
        self.trained_model.eval()

        # 入力文の前処理を行う
        inputs = [self.preprocess_questionnaire_body(self.text)]
        batch = self.tokenizer.batch_encode_plus(
            inputs, max_length=MAX_SOURCE_LENGTH, truncation=True,
            padding="longest", return_tensors="pt")

        input_ids = batch['input_ids']
        input_mask = batch['attention_mask']
        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        # モデルに入力し、生成文を取得
        outputs = self.trained_model.generate(
            input_ids=input_ids, attention_mask=input_mask,
            max_length=MAX_TARGET_LENGTH,
            return_dict_in_generate=True, output_scores=True,
            temperature=1.0,            # 生成にランダム性を入れる温度パラメータ
            num_beams=10,               # ビームサーチの探索幅
            diversity_penalty=1.0,      # 生成結果の多様性を生み出すためのペナルティ
            num_beam_groups=10,         # ビームサーチのグループ数
            num_return_sequences=5,    # 生成する文の数
            repetition_penalty=1.5,     # 同じ文の繰り返し（モード崩壊）へのペナルティ
        )

        # 生成された文をデコードする
        generated_sentences = [
            self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False) for ids in outputs.sequences]

        # 生成された文を表示する
        summary = []
        for i, sentence in enumerate(generated_sentences):
            if i == 0:
                summary.append(self.preprocess_questionnaire_body(sentence))
        print("summary", summary)
        print("summary", summary[0])
        print("summary", summary[0][12:])
        return self.text, summary[0][12:]

    # ユニコード正規化
    def unicode_normalize(self, cls, s):
        pt = re.compile('([{}]+)'.format(cls))

        def norm(c):
            return unicodedata.normalize('NFKC', c) if pt.match(c) else c

        s = ''.join(norm(x) for x in re.split(pt, s))
        s = re.sub('-', '-', s)
        return s

    # 余分な空白を削除する処理
    def remove_extra_spaces(self, s):
        s = re.sub('[  ]+', ' ', s)
        blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                          '\u3040-\u309F',  # HIRAGANA
                          '\u30A0-\u30FF',  # KATAKANA
                          '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                          '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                          ))
        basic_latin = '\u0000-\u007F'

        def remove_space_between(cls1, cls2, s):
            p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
            while p.search(s):
                s = p.sub(r'\1\2', s)
            return s

        s = remove_space_between(blocks, blocks, s)
        s = remove_space_between(blocks, basic_latin, s)
        s = remove_space_between(basic_latin, blocks, s)
        return s

    # NEologdを用いて正規化する処理
    def normalize_neologd(self, s):
        s = s.strip()
        s = self.unicode_normalize('0-9A-Za-ｚ｡-ﾟ', s)

        def maketrans(f, t):
            return {ord(x): ord(y) for x, y in zip(f, t)}

        s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
        s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
        # normalize tildes (modified by Isao Sonobe)
        s = re.sub('[~∼∾〜〰～]+', '〜', s)
        s = s.translate(
            maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                      '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

        s = self.remove_extra_spaces(s)
        s = self.unicode_normalize(
            '！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
        s = re.sub('[’]', '\'', s)
        s = re.sub('[”]', '"', s)
        return s

    # 前処理
    def preprocess_questionnaire_body(self, markdown_text):
        markdown_text = CODE_PATTERN.sub(r"", markdown_text)
        markdown_text = LINK_PATTERN.sub(r"\1", markdown_text)
        markdown_text = IMG_PATTERN.sub(r"", markdown_text)
        markdown_text = URL_PATTERN.sub(r"", markdown_text)
        markdown_text = NEWLINES_PATTERN.sub(r"\n", markdown_text)
        markdown_text = markdown_text.replace("`", "")
        markdown_text = markdown_text.replace("\t", " ")
        markdown_text = self.normalize_neologd(markdown_text).lower()
        markdown_text = markdown_text.replace("\n", " ")
        markdown_text = markdown_text[:4000]
        return "body: " + markdown_text
