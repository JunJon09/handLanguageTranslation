import os
import openai
from dotenv import load_dotenv
from pathlib import Path
import os

def word_translate(text):
    # 環境変数からAPIキーを取得

    project_root = Path(__file__).resolve().parent
    print(project_root)
    # .env ファイルのパスを設定
    env_path = project_root / '.env'


    # .env ファイルを読み込む
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv('OPENAI_API_KEY')
    # print(api_key)
    # APIキーを設定
    openai.api_key = api_key

    # ChatGPTに送信するプロンプト
    prompt = text
    a = (
    "リストの中を全て使用して一文を作成してください。条件は以下の通りです。"
    "・リスト以外の文は助詞しか使用してはいけません。"
    "・日本語の並び順はリストの並び順と同じである必要があります。"
    "以上の条件に従って、一文を作成してください。"
    )



    # APIリクエストの送信
    response = openai.ChatCompletion.create(
        model="gpt-4",  # モデルを指定。必要に応じて他のモデルに変更可能。
        messages=[
            {"role": "system", "content": a},
            {"role": "user", "content": text}
        ],
        max_tokens=150,  # 応答の最大トークン数
        temperature=0.7,  # クリエイティブ度合い。0.0～1.0の範囲
    )

    # 応答の取得
    reply = response['choices'][0]['message']['content']

    return reply

