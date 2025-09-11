import os
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI, RateLimitError # Import RateLimitError
from dotenv import load_dotenv
from newspaper import Article
import nltk
import time

# newspaper3kが使用するnltkのデータをダウンロード（初回のみ）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # このログはサーバー起動時に一度だけ表示される可能性があります
    print("Downloading 'punkt' for nltk...")
    nltk.download('punkt')
    print("'punkt' download complete.")

# .envファイルから環境変数を読み込む
load_dotenv()

# Flaskアプリケーションのインスタンスを作成
# static_folderのデフォルトは 'static' なので、
# このファイルと同じ階層に 'static' フォルダがあれば自動的にそこが使われます。
app = Flask(__name__)

# 開発モード時に静的ファイルのキャッシュを無効にする
if app.debug:
    @app.after_request
    def add_header(response):
        # /static/ 以下のファイルに対するリクエストの場合
        if request.endpoint == 'static':
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache' # HTTP/1.0 backward compatibility
            response.headers['Expires'] = '0' # Proxies
        return response


# OpenRouter APIキーと関連情報を環境変数から取得
# このキーはサーバーサイドで安全に管理してください
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:5000") # Default if not set
APP_NAME = os.getenv("YOUR_APP_NAME", "FlaskVueApp") # Default if not set

# URL:/ に対して、static/index.htmlを表示して
    # クライアントサイドのVue.jsアプリケーションをホストする
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')
    
# URL:/send_api に対するメソッドを定義
@app.route('/send_api', methods=['POST'])
def send_api():
    if not OPENROUTER_API_KEY:
        app.logger.error("OpenRouter API key not configured.")
        return jsonify({"error": "OpenRouter API key is not configured on the server."}), 500

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={ # Recommended by OpenRouter
            "HTTP-Referer": SITE_URL,
            "X-Title": APP_NAME,
        }
    )
    
    # POSTリクエストからJSONデータを取得
    data = request.get_json()

    # デバッグ用のモック応答機能
    # テキスト入力で "DEBUG_MOCK" と入力すると、APIを消費せずに偽の応答を返す
    if data and data.get('text') == 'DEBUG_MOCK':
        app.logger.info("Returning a mock response for debugging.")
        time.sleep(1) # ネットワーク遅延をシミュレート
        return jsonify({
            "message": "This is a mock response.",
            "processed_text": "これはモック（偽の）応答です。API制限を消費せずに開発を続けるために使用します。UIの動作確認にご利用ください。"
        })


    received_text = ""

    # 'url'フィールドが存在し、中身があるかチェック
    if data and 'url' in data and data['url'].strip():
        url = data['url'].strip()
        app.logger.info(f"Received URL for summarization: {url}")
        try:
            # newspaper3kを使用して記事の本文を抽出
            article = Article(url)
            article.download()
            article.parse()
            received_text = article.text

            if not received_text.strip():
                app.logger.warning(f"Could not extract text from URL using newspaper3k: {url}")
                return jsonify({"error": "URLから本文を抽出できませんでした。記事形式のページでない可能性があります。"}), 400

        except Exception as e:
            app.logger.error(f"URLの処理中に予期せぬエラーが発生しました (newspaper3k): {url}, Error: {e}")
            return jsonify({"error": "URLの処理中に予期せぬエラーが発生しました。"}), 500

    # 'url'がない場合、'text'フィールドが存在するかチェック
    elif data and 'text' in data and data['text'].strip():
        received_text = data['text']
    
    # どちらも無い場合はエラー
    else:
        app.logger.error("Request JSON is missing 'text' or 'url' field, or they are empty.")
        return jsonify({"error": "要約するテキストまたはURLを入力してください。"}), 400

    # contextがあればsystemプロンプトに設定、なければデフォルト値
    system_prompt = "あなたは役立つアシスタントです。" # デフォルトのシステムプロンプト
    if 'context' in data and data['context'] and data['context'].strip():
        system_prompt = data['context'].strip()
        app.logger.info(f"Using custom system prompt from context: {system_prompt}")
    else:
        app.logger.info(f"Using default system prompt: {system_prompt}")

    try:
        # OpenRouter APIを呼び出し
        # モデル名はOpenRouterで利用可能なモデルを指定してください。
        # 例: "mistralai/mistral-7b-instruct", "google/gemini-pro", "openai/gpt-3.5-turbo"
        # 詳細はOpenRouterのドキュメントを参照してください。

        # Gemma 3モデルはsystemプロンプトをサポートしていないため、userメッセージに指示をまとめます。
        full_prompt = f"{system_prompt}\n\n---\n\n{received_text}"

        chat_completion = client.chat.completions.create(
            messages=[ # type: ignore
                {"role": "user", "content": full_prompt}
            ], # type: ignore
            model="google/gemma-3-27b-it:free", 
        )
        
        # APIからのレスポンスを取得
        if chat_completion.choices and chat_completion.choices[0].message:
            processed_text = chat_completion.choices[0].message.content
        else:
            processed_text = "AIから有効な応答がありませんでした。"
            
        return jsonify({"message": "AIによってデータが処理されました。", "processed_text": processed_text})

    except RateLimitError as e:
        app.logger.warning(f"OpenRouter API rate limit exceeded: {e}")
        return jsonify({"error": "APIの利用回数制限に達しました。時間をおいてから再度お試しください。"}), 429

    except Exception as e:
        app.logger.error(f"OpenRouter API call failed: {e}")
        # クライアントには具体的なエラー詳細を返しすぎないように注意
        return jsonify({"error": f"AIサービスとの通信中にエラーが発生しました。"}), 500

# スクリプトが直接実行された場合にのみ開発サーバーを起動
if __name__ == '__main__':
    if not OPENROUTER_API_KEY:
        print("警告: 環境変数 OPENROUTER_API_KEY が設定されていません。API呼び出しは失敗します。")
    app.run(debug=True, host='0.0.0.0', port=5000)