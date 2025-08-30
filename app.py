import streamlit as st
from PIL import Image
import whisper
import tempfile
import numpy as np

# CSSで余白とスクロール領域を調整
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    img {
        margin-bottom: -10px;
    }
    .stFileUploader {
        margin-top: -5px;
    }
    h2 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .scroll-box {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ccc;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# UIを左右に分割（右欄を広めに調整）
left, right = st.columns([2.5, 1.5])

with left:
    # ロゴとタイトル
    logo = Image.open("logo.png")
    st.image(logo, use_container_width=True)
    st.markdown("## 🎙️ 議事録自動生成ツール")
    st.caption("音声ファイルをアップするだけで、議事録がすぐに手に入ります。")

    # モード選択
    mode = st.radio(
        "🛠️ 出力形式を選んでください",
        ("① ✏️ 文字起こし全文", "② 💬 会話重視の議事録", "③ 📌 要点重視の議事録")
    )

    # 音声ファイルアップロード
    audio_file = st.file_uploader("🎧 音声ファイルをアップロードしてください", type=None)

    # 整形関数（①用）
    def clean_full_transcript(text):
        text = text.replace("。", "。\n")
        text = text.replace("A:", "\n**A:** ")
        text = text.replace("B:", "\n**B:** ")
        return f"""
【文字起こし全文】
-----------------------
{text}
-----------------------
※このテキストはダウンロード後、Copilotにアップロードして整形できます。
おすすめプロンプト：
「この文字起こしを読みやすく整えてください。話者ごとに分けて、改行を入れてください。」
"""

    # 音声分割関数（30秒単位）
    def split_audio(audio, chunk_duration_sec=30, sample_rate=16000):
        chunk_size = chunk_duration_sec * sample_rate
        return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    # ファイルがアップされたら処理開始
    if audio_file is not None:
        status_area = st.empty()
        status_area.info("⏳ 文字起こし中です。少々お待ちください...")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        try:
            model = whisper.load_model("base", device="cpu")
            audio = whisper.load_audio(tmp_path)
            chunks = split_audio(audio)

            progress_bar = st.progress(0)
            full_text = ""
            output_area = st.empty()

            for i, chunk in enumerate(chunks):
                result = model.transcribe(chunk, fp16=False, language="ja")
                full_text += result["text"] + "\n"
                output_area.markdown(f"""
                <div class="scroll-box">
                <pre>{full_text}</pre>
                </div>
                """, unsafe_allow_html=True)
                progress_bar.progress((i + 1) / len(chunks))

            status_area.success("✅ 文字起こしが完了しました")

            # モードごとの出力分岐
            if mode == "① ✏️ 文字起こし全文":
                output_text = clean_full_transcript(full_text)

            elif mode == "② 💬 会話重視の議事録":
                output_text = f"""
【議事録（会話重視）】
-----------------------
{full_text}
-----------------------
※このテキストはダウンロード後、Copilotにアップロードして整形できます。
おすすめプロンプト：
「この議事録を整えてください。話者ごとに分けて、会話の流れを残しつつ議事録風にしてください。」
"""

            elif mode == "③ 📌 要点重視の議事録":
                output_text = f"""
【議事録（要点重視）】
-----------------------
{full_text}
-----------------------
※このテキストはダウンロード後、Copilotにアップロードして整形できます。
おすすめプロンプト：
「この議事録から、議題・決定事項・アクションアイテムを抽出して、箇条書きでまとめてください。」
"""

            # ダウンロードボタン＋Copilot誘導ボタン（横並び）
            col1, col2 = st.columns([1, 1])

            with col1:
                st.download_button("📥 議事録をダウンロード", output_text, file_name="minutes.txt")

            with col2:
                st.markdown("""
                <a href="https://copilot.microsoft.com/" target="_blank">
                    <button style="width: 100%; padding: 0.5em; background-color: #0078D4; color: white; border: none; border-radius: 5px;">
                        Microsoft Copilotへ移動
                    </button>
                </a>
                """, unsafe_allow_html=True)

            # 整形済み出力（スクロール表示）
            st.markdown(f"""
            <div class="scroll-box">
            <pre>{output_text}</pre>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            status_area.error(f"⚠️ 文字起こしに失敗しました：{e}")

with right:
    st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)  # 高さ調整

    st.markdown("""
    <div style="max-height: 500px; overflow-y: auto; padding-right: 10px;">
    <h4>ℹ️ ご利用案内</h4>
    <ul style="line-height: 1.6;">
      <li><b>使い方</b><br>音声ファイルをアップ → モード選択 → 議事録表示 → Copilotにアップロードして整形</li>
      <li><b>対応形式</b><br>.m4a, .wav, .mp3 など</li>
      <li><b>保存について</b><br>音声・議事録は保存されません</li>
      <li><b>利用規約</b><br>無料で提供。商用利用OK。著作権音声は自己責任で。</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)