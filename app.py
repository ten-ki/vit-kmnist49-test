# streamlit_app.py

import streamlit as st
from transformers import pipeline
from PIL import Image
import io
import time
import torch

# 1. モデルのパス設定
# GitHubリポジトリのルートにあるモデルフォルダ名を指定
MODEL_NAME = "KARAGE-KUN/vit-kuzushiji49-final-tpu-xla-speed" 

# 2. モデルの初期化（@st.cache_resource でキャッシュし、ロードを高速化）
# device=-1 はCPUを意味します（Streamlit CloudはGPUがないため）
@st.cache_resource
def load_model():
    # 推論のみなので、フィーチャーエクストラクターとモデルをパイプラインでロード
    try:
        classifier = pipeline(
            "image-classification", 
            model=MODEL_NAME,
            device=-1 # CPUでの推論を指定
        )
        return classifier
    except Exception as e:
        st.error(f"モデルのロードに失敗しました。ファイルを確認してください: {e}")
        return None

# 3. アプリケーションのメインロジック
st.title("崩し字 49 分類アプリ")
st.markdown("Vision Transformer (ViT) で学習したモデルを使った崩し字の分類デモです。")

classifier = load_model()

if classifier:
    uploaded_file = st.file_uploader("崩し字の画像をアップロードしてください (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 画像をPIL Imageとして読み込む
        image = Image.open(uploaded_file).convert('RGB')
        
        # 画面に画像を表示
        st.image(image, caption='アップロードされた画像', use_column_width=True)

        st.subheader("分類結果")
        
        start_time = time.time()
        
        # 推論の実行
        # 画像をパイプラインに渡すだけで、自動で前処理（リサイズ/正規化）が行われます
        with st.spinner('分類中...'):
            results = classifier(image)

        end_time = time.time()

        # 結果の表示
        if results:
            # 確率の高い順にソート（パイプラインの仕様上、既にソートされていることが多い）
            st.success(f"推定される文字: **{results[0]['label']}** (確率: {results[0]['score']:.4f})")
            st.write(f"推論時間: {end_time - start_time:.3f}秒 (CPU)")
            
            # 詳細な結果をテーブルで表示
            st.dataframe([
                {"ラベル (漢字)": res['label'], "確率": f"{res['score']:.4f}"} 
                for res in results[:5]
            ])
        else:
            st.warning("分類結果が得られませんでした。")