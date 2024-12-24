import gradio as gr
from PIL import Image
from pathlib import Path
import os
from uuid import uuid4
import glob
import requests
from io import BytesIO
from pdf2image import convert_from_path

# 設定フラグ
IS_DEV = True  # デバッグモード
IS_DEV_INPUT = True  # 入力のテストモード

def clear_directory(directory):
    """指定されたディレクトリを削除して再作成する"""
    if os.path.exists(directory):
        os.system(f"rm -rf {directory}")
    Path(directory).mkdir(parents=True)

# def save_image(file_path, img_dir):
def save_image(im, img_dir):
    """画像を保存する"""
    # im = Image.open(file_path)
    im.save(f'{img_dir}/0001.jpg')

def save_pdf_as_images(file_path, img_dir):
    """PDFを画像に変換して保存する"""
    pages = convert_from_path(file_path, fmt="jpg")
    for i, page in enumerate(pages, start=1):
        page.save(f"{img_dir}/{i:04d}.jpg", "JPEG")

def download_images_from_manifest(url_input, img_dir):
    """IIIFマニフェストのURLから画像をダウンロードして保存する"""
    manifest = requests.get(url_input).json()
    canvases = manifest["sequences"][0]["canvases"]
    for i, canvas in enumerate(canvases):
        image_url = canvas["images"][0]["resource"]["@id"]
        image = Image.open(BytesIO(requests.get(image_url).content))
        image.save(f"{img_dir}/{str(i+1).zfill(4)}.jpg")

def create_input_dir(dir_name, input_type, im=None, file_path=None, url_input=None):
    """
    入力データを処理して画像を保存するディレクトリを作成
    """
    img_dir = f"{dir_name}/img"
    clear_directory(img_dir)

    if input_type == "image":
        save_image(im, img_dir)
    elif input_type == "pdf":
        save_pdf_as_images(file_path, img_dir)
    elif input_type == "url":
        download_images_from_manifest(url_input, img_dir)

def read_output_text(dir_name, task_id):
    """OCRの出力テキストを読み込む"""
    txt_array = []
    files = glob.glob(f"{dir_name}_output/{task_id}/txt/*_main.txt")
    files.sort()
    for file in files:
        with open(file) as f:
            txt_array.append(f.read())
    return "\n=== pb ===\n".join(txt_array)

def read_output_xml(dir_name, task_id):
    """OCRの出力XMLを読み込む"""
    xml_file = f"{dir_name}_output/{task_id}/xml/{task_id}.sorted.xml"
    with open(xml_file) as f:
        return f.read()

def inference(input_type, im=None, file_path=None, url_input=None):
    """
    推論処理のメイン関数
    """
    # 入力ディレクトリの作成
    if IS_DEV_INPUT:
        dir_name = f"data/{input_type}"
        create_input_dir(dir_name, input_type, im, file_path, url_input)

    if IS_DEV:
        task_id = "koten"
        dir_name = f"data/{task_id}"
    else:
        task_id = str(uuid4())
        dir_name = f"data/{task_id}"

        create_input_dir(dir_name, input_type, im, file_path, url_input)

        # OCR処理の実行
        os.system(f"python main.py infer {dir_name} {dir_name}_output -s s -x -p 1..3")

    # OCR結果の読み取り
    text_output = read_output_text(dir_name, task_id)
    xml_output = read_output_xml(dir_name, task_id)

    # 一時ディレクトリの削除
    if not IS_DEV:
        os.system(f"rm -rf {dir_name}*")

    return [text_output, xml_output]

# Gradioインターフェースの設定
title = "NDL OCR ver.2.1"
description = "Gradio demo for NDL OCR. NDL OCR is a text recognition (OCR) Program."
article = "<p style='text-align: center'><a href='https://github.com/ndl-lab' target='_blank'>NDL Lab</a> | <a href='https://github.com/ndl-lab/ndlocr_cli' target='_blank'>NDL OCR Repo</a></p>"

# 入力セレクターの作成
input_selector = gr.Radio(
    choices=["image", "pdf", "url"],
    label="Input Type",
    value="image"
)

# Gradioインターフェースの定義
demo = gr.Interface(
    fn=inference,
    inputs=[
        input_selector,
        gr.Image(label="Image (if Image input)", type="pil"),
        gr.File(label="File (if PDF input)"),
        gr.Text(label="URL (if URL input)", lines=1)
    ],
    outputs=[
        gr.Text(label="Text"),
        gr.Text(label="XML")
    ],
    title=title,
    description=description,
    article=article,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")