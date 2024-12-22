import gradio as gr
from PIL import Image, ImageDraw
from pathlib import Path
import os
from uuid import uuid4
import xml.etree.ElementTree as ET
import glob
import json

IS_DEV = False

def inference(im):
   
    if IS_DEV:
       task_id = "koten"
       dir_name = f"data/{task_id}"
    else:
        task_id = uuid4()
        dir_name = f"data/{task_id}"
        Path(f'{dir_name}/img').mkdir(parents=True)
        im.save(f'{dir_name}/img/image.jpg')
        os.system(f'python main.py infer {dir_name} {dir_name}_output -s s -x -p 1..3')

    txt_array = [] 

    files = glob.glob(f'{dir_name}_output/{task_id}/txt/*_main.txt')
    files.sort()
    for file in files:
       with open(file) as f:
           txt_array.append(f.read())

    xml_file = f"{dir_name}_output/{task_id}/xml/{task_id}.sorted.xml"

    with open(xml_file) as f:
        xml_text = f.read()

    if not IS_DEV:
        os.system(f"rm -rf {dir_name}*")

    return ["\n=== pb ===\n".join(txt_array), xml_text]

title = "NDL OCR ver.2.1"
description = "Gradio demo for NDL OCR. NDL OCR is a text recognition (OCR) Program."
article = "<p style='text-align: center'><a href='https://github.com/ndl-lab' target='_blank'>NDL Lab</a> | <a href='https://github.com/ndl-lab/ndlocr_cli' target='_blank'>NDL OCR Repo</a></p>"

demo = gr.Interface(
    fn=inference,
    inputs=
    [
        gr.Image(label='image', type='pil')
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