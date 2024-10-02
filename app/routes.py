from io import BytesIO
from PIL import Image
from flask import Blueprint, render_template, request, url_for
import openai
import cv2
import numpy as np
import os
import logging
import requests

bp = Blueprint("main", __name__)

# Configuração básica de logging
logging.basicConfig(level=logging.DEBUG)

# Substitua 'your-api-key' pela sua chave da OpenAI
openai.api_key = "CHAVE-GPT"


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/generate", methods=["POST"])
def generate():
    prompt = request.form["prompt"]
    steps = 15  # Número de frames para o movimento
    logging.debug(f"Prompt recebido: {prompt}")

    # Gerar prompts variáveis para simular movimento
    moving_prompts = generate_moving_prompts(prompt, steps)

    image_urls = []
    try:
        for p in moving_prompts:
            response = openai.Image.create(
                prompt=p,
                n=1,  # Uma imagem por prompt
                size="600x800",  # Tamanho da imagem
            )
            image_urls.append(response["data"][0]["url"])
            logging.debug(
                f"Imagem gerada para o prompt '{p}': {response['data'][0]['url']}"
            )

        frames = create_frames_from_images(image_urls)
        if frames:
            video_path = os.path.join("app", "static", "animated_video.mp4")
            save_video(frames, video_path)
        else:
            logging.error("Nenhum frame gerado para salvar o vídeo.")
            return "Erro ao gerar vídeo", 500

    except Exception as e:
        logging.error(f"Erro ao gerar imagens com DALL-E: {e}")
        return "Erro ao gerar vídeo", 500

    logging.info(f"Vídeo salvo em: {video_path}")
    return render_template(
        "video.html", video_path=url_for("static", filename="animated_video.mp4")
    )


def create_frames_from_images(image_urls):
    frames = []

    for img_url in image_urls:
        # Baixar a imagem
        response = requests.get(img_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            # Redimensionar a imagem para o tamanho desejado
            image = image.resize((640, 480))
            frame = np.array(image)  # Converter para array NumPy
            frames.append(frame)
        else:
            logging.error(f"Erro ao baixar a imagem: {img_url}")

    return frames


def generate_moving_prompts(base_prompt, steps):
    prompts = []
    for i in range(steps):
        # Adicione variações pequenas de movimento no prompt
        prompt = f"{base_prompt}, slight movement, frame {i + 1}, gradual shift, subtle change"
        prompts.append(prompt)
    return prompts


def save_video(video_content, video_path):
    logging.debug("Iniciando o processo de salvamento do vídeo.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20.0

    if isinstance(video_content[0], np.ndarray):
        frame_height, frame_width = video_content[0].shape[:2]
    else:
        logging.error("O primeiro quadro não é um numpy array.")
        return

    logging.debug(f"Salvando vídeo em: {video_path}")

    # Criação do diretório, se não existir
    try:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        logging.debug(f"Diretório criado/verificado: {os.path.dirname(video_path)}")
    except Exception as e:
        logging.error(f"Erro ao criar/verificar o diretório: {e}")
        return

    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    for i, frame in enumerate(video_content):
        if isinstance(frame, np.ndarray):
            out.write(frame)
            logging.debug(f"Frame {i} adicionado ao vídeo.")
        else:
            logging.error(f"Frame {i} não é um numpy array: {type(frame)}")

    out.release()
    logging.info(f"Vídeo salvo em: {video_path}")

    if os.path.exists(video_path):
        logging.info(f"Arquivo de vídeo salvo com sucesso: {video_path}")
    else:
        logging.error(f"Falha ao salvar o arquivo de vídeo: {video_path}")
