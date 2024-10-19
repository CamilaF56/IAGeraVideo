from io import BytesIO
from PIL import Image
from flask import Blueprint, render_template, request, url_for, send_from_directory
import openai
import numpy as np
import os
import logging
import requests
from moviepy.editor import ImageSequenceClip

bp = Blueprint("main", __name__)

# Configuração básica de logging
logging.basicConfig(level=logging.DEBUG)

# Substitua 'your-api-key' pela sua chave da OpenAI
openai.api_key = "Sua chave"

filename = "animated_video_encoded.mp4"

@bp.route("/")
def index():
    """
    Renderiza a página inicial do aplicativo.

    Returns:
        str: O template da página inicial.
    """
    return render_template("index.html")

@bp.route("/sobre")
def sobre():
    """
    Rota para a página 'Sobre'.

    Esta função é responsável por renderizar a página 'sobre.html',
    que contém informações sobre o projeto ou a organização. A rota
    está mapeada para o caminho '/sobre'.

    Returns:
        str: O conteúdo HTML da página 'sobre.html' renderizada.
    """
    return render_template("sobre.html")

@bp.route('/video/<filename>')
def serve_video(filename):
    """
    Serve um vídeo a partir do diretório estático.

    Args:
        filename (str): Nome do arquivo de vídeo.

    Returns:
        Response: O vídeo solicitado.
    """
    return send_from_directory(os.path.join('app', 'static'), filename)

@bp.route("/generate", methods=["POST"])
def generate():
    """
    Gera um vídeo a partir de um prompt fornecido pelo usuário.

    Método: POST

    Returns:
        str: O template do vídeo gerado ou uma mensagem de erro em caso de falha.
    """
    prompt = request.form["prompt"]
    steps = 15  # Número de frames para o movimento
    logging.debug(f"Prompt recebido: {prompt}")

    moving_prompts = generate_moving_prompts(prompt, steps)

    image_urls = []
    try:
        for p in moving_prompts:
            response = openai.Image.create(
                prompt=p,
                n=1,  # Uma imagem por prompt
                size="1024x1024",  # Tamanho da imagem
            )
            image_urls.append(response["data"][0]["url"])
            logging.debug(f"Imagem gerada para o prompt '{p}': {response['data'][0]['url']}")

        frames = create_frames_from_images(image_urls)
        if frames:
            video_path = os.path.join("app", "static", "animated_video_encoded.mp4")
            save_video(frames, video_path)
            return render_template("video.html", video_path=url_for("static", filename="animated_video_encoded.mp4"))
        else:
            logging.error("Nenhum frame gerado para salvar o vídeo.")
            return "Erro ao gerar vídeo", 500

    except Exception as e:
        logging.error(f"Erro ao gerar imagens com DALL-E: {e}")
        return "Erro ao gerar vídeo", 500


def create_frames_from_images(image_urls):
    """
    Cria frames a partir de URLs de imagens.

    Args:
        image_urls (list): Lista de URLs de imagens.

    Returns:
        list: Lista de frames em formato NumPy.
    """
    frames = []

    for img_url in image_urls:
        response = requests.get(img_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).resize((640, 480))
            frames.append(np.array(image))  # Converter para array NumPy
        else:
            logging.error(f"Erro ao baixar a imagem: {img_url}")

    return frames


def generate_moving_prompts(base_prompt, steps):
    """
    Gera prompts com variações de movimento.

    Args:
        base_prompt (str): O prompt base para geração de imagens.
        steps (int): Número de variações a serem geradas.

    Returns:
        list: Lista de prompts variáveis.
    """
    return [
        f"{base_prompt}, slight movement, frame {i + 1}, gradual shift, subtle change"
        for i in range(steps)
    ]


def save_video(video_content, video_path):
    """
    Salva um vídeo a partir de uma sequência de frames.

    Args:
        video_content (list): Lista de frames.
        video_path (str): Caminho para salvar o vídeo.

    Returns:
        None
    """
    logging.debug("Iniciando o processo de salvamento do vídeo.")

    # Usando MoviePy para salvar o vídeo
    clip = ImageSequenceClip(video_content, fps=15)
    clip.write_videofile(video_path, codec="libx264", audio_codec="aac")

    logging.info(f"Vídeo salvo em: {video_path}")

    if os.path.exists(video_path):
        logging.info(f"Arquivo de vídeo salvo com sucesso: {video_path}")
    else:
        logging.error(f"Falha ao salvar o arquivo de vídeo: {video_path}")
