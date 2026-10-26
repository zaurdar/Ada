import ast
import base64

import pyautogui
from PIL import Image, ImageDraw
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

from ui_tars.action_parser import parse_action_to_structure_output



def draw_debug_point(
    screenshot_path,
    x_norm,
    y_norm,
    output_path="screen_debug.png"
):
    """
    Dessine la position choisie par UI-TARS
    directement sur l'image 1280x720 envoyée au modèle.
    """

    image = Image.open(screenshot_path).convert("RGB")

    width, height = image.size

    # Position normalisée -> pixels de NOTRE screenshot
    x = round(x_norm * width)
    y = round(y_norm * height)

    draw = ImageDraw.Draw(image)

    radius = 18
    thickness = 4

    # Cercle
    for i in range(thickness):
        draw.ellipse(
            (
                x - radius - i,
                y - radius - i,
                x + radius + i,
                y + radius + i
            ),
            outline="red"
        )

    # Croix
    draw.line(
        (x - 30, y, x + 30, y),
        fill="red",
        width=4
    )

    draw.line(
        (x, y - 30, x, y + 30),
        fill="red",
        width=4
    )

    # Coordonnées
    label = f"UI-TARS ({x}, {y})"

    draw.text(
        (x + 25, y + 20),
        label,
        fill="red"
    )

    image.save(output_path)

    print(f"\nDebug sauvegardé : {output_path}")
    print(f"Position sur screenshot : ({x}, {y})")

    return output_path
# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/ui-tars/UI-TARS-2B-SFT-Q4_K_M.gguf"
MMPROJ_PATH = "models/ui-tars/mmproj-UI-TARS-2B-SFT-f16.gguf"

SCREENSHOT_PATH = "screen.png"

# Taille envoyée au VLM
MODEL_WIDTH = 1280
MODEL_HEIGHT = 720

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.3


# ============================================================
# PROMPT UI-TARS
# ============================================================

UI_TARS_PROMPT = r"""
You are a GUI agent. You are given a task and your action history,
with screenshots. You need to perform the next action to complete
the task.

## Output Format

Thought: ...
Action: ...

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')

left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')

right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')

drag(
    start_box='<|box_start|>(x1,y1)<|box_end|>',
    end_box='<|box_start|>(x2,y2)<|box_end|>'
)

hotkey(key='')

type(content='')

scroll(
    start_box='<|box_start|>(x1,y1)<|box_end|>',
    direction='down or up or right or left'
)

wait()

finished()

call_user()

## Note

- Use English in the Thought part.
- Write a small plan and finally summarize your next action
  with its target element in one sentence in the Thought part.

## User Instruction

"""


# ============================================================
# IMAGE
# ============================================================

def take_screenshot():
    """
    Screenshot du vrai écran puis redimensionnement pour UI-TARS.
    """

    screenshot = pyautogui.screenshot()

    print(
        f"Écran réel : "
        f"{screenshot.width}x{screenshot.height}"
    )

    screenshot = screenshot.resize(
        (MODEL_WIDTH, MODEL_HEIGHT),
        Image.Resampling.LANCZOS
    )

    screenshot.save(SCREENSHOT_PATH)

    print(
        f"Image UI-TARS : "
        f"{MODEL_WIDTH}x{MODEL_HEIGHT}"
    )

    return SCREENSHOT_PATH


def image_to_data_uri(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


# ============================================================
# CHARGEMENT UI-TARS
# ============================================================

print("Chargement vision...")

chat_handler = Qwen25VLChatHandler(
    clip_model_path=MMPROJ_PATH
)

print("Chargement UI-TARS...")

llm = Llama(
    model_path=MODEL_PATH,
    chat_handler=chat_handler,

    n_gpu_layers=-1,

    n_ctx=4096,
    n_batch=256,

    verbose=True
)

print("\nUI-TARS chargé.")


# ============================================================
# INFERENCE
# ============================================================

def inference(instruction):
    image_path = take_screenshot()
    image_uri = image_to_data_uri(image_path)

    messages = [
        {
            "role": "system",
            "content": UI_TARS_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_uri
                    }
                },
                {
                    "type": "text",
                    "text": instruction
                }
            ]
        }
    ]

    print("\nInférence...")
    print(f"Instruction : {instruction}")

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=256
    )

    text = response["choices"][0]["message"]["content"]

    return text


# ============================================================
# PARSER UI-TARS
# ============================================================

def parse_response(text):
    """
    Conversion de la sortie UI-TARS en action structurée.

    IMPORTANT :
    on donne ici la taille de l'image réellement vue par UI-TARS.
    """

    return parse_action_to_structure_output(
        text,
        factor=1000,
        origin_resized_height=MODEL_HEIGHT,
        origin_resized_width=MODEL_WIDTH,
        model_type="qwen2vl"
    )


# ============================================================
# DEBUG COORDONNEES
# ============================================================

def inspect_action(actions):
    """
    Affiche et visualise l'action choisie par UI-TARS.
    Aucun clic n'est exécuté.
    """

    if not actions:
        print("\nAucune action reconnue.")
        return

    action = actions[0]

    action_type = action["action_type"]
    inputs = action.get("action_inputs", {})

    print("\n===== ACTION INTERPRÉTÉE =====")
    print("Type :", action_type)
    print("Inputs :", inputs)

    if action_type not in {
        "click",
        "left_double",
        "right_single"
    }:
        print("\nAction sans coordonnées de clic.")
        return

    raw_box = inputs.get("start_box")

    if raw_box is None:
        print("\nPas de start_box.")
        return

    box = ast.literal_eval(raw_box)

    x_norm = box[0]
    y_norm = box[1]

    print("\nBox parsée :", box)

    print(
        f"Coordonnées normalisées : "
        f"({x_norm:.6f}, {y_norm:.6f})"
    )

    # --------------------------------------------------------
    # Coordonnées sur screenshot 1280x720
    # --------------------------------------------------------

    image_x = round(x_norm * MODEL_WIDTH)
    image_y = round(y_norm * MODEL_HEIGHT)

    print(
        f"Coordonnées screenshot : "
        f"({image_x}, {image_y})"
    )

    # --------------------------------------------------------
    # Coordonnées écran Windows
    # --------------------------------------------------------

    screen_width, screen_height = pyautogui.size()

    screen_x = round(x_norm * screen_width)
    screen_y = round(y_norm * screen_height)

    print(
        f"Coordonnées écran : "
        f"({screen_x}, {screen_y})"
    )

    # --------------------------------------------------------
    # Debug visuel
    # --------------------------------------------------------

    draw_debug_point(
        SCREENSHOT_PATH,
        x_norm,
        y_norm,
        "screen_debug.png"
    )

    print("\n⚠️ MODE TEST : aucun clic effectué.")

    # Quand on aura validé :
    #
    # pyautogui.click(screen_x, screen_y)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("      UI-TARS SANDBOX")
    print("===================================")

    print("\nCommandes de test conseillées :")
    print("  Click the Windows Start button.")
    print("  Click the PyCharm icon.")
    print("  Click the Notepad icon.")

    instruction = input(
        "\nInstruction > "
    ).strip()

    if not instruction:
        raise ValueError("Instruction vide.")

    # --------------------------------------------------------
    # 1. Screenshot + inference
    # --------------------------------------------------------

    response = inference(instruction)

    print("\n\n===== RÉPONSE UI-TARS =====\n")
    print(response)

    # --------------------------------------------------------
    # 2. Parsing
    # --------------------------------------------------------

    try:
        actions = parse_response(response)

    except Exception as e:

        print("\n===== ERREUR PARSER =====")
        print(e)

        actions = []

    print("\n===== ACTION PARSÉE =====")
    print(actions)

    # --------------------------------------------------------
    # 3. Conversion coordonnées
    # --------------------------------------------------------

    inspect_action(actions)