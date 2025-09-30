#!/usr/bin/env python3
import cv2
import pygame
import sys
import numpy as np
import time
from edge_impulse_linux.image import ImageImpulseRunner

# ========== MODEL SETUP ==========
model_path = "/home/kinetika/gestures.eim"

runner = ImageImpulseRunner(model_path)
model_info = runner.init()
print(f"✅ Loaded model: {model_info['project']['name']}")

# Determine input size
try:
    input_width = model_info['model_parameters']['input_width']
    input_height = model_info['model_parameters']['input_height']
except KeyError:
    print("⚠️ Using default input size 320x320")
    input_width, input_height = 320, 320

# ========== CAMERA SETUP ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera")
    sys.exit(1)
print("📸 Camera initialized.")

# ========== GAME SETUP ==========
pygame.init()

WINDOW_W, WINDOW_H = 800, 400
window = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Pong Gesture Game 🏓")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddles & Ball
PADDLE_W, PADDLE_H = 15, 100
BALL_SIZE = 20
BALL_SPEED_X, BALL_SPEED_Y = 10, 10
PADDLE_SPEED = 10

# Positions
left_paddle = pygame.Rect(20, WINDOW_H // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
right_paddle = pygame.Rect(WINDOW_W - 20 - PADDLE_W, WINDOW_H // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
ball = pygame.Rect(WINDOW_W // 2 - BALL_SIZE // 2, WINDOW_H // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)

# Score
score_left, score_right = 0, 0
font = pygame.font.SysFont("Arial", 28)

clock = pygame.time.Clock()
FPS = 30

def reset_ball():
    ball.x = WINDOW_W // 2 - BALL_SIZE // 2
    ball.y = WINDOW_H // 2 - BALL_SIZE // 2
    global BALL_SPEED_X, BALL_SPEED_Y
    BALL_SPEED_X *= -1
    BALL_SPEED_Y = np.random.choice([-4, 4])

print("🎮 Starting game... Press 'Q' to quit.")

# ========== GAME LOOP ==========
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            runner.stop()
            pygame.quit()
            sys.exit()

    ret, frame = cap.read()
    if not ret:
        continue

    # === Run classification ===
    try:
        features, cropped = runner.get_features_from_image(frame)
        result = runner.classify(features)
    except Exception as e:
        print("⚠️ Classification error:", e)
        continue

    # Default: no gesture
    left_move = 0
    right_move = 0

    # === Process detections ===
    if "bounding_boxes" in result["result"]:
        boxes = result["result"]["bounding_boxes"]
        for box in boxes:
            label = box["label"]
            x, y, w, h = box["x"], box["y"], box["width"], box["height"]
            center_y = y + h / 2

            # Gesture-based control
            if label == "five":  # Left paddle control
                if center_y < input_height / 2:
                    left_move = -PADDLE_SPEED  # move up
                else:
                    left_move = PADDLE_SPEED   # move down

            elif label == "peace":  # Right paddle control
                if center_y < input_height / 2:
                    right_move = -PADDLE_SPEED  # move up
                else:
                    right_move = PADDLE_SPEED   # move down

    # === Apply paddle movements ===
    left_paddle.y += left_move
    right_paddle.y += right_move

    # Clamp paddles inside window
    left_paddle.y = max(0, min(WINDOW_H - PADDLE_H, left_paddle.y))
    right_paddle.y = max(0, min(WINDOW_H - PADDLE_H, right_paddle.y))

    # === Ball physics ===
    ball.x += BALL_SPEED_X
    ball.y += BALL_SPEED_Y

    # Bounce walls
    if ball.top <= 0 or ball.bottom >= WINDOW_H:
        BALL_SPEED_Y *= -1

    # Bounce paddles
    if ball.colliderect(left_paddle) or ball.colliderect(right_paddle):
        BALL_SPEED_X *= -1

    # Scoring
    if ball.left <= 0:
        score_right += 1
        reset_ball()
    elif ball.right >= WINDOW_W:
        score_left += 1
        reset_ball()

    # === Render ===
    window.fill(BLACK)
    pygame.draw.rect(window, WHITE, left_paddle)
    pygame.draw.rect(window, WHITE, right_paddle)
    pygame.draw.ellipse(window, WHITE, ball)
    pygame.draw.aaline(window, WHITE, (WINDOW_W // 2, 0), (WINDOW_W // 2, WINDOW_H))

    score_text = font.render(f"{score_left}   |   {score_right}", True, WHITE)
    window.blit(score_text, (WINDOW_W // 2 - score_text.get_width() // 2, 20))

    pygame.display.flip()
    clock.tick(FPS)
