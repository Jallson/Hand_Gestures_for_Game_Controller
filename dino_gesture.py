#!/usr/bin/env python3

import threading
import time
import argparse
import os
import cv2
import pygame
import random

# ---------- Config ----------
MODEL_PATH_DEFAULT = "/home/kinetika/gestures.eim"
CAMERA_INDEX_DEFAULT = 0
CONFIDENCE_THRESHOLD = 0.60

WINDOW_W = 800
WINDOW_H = 300
FPS = 60

GRAVITY = 0.9
JUMP_VELOCITY = -12
GROUND_Y = WINDOW_H - 40

COOLDOWN_TIME = 0.5  # jump cooldown seconds
DUCK_DURATION = 1.0  # seconds to stay ducked
MAX_LIVES = 3

# Shared state between threads
shared = {
    "gesture": "none",
    "prob": 0.0,
    "jump_flag": False,
    "duck_flag": False,
    "running": True,
    "last_gesture_time": 0.0,
    "last_input_gesture": "none",  # for display
}

# ---------- Gesture Thread ----------
try:
    from edge_impulse_linux.image import ImageImpulseRunner
except Exception:
    ImageImpulseRunner = None


class GestureStreamingThread(threading.Thread):
    def __init__(self, model_path, cam_idx=0, conf_thresh=CONFIDENCE_THRESHOLD):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.cam_idx = int(cam_idx)
        self.conf_thresh = conf_thresh

    def run(self):
        if ImageImpulseRunner is None:
            print("[GestureThread] Edge Impulse not available.")
            return
        if not os.path.exists(self.model_path):
            print(f"[GestureThread] Model not found: {self.model_path}")
            return

        print(f"[GestureThread] Loading model: {self.model_path}")
        try:
            with ImageImpulseRunner(self.model_path) as runner:
                runner.init()
                for res, img in runner.classifier(self.cam_idx):
                    if not shared.get("running", True):
                        break
                    boxes = res.get("result", {}).get("bounding_boxes", [])
                    detected_label = "none"
                    detected_conf = 0.0

                    for bb in boxes:
                        label = bb.get('label') or bb.get('class')
                        value = float(bb.get('value') or bb.get('score') or 0)
                        if label and value >= self.conf_thresh and value > detected_conf:
                            detected_label = label
                            detected_conf = value

                    shared["gesture"] = detected_label
                    shared["prob"] = detected_conf

                    now = time.time()
                    if detected_label.lower() == "peace":
                        if now - shared.get("last_gesture_time", 0) >= COOLDOWN_TIME:
                            shared["jump_flag"] = True
                            shared["last_gesture_time"] = now
                            shared["last_input_gesture"] = "peace"
                    elif detected_label.lower() == "good":
                        shared["duck_flag"] = True
                        shared["last_input_gesture"] = "good"
                    else:
                        shared["duck_flag"] = False

                    time.sleep(0.001)
        except Exception as e:
            print("[GestureThread] Failed:", e)
        finally:
            shared["running"] = False
            print("[GestureThread] Exiting")


# ---------- Game Entities ----------
class Dino:
    def __init__(self, x=50, y=GROUND_Y):
        self.x = x
        self.y = y
        self.w = 40
        self.h = 40
        self.dy = 0.0
        self.on_ground = True
        self.is_ducking = False
        self.duck_end_time = 0.0

    def update(self):
        self.dy += GRAVITY * 0.6
        self.y += self.dy

        if self.y >= GROUND_Y - self.h:
            self.y = GROUND_Y - self.h
            self.dy = 0.0
            self.on_ground = True
        else:
            self.on_ground = False

        if self.is_ducking and time.time() > self.duck_end_time:
            self.is_ducking = False
            self.h = 40
            self.y = GROUND_Y - self.h

    def jump(self):
        if self.on_ground and not self.is_ducking:
            self.dy = JUMP_VELOCITY
            self.on_ground = False

    def duck(self):
        if self.on_ground and not self.is_ducking:
            self.is_ducking = True
            self.h = 20
            self.y = GROUND_Y - self.h
            self.duck_end_time = time.time() + DUCK_DURATION

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)


class Cactus:
    def __init__(self, x, speed):
        self.x = x
        self.y = GROUND_Y - 30
        self.w = 20
        self.h = 30
        self.speed = speed

    def update(self):
        self.x -= self.speed

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)


class Bird:
    def __init__(self, x, speed):
        self.x = x
        self.y = GROUND_Y - 50  # lower flying height so Dino must duck
        self.w = 30
        self.h = 20
        self.speed = speed

    def update(self):
        self.x -= self.speed

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)


# ---------- Main Loop ----------
def main(model_path, camera_index):
    gst = GestureStreamingThread(model_path=model_path, cam_idx=camera_index)
    gst.start()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Dino - Jump & Duck Gestures")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 48)

    dino = Dino()
    obstacles = []
    spawn_timer = 0
    score = 0
    lives = MAX_LIVES
    onboarding_frames = FPS * 3

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Gesture input only
        if shared.get("jump_flag", False):
            dino.jump()
            shared["jump_flag"] = False
        if shared.get("duck_flag", False):
            dino.duck()
            shared["duck_flag"] = False

        # Update Dino
        dino.update()

        # Spawn obstacles
        spawn_timer -= 1
        if spawn_timer <= 0:
            speed = 6 + (score // 100)
            if random.random() < 0.3:
                obstacles.append(Bird(WINDOW_W + 50, speed + 1))
            else:
                obstacles.append(Cactus(WINDOW_W + 50, speed))
            spawn_timer = max(30, int(90 - min(50, score // 10)))

        # Update obstacles
        for c in list(obstacles):
            c.update()
            if c.x + c.w < 0:
                obstacles.remove(c)
                score += 10

        # Collision
        hit = False
        for c in list(obstacles):
            if dino.rect().colliderect(c.rect()):
                hit = True
                obstacles.remove(c)
                break

        if hit:
            lives -= 1
            if lives <= 0:
                running = False  # game over
            else:
                dino.y = GROUND_Y - dino.h
                dino.dy = 0
                dino.on_ground = True

        # Draw
        screen.fill((235, 235, 235))
        pygame.draw.rect(screen, (83, 83, 83),
                         pygame.Rect(0, GROUND_Y, WINDOW_W, WINDOW_H - GROUND_Y))

        # Dino
        pygame.draw.rect(screen, (20, 20, 20), dino.rect())

        # Obstacles
        for c in obstacles:
            color = (0, 0, 255) if isinstance(c, Bird) else (34, 139, 34)
            pygame.draw.rect(screen, color, c.rect())

        # HUD
        screen.blit(font.render(f"Score: {score}", True, (0, 0, 0)), (10, 10))
        screen.blit(font.render(f"Lives: {lives}", True, (0, 0, 0)), (10, 34))

        gesture = shared.get("last_input_gesture", "none")
        if gesture != "none":
            screen.blit(font.render(f"Gesture: {gesture}", True, (0, 0, 0)), (10, 58))

        if onboarding_frames > 0:
            onboarding_frames -= 1
            screen.blit(font.render("peace = jump | good = duck",
                                    True, (0, 0, 0)), (200, 10))

        pygame.display.flip()

    # Game Over Screen
    screen.fill((235, 235, 235))
    msg = big_font.render("GAME OVER", True, (200, 0, 0))
    screen.blit(msg, (WINDOW_W // 2 - msg.get_width() // 2,
                      WINDOW_H // 2 - msg.get_height() // 2))
    pygame.display.flip()
    time.sleep(2)

    print(f"Game Over! Final Score: {score}")
    shared["running"] = False
    try:
        gst.join(timeout=1.0)
    except Exception:
        pass
    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dino Gesture Game (jump+duck+3 lives)")
    p.add_argument("model", nargs="?", default=MODEL_PATH_DEFAULT)
    p.add_argument("camera", nargs="?", default=CAMERA_INDEX_DEFAULT)
    args = p.parse_args()
    try:
        cam_idx = int(args.camera)
    except Exception:
        cam_idx = CAMERA_INDEX_DEFAULT
    main(args.model, cam_idx)
