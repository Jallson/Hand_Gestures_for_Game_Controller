## Introduction:

This Edge Impulse project is designed to demonstrate how easy it is to collect data (photos), train a model, and deploy it to a Raspberry Pi unit within a simple Python game. The project also showcases the reliability of Edge Impulse Studio in creating small-footprint ML models, enabling low latency and low energy consumption — making real-time object detection possible, even without demanding the highest-performance hardware. This approach is ideal for use in the gaming industry as a cost-effective solution.
Main goal of this project is to encourage beginners to learn how to use Edge AI in practical applications. Therefore, it utilizes simple and widely popular components such as the Raspberry Pi, the PyGame library, and classic games as learning media.


## How it works:

As an object detection project, we can use MobileNetV2 SSD with a relatively large dataset (for example, around 100 images per class), or alternatively, we can simplify the process by using YOLO, since YOLOv5 already comes pre-trained with basic hand recognition. This allows us to use a smaller dataset — in our case around 40 images per class should be ok.
In this simulation, we will create four gesture classes: neutral (fist), five (open hand), peace (V-sign), and good (thumbs up). The bounding box detection will provide class data along with the object’s x, y, w, and h values, which we will use as real-time input to replace a keyboard or joystick in the classic game we’re developing.
The trained model will then be deployed to a Raspberry Pi, integrated into our Python code that uses the PyGame library for easy development and rendering of the game on an LCD display.

## Hardware Component:

- Raspberry Pi 5
- Keyboard, mouse or PC/laptop via ssh
- USB Camera/webcam (eg. Logitech C920) or Pi Camera
- LCD/monitor
- Mini tripod (optional)

## Software & Online Services:

- EdgeImpulse Studio
- EdgeImpulse’s Linux & Python SDK
- Raspberry Pi OS
- Open CV
- PyGame Library
