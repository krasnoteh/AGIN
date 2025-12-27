from agin import StreamProcessor
from agin.utils import crop_maximal_rectangle
import cv2
import numpy as np
import time

def main():
    config_path = "configs/stream_processor_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()
    stream_processor.start()
    
    stream_processor.set_prompt("A man in the cyberpunk street, night, neon lamps, colorful")
    resolution = stream_processor.get_resolution()

    prompts = [
        "A young man in the night desert next to wooden portal to another dimension, starry colorful sky, desert plants, purple and blue tones, anime art",
        "Art in style of Wassily Kandinsky, oil on canvas, abstract, colorful",
        "A young man in the jungle scene, bioluminescent, colorful, cinematic, realistic, hdr, professional art",
        "A man in the highly detailed, immersive cyberpunk street scene set at night, drenched in vibrant neon lights, futuristic technology, and a sense of urban density. The environment should evoke a dystopian yet mesmerizing atmosphere, blending high-tech elements with urban decay",
        "A young Kazakh man in the brown national costume. Fur elements, golden patterns, metal belt. Sunny desert of Kazakhstan, cliffs of the blurry background",
        "A young man in the bioluminescent forest of giant orange and purple mushrooms, giant Mushrooms, lighting sheres are floating in the air, particles, alien flora, otherworldy, night, haze, red neon, rtx rended, reflections, watr, bokeh, epic cinematic art, colorful",
    ]

    cap = cv2.VideoCapture(0)
    t = time.time()
    i = 0
    while True:
        ret, frame = cap.read()
        resized_frame = crop_maximal_rectangle(frame, resolution["height"], resolution["width"])
        input_tensor.copy_from(resized_frame)
        processed_frame = output_tensor.to_numpy()

        processed_frame = np.concatenate([crop_maximal_rectangle(frame, 256, 256), crop_maximal_rectangle(processed_frame, 256, 256)], axis = 1)

        if time.time() - t > 3:
            i+=1
            stream_processor.set_prompt(prompts[i % len(prompts)])
            t = time.time() 

        cv2.imshow("processed stream", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream_processor.stop()

if __name__ == "__main__":
    main()