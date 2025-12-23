from agin import StreamProcessor, SharedTensor
from agin.utils import crop_maximal_rectangle
import json
import time
import cv2

def main():
    config_path = "configs/stream_processor_config.json"
    with open(config_path, 'r') as file:
        config = json.load(file)

    resolution = config["resolution"]

    stream_processor = StreamProcessor(config_path)
    stream_processor.start()
    input_shared_tensor = SharedTensor((resolution["height"], resolution["width"], 3), name=stream_processor.get_input_shared_tensor_name())
    output_shared_tensor = SharedTensor((resolution["height"], resolution["width"], 3), name=stream_processor.get_output_shared_tensor_name())

    stream_processor.set_prompt("A man in the cyberpunk street, night, neon lamps, colorful")

    cap = cv2.VideoCapture(4)
    while True:
        ret, frame = cap.read()
        resized_frame = crop_maximal_rectangle(frame, resolution["height"], resolution["width"])
        input_shared_tensor.copy_from(resized_frame)

        processed_frame = output_shared_tensor.to_numpy()

        cv2.imshow("processed stream", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)


    stream_processor.stop()


if __name__ == "__main__":
    main()