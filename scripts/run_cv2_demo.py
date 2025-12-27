from agin import StreamProcessor
from agin.utils import crop_maximal_rectangle
import cv2

def main():
    config_path = "configs/stream_processor_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()
    stream_processor.start()
    
    stream_processor.set_prompt("A man in the cyberpunk street, night, neon lamps, colorful")
    resolution = stream_processor.get_resolution()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        resized_frame = crop_maximal_rectangle(frame, resolution["height"], resolution["width"])
        input_tensor.copy_from(resized_frame)
        processed_frame = output_tensor.to_numpy()
        cv2.imshow("processed stream", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream_processor.stop()

if __name__ == "__main__":
    main()