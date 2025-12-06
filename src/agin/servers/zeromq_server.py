from agin import AginStreamProcessor, SharedTensor
from agin.utils import crop_maximal_rectangle
import json
import time
import zmq


class ZeromqServer:
    def __init__(self, stream_processor_config_path: str, port: str):
        """
        Args:
            stream_processor_config_path: config file path
            port: zeromq port, for example "tcp://*:5555"
        """
        self.stream_processor_config_path = stream_processor_config_path
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(port) 

    def run(self) -> None:
        self.stream_processor = AginStreamProcessor(self.stream_processor_config_path)
        self.stream_processor.start()
        print("Zero MQ server is running on port", self.port)

        while True:
            message = self.socket.recv_json()
            print("Received request", message)

            method = message.get("method")
            args = message.get("args", [])

            if method == "get_input_shared_tensor_name":
                result = self.stream_processor.get_input_shared_tensor_name()
            elif method == "get_output_shared_tensor_name":
                result = self.stream_processor.get_output_shared_tensor_name()
            elif method == "get_resolution":
                result = self.stream_processor.get_resolution()
            elif method == "set_param":
                self.stream_processor.set_param(name = args[0], value = args[1])
                result = "ok"
            elif method == "ping":
                result = "pong"
            else:
                result = f"Unknown method '{method}'"

            self.socket.send_json({"return": result})
