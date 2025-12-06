from multiprocessing import Process, Value, Manager, Event
from agin.utils.shared_tensor import SharedTensor
import time


class OutputSchedulerSubprocess:
    def __init__(self, config, output_batch_shared_tensor_name, output_shared_tensor_name, pack_is_ready, last_processing_time):
        self.config = config
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.output_shared_tensor_name = output_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time

        self.running = Value('b', False)
        self.process = None


    def start(self) -> None:
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self) -> None:
        self.running.value = False
        if self.process:
            self.process.join()


    def process_init(self) -> None:
        """
        Called by the internal process
        """
        height = self.config["resolution"]["height"]
        width = self.config["resolution"]["width"]
        
        self.output_batch_shared_tensor = SharedTensor((2, height, width, 3), name=self.output_batch_shared_tensor_name)
        self.output_shared_tensor = SharedTensor((height, width, 3), name=self.output_shared_tensor_name)

    def process_main(self) -> None:
        self.process_init()

        while self.running.value:
            if not self.pack_is_ready.value:
                continue
            
            self.output_shared_tensor.copy_from(self.output_batch_shared_tensor.array[0])
            time.sleep(self.last_processing_time.value / 2)
            self.output_shared_tensor.copy_from(self.output_batch_shared_tensor.array[1])
            self.pack_is_ready.value = False
