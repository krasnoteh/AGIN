from agin.servers.zeromq_server import ZeromqServer


def main():
    server = ZeromqServer("configs/agin_stream_processor_config.json", port = "tcp://*:5555")
    server.run()


if __name__ == "__main__":
    main()