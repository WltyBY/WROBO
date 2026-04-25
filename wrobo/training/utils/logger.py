from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logging = {}

    def log(self, key, value, step):
        if key not in self.logging:
            self.logging[key] = []
        self.logging[key].append((step, value))

        self.writer.add_scalar(key, value, step)

    def log_for_dict(self, d, step):
        for k, v in d.items():
            if k not in self.logging:
                self.logging[k] = []
            self.logging[k].append((step, v))

            self.writer.add_scalar(k, v, step)

    def get_checkpoint(self):
        return self.logging

    def load_checkpoint(self, logging_dict):
        if logging_dict is not None:
            self.logging = logging_dict

    def close(self):
        self.writer.close()