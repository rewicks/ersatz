import torch

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, warmup, target_lr, gamma=0.95):
        self.optimizer = optimizer
        self.warmup = warmup
        self.target_lr = target_lr
        self.step_size = step_size
        self.gamma = gamma
        self._rate = 0
        super(WarmupScheduler, self).__init__(optimizer, -1, False)




    def get_lr(self, step=None):
        "Implement `lrate` above"
        if self._step_count <= self.warmup:
            return [(self.target_lr * self._step_count) / self.warmup for group in self.optimizer.param_groups]
        else:
            if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma
                    for group in self.optimizer.param_groups]

