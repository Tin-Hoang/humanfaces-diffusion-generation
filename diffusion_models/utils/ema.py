class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.device = next(model.parameters()).device
        self.register(model)

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach().to(self.device)

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data
                    + (1.0 - self.decay) * param.data.to(self.device)
                )

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

