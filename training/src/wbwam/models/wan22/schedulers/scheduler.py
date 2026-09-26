import torch


class WanFlowMatchScheduler:
    """Minimal Flow-Matching scheduler compatible with Wan training/inference."""

    def __init__(self):
        self.num_train_timesteps = 1000
        self.sigmas = None
        self.timesteps = None
        self.training = False
        self.linear_timesteps_weights = None

    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=5.0):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas) # sth like beta distribution
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        # Copied from existing implementation for behavior parity.
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        weights = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            weights = weights * (len(self.timesteps) / steps)
            weights = weights + weights[1]
        self.linear_timesteps_weights = weights

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=5.0):
        self.sigmas, self.timesteps = self.set_timesteps_wan(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=shift,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def _resolve_timestep_ids(self, timestep):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], dtype=self.timesteps.dtype)
        timestep = timestep.detach().to(self.timesteps.device).reshape(-1)
        dist = (self.timesteps.view(-1, 1) - timestep.view(1, -1)).abs()
        return torch.argmin(dist, dim=0)

    def _resolve_sigmas(self, timestep):
        timestep_ids = self._resolve_timestep_ids(timestep)
        sigmas = self.sigmas[timestep_ids]
        if sigmas.numel() == 1:
            return sigmas[0]
        return sigmas

    def step(self, model_output, timestep, sample):
        timestep_id = self._resolve_timestep_ids(timestep)[0]
        sigma = self.sigmas[timestep_id]
        sigma_next = 0 if timestep_id + 1 >= len(self.timesteps) else self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_next - sigma).to(sample.device, dtype=sample.dtype)

    def add_noise(self, original_samples, noise, timestep):
        sigma = self._resolve_sigmas(timestep).to(original_samples.device, dtype=original_samples.dtype)
        if isinstance(sigma, torch.Tensor) and sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        del timestep
        return noise - sample

    def training_weight(self, timestep):
        timestep_ids = self._resolve_timestep_ids(timestep)
        weights = self.linear_timesteps_weights[timestep_ids]
        if weights.numel() == 1:
            return weights[0]
        return weights
