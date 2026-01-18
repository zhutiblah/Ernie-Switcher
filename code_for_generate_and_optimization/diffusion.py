import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from ema import EMA
from utils import extract
from utils_extra import extract_rna_ernie_for_prediction

class GaussianDiffusion(nn.Module):
    
    def __init__(
        self,
        model,
        img_size,
        img_channels,
        num_classes,
        betas,
        loss_type="l2",
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
        
        use_large_model=False,
        vocab_path=None,
        ernie_model_path=None,
        embedding_loss_weight=0,
        embedding_loss_steps=50,
        k_mer=1,
        max_seq_len=115,
        
        unet_feature_dim=None,
        ernie_hidden_dim=768,
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)
        self.ema = EMA(ema_decay)
        
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        
        self.use_large_model = use_large_model
        self.vocab_path = vocab_path
        self.ernie_model_path = ernie_model_path
        self.embedding_loss_weight = embedding_loss_weight
        self.embedding_loss_steps = embedding_loss_steps
        self.k_mer = k_mer
        self.max_seq_len = max_seq_len
        
        
        if use_large_model and embedding_loss_weight > 0:
            if unet_feature_dim is None:
                raise ValueError("unet_feature_dim must be specified when use_large_model=True")
            
            self.projection = nn.Linear(unet_feature_dim, ernie_hidden_dim)
            print(f"✅ Projection layer created: {unet_feature_dim} -> {ernie_hidden_dim}")
        else:
            self.projection = None

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

        
        print("\n" + "="*60)
        print("GaussianDiffusion initialized")
        print("="*60)
        print(f"use_large_model: {use_large_model}")
        print(f"embedding_loss_weight: {embedding_loss_weight}")
        print(f"embedding_loss_steps: {embedding_loss_steps}")
        print(f"Projection layer status: {'Created' if self.projection is not None else 'Not created'}")
        print("="*60 + "\n")

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y=None, use_ema=True):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x.cpu().detach()

    @torch.no_grad()
    def sample_opt(self, batch_size, device, y=None, use_ema=True,x=None):

        if  x.numel() == 0:
            x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x.cpu().detach()
    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def calculate_embedding_loss(self, diffusion_features, large_model_embeddings):
        """
        Align UNet bottleneck features with RNA-Ernie embeddings
        
        Args:
            diffusion_features: [batch_size, feature_dim] from UNet
            large_model_embeddings: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        
        Returns:
            loss: cosine similarity loss
        """
        
        if large_model_embeddings.dim() == 3:
            large_model_embeddings = large_model_embeddings.mean(dim=1)
        
        
        if self.projection is not None:
            diffusion_features = self.projection(diffusion_features)
        
        
        diffusion_features = F.normalize(diffusion_features, dim=-1)
        large_model_embeddings = F.normalize(large_model_embeddings, dim=-1)
        
        
        similarity = (diffusion_features * large_model_embeddings).sum(dim=-1)
        loss = (1 - similarity).mean()
        
        
        if self.step % 100 == 0:
            print(f"[Step {self.step}] Embedding Loss: {loss.item():.6f}, Average Similarity: {similarity.mean().item():.6f}")
        
        return loss

    def get_losses(self, x, t, y, sequences=None, force_embedding=False):
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise)
        
        
        need_embedding_loss = (
                self.use_large_model and
                self.embedding_loss_weight > 0 and
                sequences is not None and
                (force_embedding or self.step % self.embedding_loss_steps == 0)
                
            )
        
        
        if need_embedding_loss:
            estimated_noise, diffusion_features = self.model(
                perturbed_x, t, y, return_features=True
            )
        else:
            estimated_noise = self.model(perturbed_x, t, y)
        
        
        if self.loss_type == "l1":
            diffusion_loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            diffusion_loss = F.mse_loss(estimated_noise, noise)
        
        
        embedding_loss = torch.tensor(0.0, device=x.device)
        
        if need_embedding_loss:
            try:
                
                if self.step % 100 == 0:
                    print(f"\n{'='*60}")
                    print(f"[Step {self.step}] Calculating embedding loss")
                    print(f"Number of sequences: {len(sequences)}")
                    print(f"Diffusion feature shape: {diffusion_features.shape}")
                
                with torch.no_grad():
                    large_model_embeddings = extract_rna_ernie_for_prediction(
                        mRNAs_list=list(sequences),
                        vocab_path=self.vocab_path,
                        ernie_model_path=self.ernie_model_path,
                        k_mer=self.k_mer,
                        max_seq_len=self.max_seq_len,
                        batch_size=min(32, len(sequences)),
                    )
                    large_model_embeddings = torch.tensor(
                        large_model_embeddings, 
                        dtype=torch.float32
                    ).to(x.device)
                
                if self.step % 100 == 0:
                    print(f"RNA-Ernie embedding shape: {large_model_embeddings.shape}")
                    print(f"{'='*60}\n")
                
                embedding_loss = self.calculate_embedding_loss(
                    diffusion_features,
                    large_model_embeddings
                )
            
            except Exception as e:
                print(f"❌ Failed to calculate embedding loss: {e}")
                import traceback
                traceback.print_exc()
                embedding_loss = torch.tensor(0.0, device=x.device)
        
        
        total_loss = diffusion_loss + self.embedding_loss_weight * embedding_loss
        
        
        if self.step % 50 == 0:
            print(f"[Step {self.step}] Diffusion Loss: {diffusion_loss.item():.6f} | "
                  f"Embedding Loss: {embedding_loss.item():.6f} | "
                  f"Total Loss: {total_loss.item():.6f}")
        
        return {
        'total_loss': total_loss,
        'diffusion_loss': diffusion_loss,
        'embedding_loss': embedding_loss
    }

    def forward(self, x, y=None, sequences=None,force_embedding=False):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[1]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        loss_dict = self.get_losses(x, t, y, sequences=sequences,force_embedding=force_embedding)
        return loss_dict
    
    
    def print_model_info(self):
        """Print model parameter information for debugging"""
        print("\n" + "="*60)
        print("Model Parameter Diagnostics")
        print("="*60)
        
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"✓ {name:50s} | Shape: {str(list(param.shape)):20s} | Trainable")
            else:
                print(f"✗ {name:50s} | Shape: {str(list(param.shape)):20s} | Frozen")
        
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print("="*60 + "\n")
        
        
        if self.projection is not None:
            print("✅ Projection layer parameters:")
            for name, param in self.projection.named_parameters():
                print(f"   {name}: {list(param.shape)}, requires_grad={param.requires_grad}")
        else:
            print("⚠️ Projection layer not created")
        print()


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []
    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


def verify_unet_features(model, device='cuda'):
    """
    Verify if UNet correctly supports return_features parameter
    
    Args:
        model: UNet model instance
        device: device
    
    Returns:
        bool: whether feature return is supported
    """
    print("\n" + "="*60)
    print("UNet Feature Extraction Verification")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(2, 1, 4, 115).to(device)
        t_test = torch.randint(0, 1000, (2,)).to(device)
        y_test = torch.randint(0, 10, (2,)).to(device)
        
        
        try:
            out1 = model(x_test, t_test, y_test)
            print(f"✅ Normal forward pass successful: Output shape {out1.shape}")
        except Exception as e:
            print(f"❌ Normal forward pass failed: {e}")
            return False
        
        
        try:
            result = model(x_test, t_test, y_test, return_features=True)
            
            if isinstance(result, tuple) and len(result) == 2:
                out2, feat2 = result
                print(f"✅ Feature extraction successful:")
                print(f"   Noise prediction shape: {out2.shape}")
                print(f"   Feature shape: {feat2.shape}")
                print(f"   Feature dimension: {feat2.shape[-1]}")
                print("="*60 + "\n")
                return True
            else:
                print(f"❌ Incorrect return format: {type(result)}")
                return False
                
        except TypeError as e:
            print(f"❌ UNet does not support return_features parameter: {e}")
            print("   Please add return_features parameter in UNet's forward method")
            print("="*60 + "\n")
            return False
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
