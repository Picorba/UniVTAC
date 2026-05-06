# vtla_diffusion_policy.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPTokenizer, CLIPTextModel
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


class UniVTACEncoder(nn.Module):
    """
    Placeholder — replace with your actual UniVTAC encoder.
    Expected interface:
        input  : (B, 3, H, W)  per sensor independently
        output : (B, tac_feat_dim)
    """
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Linear(512, output_dim)
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CLIPLanguageEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model     = CLIPTextModel.from_pretrained(model_name)
        # freeze — CLIP is used as a fixed feature extractor
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, instructions: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(next(self.model.parameters()).device)
        return self.model(**tokens).pooler_output  # (B, 512)


class VTLAObservationEncoder(nn.Module):
    """
    Encodes all modalities into a single flat conditioning vector
    for the diffusion denoiser.
    """

    def __init__(
        self,
        hidden_dim:   int = 512,
        tac_feat_dim: int = 256,
        lang_dim:     int = 512,   # CLIP output dim
        state_dim:    int = 9,
        num_cams:     int = 2,
        univtac_ckpt: str = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Visual encoder — shared ResNet18 across cameras
        cam_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        cam_backbone.fc = nn.Identity()
        self.cam_encoder = cam_backbone
        self.cam_proj    = nn.Linear(512 * num_cams, hidden_dim)

        # Tactile encoder — UniVTAC per sensor
        self.tac_encoder = UniVTACEncoder(output_dim=tac_feat_dim)
        if univtac_ckpt:
            state = torch.load(univtac_ckpt, map_location="cpu")
            self.tac_encoder.load_state_dict(state)
        self.tac_proj = nn.Linear(tac_feat_dim * 2, hidden_dim)  # left + right

        # Language projection
        self.lang_proj  = nn.Linear(lang_dim, hidden_dim)

        # State (joint) projection
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self,
        head_rgb:      torch.Tensor,   # (B, 3, H, W)
        wrist_rgb:     torch.Tensor,   # (B, 3, H, W)
        left_tactile:  torch.Tensor,   # (B, 3, H, W)
        right_tactile: torch.Tensor,   # (B, 3, H, W)
        lang_emb:      torch.Tensor,   # (B, 512)  pre-encoded by CLIP
        state:         torch.Tensor,   # (B, state_dim)
    ) -> torch.Tensor:                 # (B, hidden_dim)

        # cameras
        cams   = torch.cat([
            self.cam_encoder(head_rgb),
            self.cam_encoder(wrist_rgb),
        ], dim=-1)
        v_feat = self.cam_proj(cams)

        # tactile
        tacs   = torch.cat([
            self.tac_encoder(left_tactile),
            self.tac_encoder(right_tactile),
        ], dim=-1)
        t_feat = self.tac_proj(tacs)

        # language + state
        l_feat = self.lang_proj(lang_emb)
        s_feat = self.state_proj(state)

        return self.fusion(torch.cat([v_feat, t_feat, l_feat, s_feat], dim=-1))


class VTLADiffusionPolicy(DiffusionPolicy):
    """
    Diffusion Policy with vision + tactile (UniVTAC) + language conditioning.
    Subclasses LeRobot's DiffusionPolicy and overrides the observation encoder.
    """

    def __init__(self, config, univtac_ckpt: str = None):
        super().__init__(config)

        self.obs_encoder = VTLAObservationEncoder(
            hidden_dim   = config.hidden_dim,
            tac_feat_dim = config.tac_feat_dim,
            state_dim    = config.state_dim,
            univtac_ckpt = univtac_ckpt,
        )
        self.lang_encoder = CLIPLanguageEncoder()

        # Project conditioning vector into diffusion model's expected dim
        self.cond_proj = nn.Linear(
            config.hidden_dim,
            config.diffusion_step_embed_dim,
        )

    def forward(self, batch: dict) -> dict:
        instructions = batch["observation.language_instruction"]   # list[str]

        with torch.no_grad():
            lang_emb = self.lang_encoder(instructions)             # (B, 512)

        cond = self.obs_encoder(
            head_rgb      = batch["observation.images.head"],
            wrist_rgb     = batch["observation.images.wrist"],
            left_tactile  = batch["observation.images.left_tactile"],
            right_tactile = batch["observation.images.right_tactile"],
            lang_emb      = lang_emb,
            state         = batch["observation.state"],
        )
        batch["observation.cond"] = self.cond_proj(cond)
        return super().forward(batch)