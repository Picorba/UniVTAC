# train_vtla.py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from vtla_diffusion_policy import VTLADiffusionPolicy
import torch

CHUNK_SIZE  = 50
BATCH_SIZE  = 32
NUM_EPOCHS  = 200
LR          = 1e-4
UNIVTAC_CKPT = "encoder/checkpoints/resnet18/20251128-125750/best.pth"  # ← set this


def main():
    dataset = LeRobotDataset(
        repo_id="local/vtla_sim",
        delta_timestamps={
            "action": [t / 10.0 for t in range(CHUNK_SIZE)],  # 50 steps at 10fps
        },
    )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    policy = VTLADiffusionPolicy(
        config       = dataset.meta.get_policy_config(),
        univtac_ckpt = UNIVTAC_CKPT,
    ).cuda()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    for epoch in range(NUM_EPOCHS):
        policy.train()
        for batch in train_loader:
            batch = {k: v.cuda() if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            loss_dict = policy(batch)
            loss      = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        print(f"Epoch {epoch:03d} | loss {loss.item():.4f}")
        if epoch % 20 == 0:
            torch.save(policy.state_dict(), f"checkpoints/vtla_epoch_{epoch}.ckpt")


if __name__ == "__main__":
    main()