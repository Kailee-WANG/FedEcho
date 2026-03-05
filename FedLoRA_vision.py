
import os
import random
import json
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm

from peft import LoraConfig, get_peft_model, TaskType
from echo_prime import EchoPrime
import utils




def parse_args():
    parser = argparse.ArgumentParser(description="FedEcho")
    
    parser.add_argument("--video_root", type=str,
                        default="/data/EchoNet-Dynamic/Videos",
                        help="root path of videos")
    parser.add_argument("--filelist_csv", type=str,
                        default="/data/EchoNet-Dynamic/FileList.csv",
                        help="FileList.csv")
    parser.add_argument("--filename_col", type=str, default="FileName",
                        help="FileList column name of filename")
    parser.add_argument("--ef_col", type=str, default="EF",
                        help="FileList EF column name")
   
    parser.add_argument("--num_clients", type=int, default=4, help="number of clients")
    parser.add_argument("--videos_per_client", type=int, default=500,
                        help="number of videos per client")
    parser.add_argument("--global_rounds", type=int, default=2, help="global rounds of federated learning")
    parser.add_argument("--local_epochs", type=int, default=1, help="local training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--n_views", type=int, default=2,
                        help="number of views per video")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="train ratio per client")
    
    parser.add_argument("--frames_to_take", type=int, default=32,
                        help="number of frames to take (before stride)")
    parser.add_argument("--video_size", type=int, default=224, help="video height and width")
    parser.add_argument("--frame_stride", type=int, default=2, help="frame stride")
    
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str,
                        default="qkv,project.0,mlp.0,mlp.3",
                        help="LoRA target modules, comma separated")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="device (cuda/cpu), default auto-select")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader num_workers")
    parser.add_argument("--checkpoint_dir", type=str, default="federated_checkpoints",
                        help="checkpoint save directory")
    return parser.parse_args()




def load_echonet_ef_samples(
    video_root,
    filelist_csv,
    filename_col,
    ef_col: str,
):
    df = pd.read_csv(filelist_csv)
    if filename_col not in df.columns:
        raise ValueError(f"FileList.csv does not contain column '{filename_col}', current columns: {list(df.columns)}")
    if ef_col not in df.columns:
        raise ValueError(f"FileList.csv does not contain column '{ef_col}', current columns: {list(df.columns)}")

    samples = []
    for _, row in df.iterrows():
        fname = str(row[filename_col]).strip()
        ef = row[ef_col]
        if pd.isna(ef) or not fname:
            continue
        base = os.path.join(video_root, fname)
        video_path = None
        for suf in ["", ".avi", ".AVI", ".mp4", ".MP4"]:
            p = base + suf
            if os.path.isfile(p):
                video_path = p
                break
        if video_path is None:
            continue
        samples.append((video_path, float(ef)))

    if len(samples) == 0:
        raise ValueError("no valid (video, EF) samples found, please check path and column names.")
    return samples


def split_clients_iid(samples,num_clients,per_client):
    random.shuffle(samples)
    needed = num_clients * per_client
    if len(samples) < needed:
        raise ValueError(f"not enough samples, {len(samples)} < {needed}")
    return {
        cid: samples[cid * per_client : (cid + 1) * per_client]
        for cid in range(num_clients)
    }


class EchoNetEFDataset(Dataset):
    def __init__(
        self,samples,mean,std,frames_to_take,video_size,frame_stride):
        self.samples = samples
        self.mean = mean
        self.std = std
        self.frames_to_take = frames_to_take
        self.video_size = video_size
        self.frame_stride = frame_stride

    def __len__(self):
        return len(self.samples)

    def _load_and_preprocess(self, path):
        pixels, _, _ = torchvision.io.read_video(path)
        pixels = np.array(pixels)
        x = np.zeros((len(pixels), self.video_size, self.video_size, 3))
        for i in range(len(x)):
            x[i] = utils.crop_and_scale(pixels[i])
        x = torch.as_tensor(x, dtype=torch.float).permute(3, 0, 1, 2)
        x.sub_(self.mean).div_(self.std)
        if x.shape[1] < self.frames_to_take:
            pad = torch.zeros(
                (3, self.frames_to_take - x.shape[1], self.video_size, self.video_size),
                dtype=torch.float,
            )
            x = torch.cat((x, pad), dim=1)
        x = x[:, : self.frames_to_take : self.frame_stride, :, :]
        return x

    def __getitem__(self, idx):
        path, ef = self.samples[idx]
        vid = self._load_and_preprocess(path)
        return vid, torch.tensor(ef / 100.0, dtype=torch.float32)


class EchoVideoEFModel(nn.Module):
    def __init__(self,device,lora_r= 8,lora_alpha= 16,lora_dropout= 0.05,lora_target_modules = None):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["qkv", "project.0", "mlp.0", "mlp.3"]
        ep = EchoPrime(device=device)
        self.mean = ep.mean.cpu()
        self.std = ep.std.cpu()

        base_encoder = ep.echo_encoder
        for p in base_encoder.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
        )
        self.encoder = get_peft_model(base_encoder, lora_config)
        self.head = nn.Linear(512, 1)
        self.encoder.to(device)
        self.head.to(device)

        for n, p in self.encoder.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.base_model(x)
        return self.head(feats).squeeze(-1)


def _multiview_forward(model,vids,n_views):
    B = vids.size(0)
    vids_mv = vids.repeat_interleave(n_views, dim=0)
    pred_mv = model(vids_mv)
    return pred_mv.view(B, n_views).mean(dim=1)


@dataclass
class ClientConfig:
    id=None
    train_samples=None
    test_samples=None


class Client:
    def __init__(self,cfg,base_model,device,batch_size,n_views,num_workers = 0,frames_to_take= 32,video_size= 224,frame_stride= 2,lora_r= 8,lora_alpha= 16,lora_dropout= 0.05,lora_target_modules= None):
        self.id = cfg.id
        self.train_samples = cfg.train_samples
        self.test_samples = cfg.test_samples
        self.device = device
        self.n_views = n_views

        if lora_target_modules is None:
            lora_target_modules = ["qkv", "project.0", "mlp.0", "mlp.3"]
        self.model = EchoVideoEFModel(
            self.device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        self.model.load_state_dict(base_model.state_dict())

        dataset = EchoNetEFDataset(
            self.train_samples,
            mean=self.model.mean,
            std=self.model.std,
            frames_to_take=frames_to_take,
            video_size=video_size,
            frame_stride=frame_stride,
        )
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

    def load_encoder_from_server(self, global_model):
        global_sd = global_model.state_dict()
        local_sd = self.model.state_dict()
        for k, v in global_sd.items():
            if k.startswith("encoder."):
                local_sd[k] = v
        self.model.load_state_dict(local_sd)

    def local_update(self, epochs, lr):
        self.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
        )
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            pbar = tqdm(
                self.loader,
                desc=f"Client {self.id} Epoch {epoch + 1}/{epochs}",
            )
            for vids, ef in pbar:
                vids = vids.to(self.device)
                ef = ef.to(self.device)

                pred = _multiview_forward(self.model, vids, self.n_views)
                loss = loss_fn(pred, ef)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, num_batches)

        upload_state = {}
        for name, tensor in self.model.state_dict().items():
            if name.startswith("encoder.") and "lora" in name.lower():
                upload_state[name] = tensor.cpu()
        return upload_state, avg_loss


class Server:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_states):
        if not client_states:
            return
        avg_state = {}
        for k in client_states[0]:
            avg_state[k] = torch.stack([cs[k] for cs in client_states]).mean(dim=0)
        global_sd = self.global_model.state_dict()
        global_sd.update(avg_state)
        self.global_model.load_state_dict(global_sd)


@torch.no_grad()
def evaluate_model(model,samples,device,batch_size,n_views,num_workers = 0,frames_to_take= 32,video_size= 224,frame_stride= 2):
    if not samples:
        return {}

    dataset = EchoNetEFDataset(
        samples,
        mean=model.mean,
        std=model.std,
        frames_to_take=frames_to_take,
        video_size=video_size,
        frame_stride=frame_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model.eval()
    all_pred, all_true = [], []
    for vids, ef in loader:
        vids = vids.to(device)
        ef = ef.to(device)
        B = vids.size(0)

        if B < 2:
            vids = torch.cat([vids, vids], dim=0)
            pred = _multiview_forward(model, vids, n_views)
            pred = pred[:B]
        else:
            pred = _multiview_forward(model, vids, n_views)

        all_pred.append(pred.cpu())
        all_true.append(ef.cpu())

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)

    mse_norm = ((preds - trues) ** 2).mean().item()
    preds_ef = preds * 100.0
    trues_ef = trues * 100.0
    mse_ef = ((preds_ef - trues_ef) ** 2).mean().item()

    return {
        "mse_norm": mse_norm,
        "rmse_norm": mse_norm ** 0.5,
        "mse_ef": mse_ef,
        "rmse_ef": mse_ef ** 0.5,
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    lora_target_modules = [s.strip() for s in args.lora_target_modules.split(",")]

    print(f"device: {device}")
    print(f"config: {args.num_clients} clients × {args.videos_per_client} videos, "
          f"n_views={args.n_views}, batch_size={args.batch_size}, "
          f"global_rounds={args.global_rounds}, local_epochs={args.local_epochs}, lr={args.lr}")

    all_samples = load_echonet_ef_samples(
        video_root=args.video_root,
        filelist_csv=args.filelist_csv,
        filename_col=args.filename_col,
        ef_col=args.ef_col,
    )
    print(f"loaded {len(all_samples)} valid samples")
    clients_full = split_clients_iid(
        all_samples, args.num_clients, args.videos_per_client
    )

    clients_train = {}
    clients_test = {}

    for cid, samples_c in clients_full.items():
        random.shuffle(samples_c)
        n_train = max(1, int(len(samples_c) * args.train_ratio))
        clients_train[cid] = samples_c[:n_train]
        clients_test[cid] = samples_c[n_train:]
        if not clients_test[cid] and len(clients_train[cid]) > 1:
            clients_test[cid] = clients_train[cid][-1:]
            clients_train[cid] = clients_train[cid][:-1]

    print("number of training samples per client:", {c: len(v) for c, v in clients_train.items()})
    print("number of testing samples per client:", {c: len(v) for c, v in clients_test.items()})

    global_model = EchoVideoEFModel(
        device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    server = Server(global_model)

    clients = []
    for cid in range(args.num_clients):
        cfg = ClientConfig(
            id=cid,
            train_samples=clients_train[cid],
            test_samples=clients_test[cid],
        )
        clients.append(Client(
            cfg,
            base_model=server.global_model,
            device=device,
            batch_size=args.batch_size,
            n_views=args.n_views,
            num_workers=args.num_workers,
            frames_to_take=args.frames_to_take,
            video_size=args.video_size,
            frame_stride=args.frame_stride,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_target_modules,
        ))

    round_train_losses = []
    round_test_results = []

    for rnd in range(args.global_rounds):
        print(f"\n===== Global Round {rnd + 1}/{args.global_rounds} =====")
        client_states = []
        client_losses = []

        for client in clients:
            client.load_encoder_from_server(server.global_model)
            state, avg_loss = client.local_update(args.local_epochs, args.lr)
            client_states.append(state)
            client_losses.append(avg_loss)

        server.aggregate(client_states)
        round_loss = sum(client_losses) / len(client_losses)
        round_train_losses.append(round_loss)
        print(f"Round {rnd + 1} average training loss: {round_loss:.6f}")

        metrics_round = {}
        for client in clients:
            client.load_encoder_from_server(server.global_model)
            m = evaluate_model(
                client.model,
                client.test_samples,
                device=device,
                batch_size=args.batch_size,
                n_views=args.n_views,
                num_workers=args.num_workers,
                frames_to_take=args.frames_to_take,
                video_size=args.video_size,
                frame_stride=args.frame_stride,
            )
            metrics_round[client.id] = m
            if m:
                print(f"  Client {client.id} test RMSE(EF): {m['rmse_ef']:.3f}")
        round_test_results.append(metrics_round)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "lora_echoprime_fedavg.pt")
    torch.save(server.global_model.state_dict(), ckpt_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = {
        "config": vars(args),
        "round_train_losses": round_train_losses,
        "round_test_results": round_test_results,
    }
    logs_path = os.path.join(
        args.checkpoint_dir, f"lora_echoprime_fedavg_logs_{ts}.json"
    )
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"\nmodel saved: {ckpt_path}")
    print(f"logs saved: {logs_path}")


if __name__ == "__main__":
    main()
