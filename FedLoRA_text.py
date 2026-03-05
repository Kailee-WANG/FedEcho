
import os
import random
import json
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description="FedEcho")
    
    parser.add_argument("--data_dir", type=str,
                        default="/home/kaile/Echo/data/echonotes",
                        help="echonote data directory")
    parser.add_argument("--train_file", type=str, default=None,
                        help="training CSV path; auto-search in data_dir if None")
    parser.add_argument("--report_col", type=str, default="text",
                        help="CSV column name for full report text")
    parser.add_argument("--section_cols", type=str,
                        default="patient_info,interpretation,conclusion",
                        help="section column names, comma separated")
   
    parser.add_argument("--num_clients", type=int, default=4, help="number of clients")
    parser.add_argument("--reports_per_client", type=int, default=500,
                        help="number of reports per client")
    parser.add_argument("--global_rounds", type=int, default=3, help="global rounds of federated learning")
    parser.add_argument("--local_epochs", type=int, default=1, help="local training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    
    parser.add_argument("--max_length", type=int, default=512,
                        help="tokenizer max length")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="MLM mask probability")
    
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="query,value",
                        help="LoRA target modules, comma separated")
    
    parser.add_argument("--text_encoder_ckpt", type=str,
                        default="model_data/weights/echo_prime_text_encoder.pt",
                        help="EchoPrimeTextEncoder pretrained checkpoint path")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="device (cuda/cpu), default auto-select")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader num_workers")
    parser.add_argument("--checkpoint_dir", type=str, default="federated_checkpoints",
                        help="checkpoint save directory")
    parser.add_argument("--candidate_pool_sizes", type=str, default=None,
                        help="per-client candidate pool sizes, comma separated; if None, d_i=n_i")
    return parser.parse_args()


def build_report_text(row, report_col: str, section_cols: List[str]):
    if report_col in row.index and isinstance(row.get(report_col), str) and row[report_col].strip():
        return row[report_col].strip()
    parts = [str(row.get(c, "")).strip() for c in section_cols if c in row.index and str(row.get(c, "")).strip()]
    return "\n\n".join(parts) if parts else ""


def load_echonote_reports(data_dir,train_file,report_col,section_cols,limit= None):
    import pandas as pd
    data_dir = os.path.abspath(data_dir)
    if train_file:
        csv_path = os.path.abspath(train_file)
    else:
        for name in ["EchoReports.csv", "train.csv", "reports.csv", "data.csv"]:
            cand = os.path.join(data_dir, name)
            if os.path.isfile(cand):
                csv_path = cand
                break
        else:
            raise FileNotFoundError(f"No training CSV found in {data_dir}, please set --train_file")
    df = pd.read_csv(csv_path)
    section_list = [c.strip() for c in section_cols.split(",") if c.strip()]
    reports = []
    for _, row in df.iterrows():
        text = build_report_text(row, report_col, section_list)
        if text:
            reports.append(text)
        if limit is not None and len(reports) >= limit:
            break
    return reports


def split_clients(reports,num_clients,per_client):
    random.shuffle(reports)
    needed = num_clients * per_client
    if len(reports) < needed:
        raise ValueError(f"Insufficient reports: need {needed}, got {len(reports)}")
    return {
        cid: reports[cid * per_client : (cid + 1) * per_client]
        for cid in range(num_clients)
    }


class EchonoteReportDataset(Dataset):
    def __init__(self, reports: List[str]):
        self.reports = reports

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, i):
        return {"text": self.reports[i]}


def mlm_collate(batch, tokenizer, device, max_length, mlm_probability):
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"].clone()
    labels = input_ids.clone()
    special_ids = set(tokenizer.all_special_ids)
    mask = torch.rand(labels.shape, device=device) < mlm_probability
    for sid in special_ids:
        mask[labels == sid] = False
    labels[~mask] = -100
    mask_token_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    r = torch.rand(labels.shape, device=device)
    input_ids[mask] = mask_token_id
    input_ids[mask & (r > 0.8)] = labels[mask & (r > 0.8)].clamp(0, vocab_size - 1)
    random_sel = mask & (r > 0.9)
    if random_sel.any():
        input_ids[random_sel] = torch.randint(0, vocab_size, (random_sel.sum().item(),), device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def build_text_encoder_with_lora(device,text_encoder_ckpt,lora_r= 8,lora_alpha= 16,lora_dropout= 0.05,lora_target_modules= None):
    from echo_prime import EchoPrimeTextEncoder

    if lora_target_modules is None:
        lora_target_modules = ["query", "value"]

    model = EchoPrimeTextEncoder(device=device)
    if os.path.isfile(text_encoder_ckpt):
        ckpt = torch.load(text_encoder_ckpt, map_location=device, weights_only=False)
        if hasattr(ckpt, "state_dict"):
            model.load_state_dict(ckpt.state_dict(), strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
    )
    model.backbone = get_peft_model(model.backbone, lora_config)
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    return model


@dataclass
class ClientConfig:
    id: int
    reports: List[str]


class Client:
    def __init__(self,cfg,global_model,device,batch_size,max_length,mlm_probability,num_workers = 0,text_encoder_ckpt = "",lora_r= 8,lora_alpha= 16,lora_dropout = 0.05,lora_target_modules= None):
        self.id = cfg.id
        self.reports = cfg.reports
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.num_workers = num_workers

        self.model = build_text_encoder_with_lora(
            device,
            text_encoder_ckpt=text_encoder_ckpt,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        self.model.load_state_dict(global_model.state_dict(), strict=False)

    def _collate_fn(self, batch):
        return mlm_collate(
            batch,
            self.model.tokenizer,
            self.device,
            self.max_length,
            self.mlm_probability,
        )

    def local_update(self, epochs, lr):
        self.model.train()
        dataset = EchonoteReportDataset(self.reports)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=len(dataset) >= self.batch_size,
        )
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
        )
        total_loss, num_batches = 0.0, 0
        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"Client {self.id} Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                out = self.model.backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / max(1, num_batches)
        upload_state = {}
        for name, tensor in self.model.state_dict().items():
            if name.startswith("backbone.") and "lora" in name.lower():
                upload_state[name] = tensor.cpu()
        return upload_state, avg_loss

    def load_backbone_from_server(self, global_model):
        global_sd = global_model.state_dict()
        local_sd = self.model.state_dict()
        for k, v in global_sd.items():
            if k.startswith("backbone."):
                local_sd[k] = v
        self.model.load_state_dict(local_sd, strict=False)


class Server:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self,client_states,n_list,d_list):
        if not client_states:
            return
        sum_n = sum(n_list)
        sum_d = sum(d_list)
        weights = [
            0.5 * (d_i / sum_d + n_i / sum_n)
            for n_i, d_i in zip(n_list, d_list)
        ]
        agg_state = {}
        for k in client_states[0].keys():
            stacked = torch.stack([cs[k] for cs in client_states], dim=0)
            agg_state[k] = sum(w * s for w, s in zip(weights, stacked))
        self.global_model.load_state_dict(agg_state, strict=False)


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    lora_target_modules = [s.strip() for s in args.lora_target_modules.split(",")]
    cand_pool_sizes = None
    if args.candidate_pool_sizes:
        cand_pool_sizes = [int(x.strip()) for x in args.candidate_pool_sizes.split(",")]
        if len(cand_pool_sizes) != args.num_clients:
            raise ValueError(
                f"--candidate_pool_sizes must provide {args.num_clients} values, got {len(cand_pool_sizes)}"
            )

    print(f"Device: {device}")
    print(f"Config: {args.num_clients} clients x {args.reports_per_client} reports, "
          f"batch_size={args.batch_size}, max_length={args.max_length}, "
          f"mlm_prob={args.mlm_probability}, "
          f"global_rounds={args.global_rounds}, local_epochs={args.local_epochs}, lr={args.lr}")

    needed = args.num_clients * args.reports_per_client
    reports = load_echonote_reports(
        data_dir=args.data_dir,
        train_file=args.train_file,
        report_col=args.report_col,
        section_cols=args.section_cols,
        limit=needed,
    )
    if len(reports) < needed:
        raise SystemExit(f"Insufficient reports: need {needed}, got {len(reports)}")
    clients_data = split_clients(reports, args.num_clients, args.reports_per_client)
    print(f"Total reports: {len(reports)}, {args.num_clients} clients, "
          f"{args.reports_per_client} per client")

    global_model = build_text_encoder_with_lora(
        device,
        text_encoder_ckpt=args.text_encoder_ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    server = Server(global_model)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(
        args.checkpoint_dir,
        f"echo_text_lora_fed_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    logs = {"rounds": [], "config": vars(args)}

    for rnd in range(args.global_rounds):
        print(f"\n=== Round {rnd + 1}/{args.global_rounds} ===")
        client_states = []
        round_log = {"round": rnd + 1, "client_losses": {}}

        n_list = []
        for cid in range(args.num_clients):
            cfg = ClientConfig(id=cid, reports=clients_data[cid])
            client = Client(
                cfg,
                global_model=server.global_model,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                mlm_probability=args.mlm_probability,
                num_workers=args.num_workers,
                text_encoder_ckpt=args.text_encoder_ckpt,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target_modules=lora_target_modules,
            )
            client.load_backbone_from_server(server.global_model)
            state, avg_loss = client.local_update(args.local_epochs, args.lr)
            client_states.append(state)
            n_list.append(len(clients_data[cid]))
            round_log["client_losses"][str(cid)] = avg_loss
            print(f"  Client {cid} avg loss: {avg_loss:.4f}")
        d_list = cand_pool_sizes if cand_pool_sizes else n_list
        server.aggregate(client_states, n_list, d_list)
        logs["rounds"].append(round_log)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"\nFederated training finished, logs: {log_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.checkpoint_dir, f"fedtextlora_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    global_model.backbone.save_pretrained(out_dir)
    torch.save(
        global_model.text_projection.state_dict(),
        os.path.join(out_dir, "fedtextlora_text_projection.pt"),
    )
    print(f"Global LoRA and text_projection saved: {out_dir}")


if __name__ == "__main__":
    main()
