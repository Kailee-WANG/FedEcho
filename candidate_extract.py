import os
import re
import json
import argparse
import pickle
from typing import List, Dict

import torch
from tqdm import tqdm


ECHONOTE_SECTIONS = [
    "PATIENT/TEST INFORMATION",
    "Indication",
    "Height",
    "Weight",
    "BSA",
    "BP",
    "HR",
    "Status",
    "INTERPRETATION",
    "Findings",
    "LEFT ATRIUM",
    "RIGHT ATRIUM/INTERATRIAL SEPTUM",
    "LEFT VENTRICLE",
    "RIGHT VENTRICLE",
    "AORTA",
    "AORTIC VALVE",
    "MITRAL VALVE",
    "TRICUSPID VALVE",
    "PERICARDIUM",
    "Conclusions",
]

ECHONOTE_TO_ECHOPRIME = {
    "LEFT VENTRICLE":   "Left Ventricle",
    "RIGHT VENTRICLE":  "Right Ventricle",
    "LEFT ATRIUM":      "Left Atrium",
    "RIGHT ATRIUM/INTERATRIAL SEPTUM": "Right Atrium",
    "AORTIC VALVE":     "Aortic Valve",
    "MITRAL VALVE":     "Mitral Valve",
    "TRICUSPID VALVE":  "Tricuspid Valve",
    "PERICARDIUM":      "Pericardium",
    "AORTA":            "Aorta",
}


def parse_args():
    p = argparse.ArgumentParser(description="Build echonote candidate pool")
    p.add_argument("--data_dir", type=str, default="/home/kaile/Echo/data/echonotes",
                    help="echonote data directory")
    p.add_argument("--train_file", type=str, default=None,
                    help="training CSV path; auto-search in data_dir if None")
    p.add_argument("--report_col", type=str, default="text",
                    help="CSV column for full report text")
    p.add_argument("--section_cols", type=str,
                    default="patient_info,interpretation,conclusion",
                    help="section column names, comma separated")
    p.add_argument("--output_dir", type=str, default="model_data/candidates_data",
                    help="output directory for .pt and .pkl files")
    p.add_argument("--output_prefix", type=str, default="echonote",
                    help="file name prefix")
    p.add_argument("--sections_json", type=str, default=None,
                    help="custom echonote sections JSON; if None use built-in list")
    p.add_argument("--text_encoder_ckpt", type=str,
                    default="model_data/weights/echo_prime_text_encoder.pt",
                    help="EchoPrimeTextEncoder checkpoint")
    p.add_argument("--lora_dir", type=str, default=None,
                    help="LoRA adapter directory (optional)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_encode", action="store_true",
                    help="skip encoding, only save reports and sections")
    return p.parse_args()


def build_report_text(row, report_col: str, section_cols: List[str]) -> str:
    if report_col in row.index and isinstance(row.get(report_col), str) and row[report_col].strip():
        return row[report_col].strip()
    parts = [str(row.get(c, "")).strip() for c in section_cols
             if c in row.index and str(row.get(c, "")).strip()]
    return "\n\n".join(parts) if parts else ""


def load_echonote_reports(
    data_dir: str,
    train_file: str | None,
    report_col: str,
    section_cols: str,
) -> List[str]:
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
            raise FileNotFoundError(
                f"No training CSV found in {data_dir}, please set --train_file"
            )
    df = pd.read_csv(csv_path)
    section_list = [c.strip() for c in section_cols.split(",") if c.strip()]
    reports = []
    for _, row in df.iterrows():
        text = build_report_text(row, report_col, section_list)
        if text:
            reports.append(text)
    return reports


def extract_section_echonote(report: str, header: str, all_headers: List[str]) -> str:
    pattern = (
        r"(?m)^"
        + re.escape(header)
        + r".*?(?=^("
        + "|".join(map(re.escape, all_headers))
        + r")|\Z)"
    )
    m = re.search(pattern, report, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    idx = report.find(header)
    if idx == -1:
        return ""
    next_pos = len(report)
    for h in all_headers:
        if h == header:
            continue
        j = report.find(h, idx + len(header))
        if j != -1 and j < next_pos:
            next_pos = j
    return report[idx:next_pos].strip()


def detect_sections_from_reports(
    reports: List[str],
    candidate_headers: List[str],
    min_ratio: float = 0.01,
) -> List[str]:
    counts = {h: 0 for h in candidate_headers}
    for report in reports:
        for h in candidate_headers:
            if h in report:
                counts[h] += 1
    n = len(reports)
    return [h for h in candidate_headers if counts[h] / max(n, 1) >= min_ratio]


def encode_reports(
    reports: List[str],
    text_encoder_ckpt: str,
    lora_dir: str | None,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    from echo_prime import EchoPrimeTextEncoder

    text_encoder = EchoPrimeTextEncoder(device=device)
    if os.path.isfile(text_encoder_ckpt):
        ckpt = torch.load(text_encoder_ckpt, map_location=device, weights_only=False)
        if hasattr(ckpt, "state_dict"):
            text_encoder.load_state_dict(ckpt.state_dict(), strict=True)
        else:
            text_encoder.load_state_dict(ckpt, strict=True)
    if lora_dir:
        from peft import PeftModel
        text_encoder.backbone = PeftModel.from_pretrained(
            text_encoder.backbone, lora_dir
        )
        for proj_name in ["fedtextlora_text_projection.pt", "text_projection.pt"]:
            proj_path = os.path.join(lora_dir, proj_name)
            if os.path.isfile(proj_path):
                text_encoder.text_projection.load_state_dict(
                    torch.load(proj_path, map_location=device, weights_only=True)
                )
                break
    text_encoder.eval()

    embs = []
    for i in tqdm(range(0, len(reports), batch_size), desc="Encoding"):
        batch = reports[i : i + batch_size]
        enc = text_encoder.tokenizer(
            batch,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = text_encoder.backbone(
                **enc, output_hidden_states=True
            ).hidden_states[-1][:, 0, :]
            e = text_encoder.text_projection(hidden)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.output_dir, exist_ok=True)

    reports = load_echonote_reports(
        args.data_dir, args.train_file, args.report_col, args.section_cols
    )
    print(f"Loaded {len(reports)} echonote reports")

    if args.sections_json and os.path.isfile(args.sections_json):
        with open(args.sections_json, encoding="utf-8") as f:
            sections = json.load(f)
        print(f"Loaded {len(sections)} sections from {args.sections_json}")
    else:
        sections = ECHONOTE_SECTIONS

    detected = detect_sections_from_reports(reports, sections)
    print(f"Detected {len(detected)} sections present in reports: {detected}")

    reports_pkl_path = os.path.join(
        args.output_dir, f"{args.output_prefix}_candidate_reports.pkl"
    )
    with open(reports_pkl_path, "wb") as f:
        pickle.dump(reports, f)
    print(f"Saved {len(reports)} reports -> {reports_pkl_path}")

    sections_json_path = os.path.join(
        args.output_dir, f"{args.output_prefix}_sections.json"
    )
    sections_data = {
        "all_sections": sections,
        "detected_sections": detected,
        "section_mapping_to_echoprime": ECHONOTE_TO_ECHOPRIME,
    }
    with open(sections_json_path, "w", encoding="utf-8") as f:
        json.dump(sections_data, f, indent=2, ensure_ascii=False)
    print(f"Saved sections -> {sections_json_path}")

    if args.no_encode:
        print("Skipped encoding (--no_encode). Done.")
        return

    embs = encode_reports(
        reports,
        text_encoder_ckpt=args.text_encoder_ckpt,
        lora_dir=args.lora_dir,
        device=device,
        batch_size=args.batch_size,
    )
    embs_path = os.path.join(
        args.output_dir, f"{args.output_prefix}_candidate_embeddings.pt"
    )
    torch.save(embs, embs_path)
    print(f"Saved embeddings {embs.shape} -> {embs_path}")

    print(
        f"\nTo use in EchoPrime:\n"
        f"  ep.candidate_embeddings = torch.load('{embs_path}')\n"
        f"  ep.candidate_reports = pickle.load(open('{reports_pkl_path}', 'rb'))\n"
        f"  report = ep.generate_report_echonote(study_emb)"
    )


if __name__ == "__main__":
    main()
