"""Train DeBERTa to score evidence windows and predict a contract-level label."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch import nn
from transformers import __version__ as transformers_version

from .infer import DEFAULT_CHECKPOINT_PATH
from .model import ID2LABEL, LABEL2ID, TriageModel

ROOT = Path(__file__).resolve().parents[3]


def encode_document(tokenizer, text, hypothesis, max_length, stride):
    encoded = tokenizer(
        hypothesis, text, truncation="only_second", max_length=max_length,
        stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True,
    )
    windows, bounds = [], []
    for index, offsets in enumerate(encoded["offset_mapping"]):
        document_offsets = [offset for offset, part in zip(offsets, encoded.sequence_ids(index))
                            if part == 1 and offset[0] < offset[1]]
        if not document_offsets:
            raise ValueError("Document has no tokenizable text.")
        bounds.append((document_offsets[0][0], document_offsets[-1][1]))
        windows.append({key: encoded[key][index] for key in tokenizer.model_input_names
                        if key in encoded})
    return windows, bounds


def evidence_targets(text, spans, bounds):
    trimmed = []
    for start, end in spans:
        excerpt = text[start:end]
        if excerpt.strip():
            trimmed.append((start + len(excerpt) - len(excerpt.lstrip()),
                            start + len(excerpt.rstrip())))
    # Exclude partially visible positives from the evidence loss.
    return [1 if any(left <= start and end <= right for start, end in trimmed)
            else -1 if any(start < right and end > left for start, end in trimmed)
            else 0 for left, right in bounds]


class DocumentNLI(nn.Module):
    def __init__(self, backbone, window_batch):
        super().__init__()
        self.backbone = backbone
        self.evidence = nn.Linear(backbone.classifier.in_features, 1)
        self.window_batch = window_batch

    def forward(self, windows, tokenizer, device):
        representations = []
        for offset in range(0, len(windows), self.window_batch):
            batch = tokenizer.pad(windows[offset:offset + self.window_batch],
                                  padding=True, return_tensors="pt").to(device)
            hidden = self.backbone.deberta(**batch).last_hidden_state
            representations.append(self.backbone.pooler(hidden))
        pooled = torch.cat(representations)
        evidence = self.evidence(pooled).squeeze(-1)
        attention = evidence.float().softmax(dim=0).to(pooled.dtype)
        document = (attention[:, None] * pooled).sum(dim=0)
        return self.backbone.classifier(self.backbone.dropout(document)), evidence


def read_split(path):
    split = json.loads(path.read_text(encoding="utf-8"))
    ids = [doc["id"] for doc in split["documents"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate contract IDs in {path}")
    return split


def examples(split, smoke):
    for doc in split["documents"]:
        for hypothesis_id, annotation in doc["annotation_sets"][0]["annotations"].items():
            yield doc, hypothesis_id, split["labels"][hypothesis_id]["hypothesis"], annotation
        if smoke:
            return


def run_epoch(model, tokenizer, items, args, device, optimizer=None, save_progress=None, start_index=0,
              total_loss=0.0):
    training = optimizer is not None
    model.train(training)
    rows = []
    with torch.set_grad_enabled(training):
        for index in range(start_index + 1, len(items) + 1):
            doc, hypothesis_id, hypothesis, annotation = items[index - 1]
            windows, bounds = encode_document(tokenizer, doc["text"], hypothesis,
                                              args.max_length, args.stride)
            logits, evidence = model(windows, tokenizer, device)
            gold = LABEL2ID[annotation["choice"]]
            if training:
                loss = F.cross_entropy(logits[None], torch.tensor([gold], device=device))
                spans = [doc["spans"][i] for i in annotation["spans"]]
                targets = torch.tensor(evidence_targets(doc["text"], spans, bounds), device=device)
                valid = targets >= 0
                if valid.any():
                    loss += args.evidence_weight * F.binary_cross_entropy_with_logits(
                        evidence[valid], targets[valid].float(),
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss on {doc['id']}/{hypothesis_id}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item()
            else:
                best_window = evidence.argmax().item()
                rows.append({"document_id": doc["id"], "hypothesis_id": hypothesis_id,
                             "gold": annotation["choice"], "predicted": ID2LABEL[logits.argmax().item()],
                             "window_count": len(windows), "evidence_window": bounds[best_window]})
            if index == 1 or index % 200 == 0:
                print(f"{'Train' if training else 'Dev'} {index}/{len(items)}", flush=True)
            if save_progress and index % args.save_every == 0:
                save_progress(index, total_loss)
    return total_loss / len(items), rows


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_checkpoint(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "backend/data/contractnli")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--window-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--evidence-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="GPU smoke test with two training and three validation examples.")
    args = parser.parse_args()
    if args.smoke:
        args.epochs, args.save_every = 1, 1
    if not (0 <= args.stride < args.max_length <= 512 and args.window_batch > 0
            and args.epochs > 0 and args.save_every > 0):
        parser.error("Check max length, stride, window batch, and epochs.")
    if not torch.cuda.is_available():
        parser.error("Document training and smoke tests require CUDA.")
    if args.out.resolve() == args.checkpoint.parent.resolve():
        parser.error("Use a separate output directory for the document model.")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    latest = args.out / "latest_document.pt"
    if manifest_path.exists() and not args.resume:
        parser.error("Use a fresh output directory.")
    if args.resume and not (manifest_path.is_file() and latest.is_file()):
        parser.error("Resume requires an existing manifest and latest_document.pt.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_properties(device)
    print(f"GPU: {gpu.name}, {gpu.total_memory / 2**30:.1f} GiB", flush=True)
    train_path, dev_path = (args.data_dir / f"{split}.json" for split in ("train", "dev"))
    train, dev = read_split(train_path), read_split(dev_path)
    if {d["id"] for d in train["documents"]} & {d["id"] for d in dev["documents"]}:
        raise ValueError("Training and development contract IDs overlap.")
    training_items, dev_items = list(examples(train, False)), list(examples(dev, False))
    if (len(training_items), len(dev_items)) != (7191, 1037):
        raise ValueError("Unexpected train or development example count.")
    train_text = {" ".join(doc["text"].split()) for doc in train["documents"]}
    if any(" ".join(doc["text"].split()) in train_text for doc in dev["documents"]):
        raise ValueError("Identical normalized contract text across train and development.")
    wrapper = TriageModel().load(checkpoint_path=args.checkpoint, device=device)
    if args.smoke:
        for doc, hid, hypothesis, annotation in training_items:
            if not annotation["spans"]:
                continue
            windows, bounds = encode_document(wrapper._tokenizer, doc["text"], hypothesis,
                                              args.max_length, args.stride)
            spans = [doc["spans"][i] for i in annotation["spans"]]
            if len(windows) > 1 and 1 in evidence_targets(doc["text"], spans, bounds):
                positive = (doc, hid, hypothesis, annotation)
                break
        else:
            raise ValueError("No multi-window annotated smoke case.")
        negative = next(item for item in training_items
                        if item[0]["id"] == positive[0]["id"] and not item[3]["spans"])
        training_items, dev_items = [positive, negative], dev_items[:3]
    model = DocumentNLI(wrapper._model, args.window_batch).to(device)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    probe = next(model.backbone.deberta.encoder.layer[-1].parameters())
    if args.smoke:
        evidence_before = model.evidence.weight.detach().clone()
        encoder_before = probe.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    settings = {key: value for key, value in vars(args).items()
                if key not in {"resume", "out", "checkpoint", "data_dir"}}
    manifest = {"settings": settings, "backbone": wrapper.hf_checkpoint,
                "checkpoint_sha256": sha256(args.checkpoint), "train_sha256": sha256(train_path),
                "dev_sha256": sha256(dev_path), "torch": torch.__version__,
                "transformers": transformers_version, "train_examples": len(training_items),
                "dev_examples": len(dev_items), "source_sha256": sha256(Path(__file__)),
                "selection_metric": ["macro_f1", "contradiction_recall"]}
    best = (-1.0, -1.0)
    history = []
    first_epoch, first_index, first_loss = 1, 0, 0.0
    if args.resume:
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume inputs or settings differ from the saved manifest.")
        resume = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["cpu_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"], device)
        first_epoch, first_index, first_loss = resume["epoch"], resume["index"], resume["loss"]
        best, history = tuple(resume["best"]), resume["history"]
        del resume
    else:
        save_json(manifest_path, manifest)

    def persist(epoch, index, loss):
        save_checkpoint(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
                                "epoch": epoch, "index": index, "loss": loss,
                                "best": best, "history": history})

    for epoch in range(first_epoch, args.epochs + 1):
        start = time.perf_counter()
        shuffled = training_items.copy()
        random.Random(args.seed + epoch).shuffle(shuffled)
        loss, _ = run_epoch(model, wrapper._tokenizer, shuffled, args, device, optimizer,
                            lambda index, loss: persist(epoch, index, loss),
                            first_index if epoch == first_epoch else 0,
                            first_loss if epoch == first_epoch else 0.0)
        _, rows = run_epoch(model, wrapper._tokenizer, dev_items, args, device)
        metrics = classification_report([r["gold"] for r in rows], [r["predicted"] for r in rows],
                                        labels=list(LABEL2ID), output_dict=True, zero_division=0)
        accuracy = sum(row["gold"] == row["predicted"] for row in rows) / len(rows)
        macro_f1 = metrics["macro avg"]["f1-score"]
        contradiction_recall = metrics["Contradiction"]["recall"]
        history.append({"epoch": epoch, "train_loss": loss, "accuracy": accuracy,
                        "macro_f1": macro_f1, "contradiction_recall": contradiction_recall,
                        "seconds": time.perf_counter() - start})
        if args.smoke and (torch.equal(evidence_before, model.evidence.weight)
                           or torch.equal(encoder_before, probe)):
            raise RuntimeError("Smoke run did not update both encoder and evidence head.")
        if (macro_f1, contradiction_recall) > best:
            best = macro_f1, contradiction_recall
            save_checkpoint(args.out / "best_document.pt", model.state_dict())
            save_json(args.out / "dev_predictions.json", {"epoch": epoch, "accuracy": accuracy,
                      "metrics": metrics, "rows": rows})
        save_json(args.out / "history.json", history)
        persist(epoch + 1, 0, 0.0)
        print(history[-1], flush=True)


if __name__ == "__main__":
    main()
