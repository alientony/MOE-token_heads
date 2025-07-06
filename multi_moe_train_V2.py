#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Multi-token MoE 
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
try:
    from trl import PPOTrainer, PPOConfig
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False

BACKBONE_DEV = torch.device("cuda:0")
HEAD_DEV     = torch.device("cuda:1")

def _disable_dataparallel_temporarily():
    original = torch.cuda.device_count
    torch.cuda.device_count = lambda: 1  # type: ignore
    return original

class Router(nn.Module):
    def __init__(self, d: int, H: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, H),
        )
    def forward(self, h):
        return self.net(h)

class ProjectionHead(nn.Module):
    def __init__(self, d: int, V: int):
        super().__init__()
        self.proj = nn.Linear(d, V, bias=False)
    def forward(self, h):
        return self.proj(h)

class MultiTokenMoE(nn.Module):
    def __init__(self, backbone: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 H: int, k_max: int):
        super().__init__()
        self.tok   = tokenizer
        self.H     = H
        self.k_max = k_max

        print(f"Loading backbone to {BACKBONE_DEV}")
        self.backbone = backbone.to(BACKBONE_DEV)
        #input(f"Backbone loaded on {BACKBONE_DEV}. Press Enter to load router...")

        print(f"Loading router to {HEAD_DEV}")
        self.router = Router(backbone.config.hidden_size, H).to(HEAD_DEV)
        #input(f"Router loaded on {HEAD_DEV}. Press Enter to load heads...")

        print(f"Loading {H} projection heads to {HEAD_DEV}")
        self.heads = nn.ModuleList(
            ProjectionHead(backbone.config.hidden_size, backbone.config.vocab_size)
            .to(HEAD_DEV) for _ in range(H)
        )
        #input(f"Heads loaded on {HEAD_DEV}. Press Enter to continue...")

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.OFF = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.w_bce, self.w_lb, self.w_ent, self.w_budget = 1.0, 0.2, 0.01, 0.05

    def forward(self, input_ids, attention_mask, labels=None, teacher_mask=None):
        B = input_ids.size(0)
        outputs = self.backbone(
            input_ids=input_ids.to(BACKBONE_DEV),
            attention_mask=attention_mask.to(BACKBONE_DEV),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden  = outputs.hidden_states[-1][:, -1]
        r_dev    = next(self.router.parameters()).device
        r_dtype  = next(self.router.parameters()).dtype
        hidden   = hidden.to(device=r_dev, dtype=r_dtype, non_blocking=True)

        gate_logits = self.router(hidden)
        if self.training and teacher_mask is not None:
            active = teacher_mask.bool().to(r_dev)
        else:
            topk   = gate_logits.topk(self.k_max, dim=-1).indices
            active = F.one_hot(topk, self.H).sum(2).bool()

        V      = self.backbone.config.vocab_size
        logits = torch.zeros((B, self.H, V),
                             device=r_dev, dtype=r_dtype)
        for h_idx, head in enumerate(self.heads):
            logits[:, h_idx] = head(hidden)

        loss = None
        if labels is not None:
            logits = logits.float()
            labels = labels.to(r_dev, non_blocking=True)
            if teacher_mask is not None:
                teacher_mask = teacher_mask.to(r_dev, non_blocking=True)
            mask_flat = active.view(B*self.H)
            logits_f  = logits.view(B*self.H, V)[mask_flat]
            labels_f  = labels.view(B*self.H)[mask_flat]
            ce        = F.cross_entropy(logits_f, labels_f,
                                        ignore_index=self.OFF)
            tgt_mask  = (teacher_mask.float()
                         if teacher_mask is not None else active.float())
            bce       = F.binary_cross_entropy_with_logits(
                            gate_logits, tgt_mask)
            freq      = tgt_mask.mean(0)
            lb        = ((freq - 1.0/self.H)**2).sum()
            p         = torch.softmax(gate_logits, dim=-1)
            ent       = -(p * torch.log(p + 1e-9)).sum(-1).mean()
            budget    = active.float().mean()
            loss      = (ce
                         + self.w_bce*bce
                         + self.w_lb*lb
                         + self.w_ent*ent
                         + self.w_budget*budget)
        return {
            "logits": logits,
            "active": active,
            "loss": loss,
            "gate_logits": gate_logits,
        }

    @torch.no_grad()
    def generate_multi(self, prompt: str, max_tokens: int = 50):
        ids       = self.tok(prompt, return_tensors="pt").input_ids
        seq       = ids.to(BACKBONE_DEV).clone()
        outs: List[int] = []
        off_streak = 0
        while len(outs) < max_tokens:
            mask = torch.ones_like(seq).to(BACKBONE_DEV)
            res  = self.forward(seq, mask)
            active = res["active"][0]
            off_streak = off_streak + 1 if active.sum() == 0 else 0
            if off_streak >= 2:
                break
            logits = res["logits"][0]
            for h in range(self.H):
                if active[h]:
                    tid = int(logits[h].argmax())
                    outs.append(tid)
                    seq = torch.cat([
                        seq, torch.tensor([[tid]],
                                          device=BACKBONE_DEV)
                    ], dim=1)
        return self.tok.decode(outs, skip_special_tokens=True)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.backbone.save_pretrained(save_directory)
        self.backbone.config.save_pretrained(save_directory)
        torch.save(self.router.state_dict(),
                   os.path.join(save_directory, "router.pt"))
        torch.save([h.state_dict() for h in self.heads],
                   os.path.join(save_directory, "heads.pt"))
        with open(os.path.join(save_directory, "moe_meta.json"), "w") as f:
            json.dump({"H": self.H, "k_max": self.k_max}, f)

    @classmethod
    def from_pretrained(cls, save_directory: str, quant: str="none"):
        with open(os.path.join(save_directory, "moe_meta.json")) as f:
            meta = json.load(f)
        H, k_max = meta["H"], meta["k_max"]
        tok = AutoTokenizer.from_pretrained(save_directory)
        bnb = None
        if quant in {"8bit", "4bit"}:
            bnb = BitsAndBytesConfig(
                load_in_8bit   = quant=="8bit",
                load_in_4bit   = quant=="4bit",
                llm_int8_threshold    = 6.0,
                bnb_4bit_compute_dtype= torch.float16,
            )
        backbone = AutoModelForCausalLM.from_pretrained(
            save_directory,
            quantization_config=bnb,
            device_map=None,
        )
        model = cls(backbone, tok, H, k_max)
        model.router.load_state_dict(
            torch.load(os.path.join(save_directory, "router.pt"))
        )
        heads_sd = torch.load(os.path.join(save_directory, "heads.pt"))
        for h, sd in zip(model.heads, heads_sd):
            h.load_state_dict(sd)
        return model

class MoETrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        outputs = model(**inputs)
        loss    = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

class QADataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tok: AutoTokenizer, H: int):
        self.samples = [json.loads(l) for l in Path(path).open()]
        self.tok, self.H = tok, H
        self.OFF = tok.pad_token_id or tok.eos_token_id
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        q, r = self.samples[idx]["question"], self.samples[idx]["response"]
        enc  = self.tok(q, truncation=True, padding=False)
        dec  = self.tok(r, add_special_tokens=False)
        labels = torch.full((self.H,), self.OFF, dtype=torch.long)
        tmask  = torch.zeros((self.H,), dtype=torch.long)
        for j, tid in enumerate(dec["input_ids"]):
            h = j % self.H
            labels[h] = tid
            tmask[h]  = 1
        return dict(
            input_ids      = torch.tensor(enc["input_ids"],
                                          dtype=torch.long),
            attention_mask = torch.ones(len(enc["input_ids"]),
                                        dtype=torch.long),
            labels         = labels,
            teacher_mask   = tmask,
        )

class Collator(DataCollatorWithPadding):
    def __init__(self, tok: AutoTokenizer, tf_prob: float):
        super().__init__(tok, pad_to_multiple_of=8)
        self.tf_prob = tf_prob
    def __call__(self, batch):
        base   = super().__call__([
            {k: x[k] for k in ("input_ids","attention_mask")}
            for x in batch
        ])
        labels = torch.stack([x["labels"] for x in batch])
        mask   = torch.stack([x["teacher_mask"] for x in batch])
        if self.tf_prob < 1.0:
            drop = torch.rand_like(mask.float()) > self.tf_prob
            mask = mask.masked_fill(drop.bool(), 0)
        base.update(labels=labels, teacher_mask=mask)
        return base

def supervised_train(model: MultiTokenMoE, dataset: QADataset,
                     out_dir: str, tf_start: float, tf_end: float,
                     k_start: int, k_end: int,
                     epochs: int, batch: int, lr: float):

    _orig = _disable_dataparallel_temporarily()
    try:
        for ep in range(epochs):
            model.k_max = int(k_start - (k_start - k_end)*ep
                              / max(epochs-1,1))
            tf_prob     = tf_start - (tf_start - tf_end)*ep \
                          / max(epochs-1,1)
            coll        = Collator(dataset.tok, tf_prob)

            args = TrainingArguments(
                out_dir,
                num_train_epochs=1,
                per_device_train_batch_size=batch,
                learning_rate=lr,
                logging_steps=50,
                report_to="none",
                save_strategy="steps",
                save_steps=5000,
                save_total_limit=5,
            )
            trainer = MoETrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                data_collator=coll
            )
            # find latest checkpoint
            ckpt_dirs = [d for d in os.listdir(out_dir)
                         if d.startswith("checkpoint-")]
            if ckpt_dirs:
                ckpt_dirs = sorted(ckpt_dirs,
                    key=lambda x: int(x.split("-")[1]))
                resume_ckpt = os.path.join(out_dir, ckpt_dirs[-1])
            else:
                resume_ckpt = None
            trainer.train(resume_from_checkpoint=resume_ckpt)
    finally:
        torch.cuda.device_count = _orig

    model.save_pretrained(out_dir)

def ppo_finetune(model: MultiTokenMoE, dataset: QADataset,
                 steps: int, batch: int, lr: float):
    if not _TRL_AVAILABLE:
        print("[WARN] trl not installed – skipping PPO stage.")
        return

    import os
    from trl import PPOTrainer, PPOConfig

    # Monkey-patch PPOTrainer to accept the old `tokenizer=` kwarg
    _orig_init = PPOTrainer.__init__
    def _patched_init(self, args, model, ref_model=None, tokenizer=None, **kwargs):
        if tokenizer is not None:
            # map `tokenizer` to the new `processing_class` argument
            kwargs["processing_class"] = tokenizer
        return _orig_init(self,
                          args=args,
                          model=model,
                          ref_model=ref_model,
                          **kwargs)
    PPOTrainer.__init__ = _patched_init

    # prepare log directory
    log_dir = "ppo_logs"
    os.makedirs(log_dir, exist_ok=True)

    # build PPO config
    cfg = PPOConfig(
        batch_size=batch,
        learning_rate=lr,
        output_dir=log_dir
    )

    # initialize trainer exactly as before
    ppo = PPOTrainer(cfg, model, tokenizer=dataset.tok)

    # run PPO episodes
    for _ in range(0, steps, batch):
        samples   = random.sample(dataset.samples, batch)
        prompts   = [s["question"] for s in samples]
        generated = [model.generate_multi(p, max_tokens=60) for p in prompts]
        rewards   = [
            1.0 if g.strip() in s["response"] else -0.5
            for g, s in zip(generated, samples)
        ]
        ppo.step(prompts, generated, rewards)

    # save final PPO-tuned model
    model.save_pretrained("ppo_final")


def parse_args():
    p   = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t   = sub.add_parser("train")
    f   = sub.add_parser("infer")

    t.add_argument("--model")
    t.add_argument("--data")
    t.add_argument("--out")
    t.add_argument("--heads",   type=int,   default=3)
    t.add_argument("--k_max",   type=int,   default=2)
    t.add_argument("--batch",   type=int,   default=3)
    t.add_argument("--lr",      type=float, default=2e-4)
    t.add_argument("--stage1_epochs",
                  type=int,    default=2)
    t.add_argument("--stage2_epochs",
                  type=int,    default=2)
    t.add_argument("--ppo_steps",
                  type=int,    default=0)
    t.add_argument("--quant",
                  choices=["none","8bit","4bit"],
                  default="none")

    f.add_argument("--ckpt")
    f.add_argument("--prompt")
    f.add_argument("--max_tokens",
                  type=int,    default=60)
    f.add_argument("--quant",
                  choices=["none","8bit","4bit"],
                  default="none")
    return p.parse_args()

def main():
    args = parse_args()

    if args.cmd == "train":
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        bnb = None
        if args.quant in {"8bit","4bit"}:
            bnb = BitsAndBytesConfig(
                load_in_8bit   = args.quant=="8bit",
                load_in_4bit   = args.quant=="4bit",
                llm_int8_threshold    = 6.0,
                bnb_4bit_compute_dtype= torch.float16,
            )

        backbone = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            device_map=None,
        )

        model   = MultiTokenMoE(backbone, tok,
                                args.heads, args.k_max)
        dataset = QADataset(args.data, tok, args.heads)

        supervised_train(
            model, dataset, f"{args.out}/s1",
            tf_start=1.0, tf_end=1.0,
            k_start=args.heads, k_end=args.heads,
            epochs=args.stage1_epochs,
            batch=args.batch, lr=args.lr,
        )

        model2 = MultiTokenMoE.from_pretrained(
            f"{args.out}/s1", quant=args.quant
        )
        supervised_train(
            model2, dataset, f"{args.out}/s2",
            tf_start=1.0, tf_end=0.0,
            k_start=args.heads, k_end=args.k_max,
            epochs=args.stage2_epochs,
            batch=args.batch, lr=args.lr,
        )

        if args.ppo_steps > 0:
            model3 = MultiTokenMoE.from_pretrained(
                f"{args.out}/s2", quant=args.quant
            )
            ppo_finetune(model3,
                         dataset,
                         args.ppo_steps,
                         args.batch,
                         lr=args.lr)

    else:  # infer
        model = MultiTokenMoE.from_pretrained(
            args.ckpt, quant=args.quant
        )
        model.eval()
        print(model.generate_multi(
            args.prompt, args.max_tokens
        ))

if __name__ == "__main__":
    main()
