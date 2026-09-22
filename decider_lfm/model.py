"""Backbone -> slot hidden states -> restricted logits over option letters."""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from decider_lfm.prompt import letter_ids


class DecisionModel(nn.Module):
    def __init__(self, name, dtype=torch.bfloat16, grad_ckpt=False):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        # the readout needs these two paths; assert before any training (Phase 0 guards)
        assert hasattr(self.lm, "model") and hasattr(self.lm, "lm_head"), "backbone needs lm.model(...).last_hidden_state and lm.lm_head"
        if getattr(self.lm.config, "tie_word_embeddings", False):
            assert self.lm.lm_head.weight.data_ptr() == self.lm.model.embed_tokens.weight.data_ptr(), \
                "config says tied embeddings but lm_head is not embed_tokens"
        if grad_ckpt:
            self.lm.gradient_checkpointing_enable()
        # one row per single-token label; the width is the tokenizer's capacity, not a constant
        self.register_buffer("letters", torch.tensor(letter_ids(self.tok)), persistent=False)

    def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
        """input_ids [B,T]; slot_idx/slot_batch [N] flat slot positions; nopts [N].
        Returns [N,K] logits (K = label capacity) with invalid options masked to -inf."""
        h = self.lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hs = h[slot_batch, slot_idx]                                   # [N,H]
        W = self.lm.lm_head.weight[self.letters]                       # [K,H]
        logits = F.linear(hs, W).float()                               # [N,K]
        ar = torch.arange(W.shape[0], device=logits.device)[None, :]
        logits = logits.masked_fill(ar >= nopts[:, None], float("-inf"))
        return logits

    def forward(self, batch):
        return self.slot_logits(batch["input_ids"], batch["attention_mask"], batch["slot_idx"], batch["slot_batch"], batch["nopts"])


def pad_id(tok):
    """Some tokenizers ship no pad token; fall back to EOS (the prompt never emits it)."""
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def collate(items, pad_id):
    """items: list of dicts from prompt.build (+ 'task', 'ex_id'). Right-pad."""
    T = max(len(it["ids"]) for it in items)
    T = ((T + 63) // 64) * 64          # few distinct shapes -> fewer kernel (re)compiles
    B = len(items)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    slot_idx, slot_batch, golds, nopts, tasks, qidx = [], [], [], [], [], []
    for b, it in enumerate(items):
        n = len(it["ids"])
        input_ids[b, :n] = torch.tensor(it["ids"])
        attn[b, :n] = 1
        for k, s in enumerate(it["slots"]):
            slot_idx.append(s); slot_batch.append(b); golds.append(it["golds"][k]); nopts.append(it["nopts"][k])
            tasks.append(it.get("task", "")); qidx.append(k)
    return dict(input_ids=input_ids, attention_mask=attn, slot_idx=torch.tensor(slot_idx), slot_batch=torch.tensor(slot_batch),
                golds=torch.tensor(golds), nopts=torch.tensor(nopts), tasks=tasks, qidx=qidx)
