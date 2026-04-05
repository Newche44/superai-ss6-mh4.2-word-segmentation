# superai-ss6-mh4.2-word-segmentation

# ✂️ MH4.2 — Thai Word Segmentation

**Super AI Engineer Season 6 | Mini Hackathon 4.2**

Thai word segmentation ด้วย XLM-RoBERTa fine-tuned บน LST20 corpus
โดยใช้ BIE (Begin-Inner-End) tagging scheme สำหรับ character-level prediction

---

## 📊 Result

| Metric | Value |
|--------|-------|
| Score (F1 macro) | 0.89254 |
| Rank | 287 |
| Baseline | 0.97174 |
| Status | Below baseline |

---

## 🔧 Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `XLM-RoBERTa` · `seqeval` · `Google Colab`

---

## 🏗️ Approach

### Model
- **Base:** `xlm-roberta-base` (HuggingFace)
- **Task:** Token Classification (NER-style)

### Tagging Scheme — BIE
```
B_WORD  →  อักขระแรกของคำ
I_WORD  →  อักขระกลางของคำ
E_WORD  →  อักขระสุดท้ายของคำ
```

### Training Data
- **Dataset:** `plukio/modified_lst20` (HuggingFace Hub)
- LST20 เป็น Thai NLP corpus มาตรฐาน

### Training Config
```python
Model:          xlm-roberta-base
Max Length:     128 tokens
Epochs:         5
Batch size:     16 (GPU)
Learning rate:  2e-5
Weight decay:   0.01
Warmup steps:   200
fp16:           True
Metric:         F1 macro (seqeval)
```

### Inference — Chunked Prediction
```
test_chars → chunks of 120 chars
→ predict per chunk
→ concatenate all tags
→ export submission.csv
```

---

## 💡 Post-hoc Analysis

Score 0.89 vs baseline 0.97 — gap น่าจะมาจาก:
- Domain mismatch ระหว่าง LST20 (general Thai) กับ test set
- BIE scheme อาจให้ผล E_WORD ผิดพลาดบ่อยกว่า BI scheme
- `xlm-roberta-base` อาจ underfit — ควรลอง `wangchanberta` หรือ `phayathai`

---

## 📁 Files

```
mh4.2-colab.py   # Main notebook
```

---
