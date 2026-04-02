from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SSL DINOv2 Visuals</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #10161e; color: #e8eef7; }
    .wrap { padding: 14px; }
    h1 { margin: 0 0 10px; font-size: 20px; }
    .controls { display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 8px; margin-bottom: 10px; }
    label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
    select, input { border: 1px solid #344458; border-radius: 6px; padding: 6px; background: #17212d; color: #e8eef7; }
    .meta { font-size: 12px; color: #a6b6c8; margin-bottom: 10px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }
    .card { border: 1px solid #2c3b4e; border-radius: 8px; overflow: hidden; background: #141d27; }
    .card img { width: 100%; display: block; background: #000; }
    .cap { padding: 8px; font-size: 12px; line-height: 1.35; color: #c9d6e6; }
    .empty { color: #9db0c5; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SSL DINOv2 Training Visuals</h1>
    <div class="controls">
      <label>Epoch<select id="epoch"></select></label>
      <label>Group<select id="group"></select></label>
      <label>Sample<select id="sample"></select></label>
      <label>Kind<select id="kind"></select></label>
      <label>Search<input id="search" placeholder="path contains..." /></label>
    </div>
    <div class="meta" id="meta"></div>
    <div class="grid" id="grid"></div>
  </div>
  <script>
    const els = {
      epoch: document.getElementById("epoch"),
      group: document.getElementById("group"),
      sample: document.getElementById("sample"),
      kind: document.getElementById("kind"),
      search: document.getElementById("search"),
      meta: document.getElementById("meta"),
      grid: document.getElementById("grid"),
    };
    let rows = [];
    function uniq(a){ return [...new Set(a)].sort((x,y)=>String(x).localeCompare(String(y),undefined,{numeric:true})); }
    function fill(el, vals, lbl){
      const cur = el.value; el.innerHTML = "";
      const a = document.createElement("option"); a.value=""; a.textContent=lbl; el.appendChild(a);
      vals.forEach(v=>{ const o=document.createElement("option"); o.value=String(v); o.textContent=String(v); el.appendChild(o); });
      if([...el.options].some(o=>o.value===cur)) el.value=cur;
    }
    function render(list){
      els.grid.innerHTML = "";
      if(!list.length){ const e=document.createElement("div"); e.className="empty"; e.textContent="No artifacts match filters."; els.grid.appendChild(e); return; }
      list.forEach(r=>{
        const c=document.createElement("div"); c.className="card";
        c.innerHTML = `<img loading="lazy" src="${r.path}" alt="${r.path}" /><div class="cap">
        <div>epoch=${r.epoch} sample=${r.sample_idx} group=${r.group||"-"} kind=${r.kind||"-"}</div>
        <div>${r.path}</div></div>`;
        els.grid.appendChild(c);
      });
      els.meta.textContent = `${list.length} / ${rows.length} artifacts`;
    }
    function apply(){
      const f = rows.filter(r =>
        (!els.epoch.value || String(r.epoch)===els.epoch.value) &&
        (!els.group.value || String(r.group||"")===els.group.value) &&
        (!els.sample.value || String(r.sample_idx)===els.sample.value) &&
        (!els.kind.value || String(r.kind||"")===els.kind.value) &&
        (!els.search.value || String(r.path||"").toLowerCase().includes(els.search.value.toLowerCase()))
      );
      render(f);
    }
    async function boot(){
      try{
        const t = await (await fetch("manifest.jsonl", {cache:"no-store"})).text();
        rows = t.split("\\n").map(s=>s.trim()).filter(Boolean).map(s=>JSON.parse(s));
      } catch(err) {
        els.meta.textContent = "Erro ao carregar manifest.jsonl";
        return;
      }
      fill(els.epoch, uniq(rows.map(r=>r.epoch)), "All epochs");
      fill(els.group, uniq(rows.map(r=>r.group).filter(Boolean)), "All groups");
      fill(els.sample, uniq(rows.map(r=>r.sample_idx)), "All samples");
      fill(els.kind, uniq(rows.map(r=>r.kind).filter(Boolean)), "All kinds");
      ["epoch","group","sample","kind"].forEach(k => els[k].addEventListener("change", apply));
      els.search.addEventListener("input", apply);
      apply();
    }
    boot();
  </script>
</body>
</html>
"""


def ensure_viewer(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "index.html"
    if not p.exists():
        p.write_text(VIEWER_HTML, encoding="utf-8")


def append_manifest(out_dir: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    out = out_dir / "manifest.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _panel_h(images: list[np.ndarray], titles: list[str], out_path: Path) -> None:
    if not images:
        return
    h = max(int(img.shape[0]) for img in images)
    w = max(int(img.shape[1]) for img in images)
    bar_h = 24
    panel = np.zeros((h + bar_h, w * len(images), 3), dtype=np.uint8)
    for i, (img, title) in enumerate(zip(images, titles)):
        x0 = i * w
        ih, iw = int(img.shape[0]), int(img.shape[1])
        y_off = (h - ih) // 2
        x_off = x0 + (w - iw) // 2
        panel[bar_h + y_off : bar_h + y_off + ih, x_off : x_off + iw] = img
    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    for i, title in enumerate(titles):
        x0 = i * w
        draw.text((x0 + 6, 6), title, fill=(230, 230, 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def _attention_overlay(base_rgb: np.ndarray, att01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = base_rgb.shape[:2]
    att = np.array(Image.fromarray((att01 * 255.0).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)).astype(np.float32) / 255.0
    # Mapa simples frio->quente sem dependencia de OpenCV.
    heat_r = np.clip(2.0 * att, 0.0, 1.0)
    heat_g = np.clip(1.5 - np.abs(2.0 * att - 1.0) * 1.5, 0.0, 1.0)
    heat_b = np.clip(2.0 * (1.0 - att), 0.0, 1.0)
    heat = np.stack([heat_r, heat_g, heat_b], axis=-1) * 255.0
    out = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _vector_to_map(vec: np.ndarray, h_tokens: int, w_tokens: int) -> np.ndarray:
    if vec.size < h_tokens * w_tokens:
        raise ValueError("vetor de atencao menor que grid de patches")
    v = vec[-(h_tokens * w_tokens) :].reshape(h_tokens, w_tokens)
    v = (v - v.min()) / max(1e-8, (v.max() - v.min()))
    return v.astype(np.float32)


@torch.no_grad()
def capture_ssl_epoch_visuals(
    out_dir: Path,
    epoch: int,
    sample_paths: list[Path],
    transform,
    student_backbone: torch.nn.Module,
    device: str,
    max_samples: int = 8,
    interval: int = 1,
) -> None:
    if interval < 1 or (epoch % interval != 0):
        return
    ensure_viewer(out_dir)
    n = min(max_samples, len(sample_paths))
    ts = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []

    was_training = student_backbone.training
    student_backbone.eval()
    for i in range(n):
        p = sample_paths[i]
        rgb = Image.open(p).convert("RGB")
        bundle = transform.sample_views(rgb)

        previews = bundle["preview_images"]
        metas = bundle["meta"]
        tiles = [x for x in previews]
        titles = [f"{m['kind']} sz={m['size']} a={m['area_scale']:.3f}" for m in metas]

        rel = Path(f"epoch_{epoch:04d}/views/sample_{i:02d}.png")
        _panel_h(tiles, titles, out_dir / rel)
        records.append(
            {
                "timestamp_utc": ts,
                "epoch": int(epoch),
                "sample_idx": int(i),
                "group": "views",
                "kind": "multicrop",
                "path": rel.as_posix(),
                "source_path": str(p),
            }
        )

        # Mapa de atenção ViT na primeira view global.
        # Método principal: CLS · patch_tokens (igual ao laboratório de clusterização).
        try:
            x = bundle["student_crops"][0].unsqueeze(0).to(device)
            out = student_backbone(pixel_values=x, output_attentions=True)
            patch_size = int(getattr(getattr(student_backbone, "config", object()), "patch_size", 14))
            h_tokens = max(1, x.shape[-2] // patch_size)
            w_tokens = max(1, x.shape[-1] // patch_size)
            n_patch = int(h_tokens * w_tokens)

            att_map = None
            kind = "cls_dot_patch"
            if getattr(out, "last_hidden_state", None) is not None:
                hidden = out.last_hidden_state[0].detach().cpu().numpy()  # [tokens, D]
                if hidden.shape[0] > 1:
                    cls_tok = hidden[0]
                    patch_toks = hidden[-n_patch:]
                    cls_norm = np.linalg.norm(cls_tok) + 1e-8
                    patch_norm = np.linalg.norm(patch_toks, axis=1, keepdims=True) + 1e-8
                    cls_u = cls_tok / cls_norm
                    patch_u = patch_toks / patch_norm
                    vec = np.sum(patch_u * cls_u[None, :], axis=1)
                    att_map = _vector_to_map(vec, h_tokens, w_tokens)

            # Fallback opcional (somente se CLS·patch não estiver disponível).
            if att_map is None and getattr(out, "attentions", None) is not None and out.attentions[-1] is not None:
                att = out.attentions[-1][0].mean(dim=0)  # [tokens, tokens]
                cls_row = att[0].detach().cpu().numpy()
                if cls_row.size > 1:
                    vec = cls_row[1:]
                    att_map = _vector_to_map(vec, h_tokens, w_tokens)
                    kind = "cls_row_attention_fallback"

            if att_map is not None:
                overlay = _attention_overlay(tiles[0], att_map)
                rel_att = Path(f"epoch_{epoch:04d}/attention/sample_{i:02d}.png")
                _panel_h([tiles[0], overlay], ["Global view", f"Attention ({kind})"], out_dir / rel_att)
                records.append(
                    {
                        "timestamp_utc": ts,
                        "epoch": int(epoch),
                        "sample_idx": int(i),
                        "group": "attention",
                        "kind": kind,
                        "path": rel_att.as_posix(),
                        "source_path": str(p),
                    }
                )
        except Exception:
            # Sem quebrar o treino se uma visualizacao falhar.
            pass

    if was_training:
        student_backbone.train()
    append_manifest(out_dir, records)
