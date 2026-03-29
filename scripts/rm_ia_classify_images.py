#!/usr/bin/env python3
import argparse
import base64
import json
import os
import random
import html
from pathlib import Path
from typing import Dict, Iterable, List

import requests

DEFAULT_BASE_URL = os.getenv("RM_BASE_URL", "https://api.radiomemory.com.br/ia-idoc")
DEFAULT_USERNAME = os.getenv("RM_USERNAME", "test")
DEFAULT_PASSWORD = os.getenv("RM_PASSWORD", "")
DEFAULT_TIMEOUT = int(os.getenv("RM_TIMEOUT", "60"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def login(base_url: str, username: str, password: str, timeout: int) -> Dict[str, str]:
    url = f"{base_url.rstrip('/')}/v1/auth/token"
    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "accept": "application/json",
    }
    body = (
        f"grant_type=&username={username}&password={password}"
        "&scope=&client_id=&client_secret="
    )
    resp = requests.post(url, headers=headers, data=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    token_type = data.get("token_type")
    access_token = data.get("access_token")
    if not token_type or not access_token:
        raise RuntimeError(f"Falha na autenticacao: {data}")
    return {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }


def iter_images(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in sorted(folder.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_payload(path: Path, threshold: float, resource: str, lang: str, use_cache: bool) -> Dict:
    return {
        "base64_image": image_to_base64(path),
        "output_width": 0,
        "output_height": 0,
        "threshold": threshold,
        "resource": resource,
        "lang": lang,
        "use_cache": use_cache,
    }


def post_image(base_url: str, endpoint: str, headers: Dict[str, str], payload: Dict, timeout: int) -> requests.Response:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


def _format_entities_preview(resp: Dict, limit: int = 6) -> str:
    entities = resp.get("entities")
    if not isinstance(entities, list) or not entities:
        return "Sem entities"
    rows = []
    for ent in entities[:limit]:
        if not isinstance(ent, dict):
            rows.append(html.escape(str(ent)))
            continue
        cname = ent.get("class_name", "-")
        score = ent.get("score")
        if isinstance(score, (int, float)):
            rows.append(f"{html.escape(str(cname))} ({score:.3f})")
        else:
            rows.append(f"{html.escape(str(cname))}")
    extra = len(entities) - limit
    if extra > 0:
        rows.append(f"... +{extra}")
    return "<br>".join(rows)


def build_html_report(rows: List[Dict], title: str) -> str:
    cards = []
    for row in rows:
        image_path = row.get("image", "")
        image_name = Path(image_path).name if image_path else "(sem imagem)"
        status = row.get("status")
        response = row.get("response", {})
        error = row.get("error", "")

        status_text = f"HTTP {status}" if status is not None else "ERRO"

        model_name = "-"
        entities_count = "-"
        preview = ""
        if isinstance(response, dict):
            model_name = str(response.get("model_name", "-"))
            ents = response.get("entities")
            if isinstance(ents, list):
                entities_count = str(len(ents))
            preview = _format_entities_preview(response)
        elif error:
            preview = html.escape(str(error))
        else:
            preview = html.escape(str(response))[:500]

        if image_path and Path(image_path).exists():
            img_src = Path(image_path).resolve().as_uri()
        else:
            img_src = ""

        img_html = (
            f'<img src="{img_src}" alt="{html.escape(image_name)}" loading="lazy" />'
            if img_src
            else '<div class="img-missing">Imagem indisponivel</div>'
        )

        cards.append(
            f"""
            <article class="card">
              <div class="thumb">{img_html}</div>
              <div class="body">
                <h3 title="{html.escape(image_path)}">{html.escape(image_name)}</h3>
                <p><b>Status:</b> {html.escape(status_text)}</p>
                <p><b>Modelo:</b> {html.escape(model_name)}</p>
                <p><b>Entities:</b> {html.escape(entities_count)}</p>
                <div class="preview">{preview}</div>
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe2ea;
      --accent: #0369a1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #e2e8f0 0%, var(--bg) 45%);
      color: var(--ink);
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 18px; font-size: 24px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 14px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: var(--card);
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }}
    .thumb {{
      height: 180px;
      background: #eef2f7;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .thumb img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #f8fafc;
    }}
    .img-missing {{
      font-size: 12px;
      color: var(--muted);
    }}
    .body {{ padding: 12px 12px 14px; }}
    .body h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      line-height: 1.35;
      color: var(--ink);
      word-break: break-all;
    }}
    .body p {{
      margin: 4px 0;
      font-size: 12px;
      color: var(--muted);
    }}
    .preview {{
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px dashed var(--line);
      font-size: 12px;
      color: var(--ink);
      line-height: 1.35;
      min-height: 44px;
    }}
    .footer {{
      margin-top: 14px;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>{html.escape(title)}</h1>
    <section class="grid">
      {''.join(cards)}
    </section>
    <div class="footer">Gerado por rm_ia_classify_images.py</div>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifica imagens odontologicas via RM IA API (endpoint configuravel)."
    )
    parser.add_argument(
        "--input-dir",
        default="/Users/fabioandrade/RMFM/Downloads/periapicais_100",
        help="Pasta com imagens para enviar.",
    )
    parser.add_argument(
        "--endpoint",
        default="v1/panoramics/dentition",
        help="Endpoint de classificacao (ex.: v1/panoramics/dentition).",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--resource", default="describe")
    parser.add_argument("--lang", default="pt-br")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=8, help="Numero de imagens para teste.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--save-json",
        default="",
        help="Caminho opcional para salvar todas as respostas em JSON.",
    )
    parser.add_argument(
        "--save-html",
        default="",
        help="Caminho opcional para salvar relatorio HTML em mosaico.",
    )
    args = parser.parse_args()

    folder = Path(args.input_dir)
    if not folder.exists():
        raise SystemExit(f"Pasta nao encontrada: {folder}")
    if not args.password:
        raise SystemExit("Defina RM_PASSWORD via --password ou variavel de ambiente RM_PASSWORD")

    images = list(iter_images(folder, recursive=args.recursive))
    if not images:
        raise SystemExit(f"Nenhuma imagem encontrada em {folder}")

    if args.shuffle:
        random.shuffle(images)
    if args.limit > 0:
        images = images[: args.limit]

    print(f"Base URL: {args.base_url}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Imagens selecionadas: {len(images)}")

    headers = login(args.base_url, args.username, args.password, args.timeout)

    all_rows: List[Dict] = []
    ok = 0
    fail = 0

    for i, img in enumerate(images, start=1):
        payload = build_payload(
            img,
            threshold=args.threshold,
            resource=args.resource,
            lang=args.lang,
            use_cache=args.use_cache,
        )
        try:
            resp = post_image(args.base_url, args.endpoint, headers, payload, args.timeout)
            status = resp.status_code
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text[:1000]}

            if 200 <= status < 300:
                ok += 1
            else:
                fail += 1

            print(f"[{i:02d}] {img.name} -> HTTP {status}")
            if isinstance(body, dict):
                preview_keys = list(body.keys())[:8]
                print(f"     keys: {preview_keys}")
            else:
                print(f"     body_type: {type(body).__name__}")

            all_rows.append(
                {
                    "image": str(img),
                    "status": status,
                    "response": body,
                }
            )
        except Exception as exc:
            fail += 1
            print(f"[{i:02d}] {img.name} -> ERRO: {exc}")
            all_rows.append(
                {
                    "image": str(img),
                    "status": None,
                    "error": str(exc),
                }
            )

    print("\nResumo")
    print(f"  sucesso: {ok}")
    print(f"  falha:   {fail}")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  respostas salvas em: {out}")

    if args.save_html:
        out_html = Path(args.save_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        title = f"RM IA - {args.endpoint} ({len(all_rows)} imagens)"
        out_html.write_text(build_html_report(all_rows, title), encoding="utf-8")
        print(f"  relatorio HTML salvo em: {out_html}")


if __name__ == "__main__":
    main()
