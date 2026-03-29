#!/usr/bin/env python
from huggingface_hub import HfApi


def main() -> None:
    api = HfApi()
    models = api.list_models(author="facebook", search="dinov3", limit=100)
    print("Model IDs encontrados para 'facebook/dinov3*':")
    for m in models:
        print(f"- {m.id}")


if __name__ == "__main__":
    main()
