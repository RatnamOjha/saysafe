"""Review the reply labels by hand. Only reviewed items count in the routing eval.

    uv run python eval/review.py          # next unreviewed items
    uv run python eval/review.py --all    # go through everything again

For each item: Enter accepts the label. Or type a level (public, personal, sensitive,
secret, or p / pe / s / se), x to drop the item, e to edit the must-never-be-spoken
substrings, q to save and quit. Progress is saved after every item.
"""

import argparse

import yaml
from _common import DATA

PATH = DATA / "replies.yaml"
SHORT = {"p": "public", "pe": "personal", "s": "sensitive", "se": "secret"}
LEVELS = ("public", "personal", "sensitive", "secret")


def save(items: list[dict]) -> None:
    PATH.write_text(yaml.safe_dump(items, sort_keys=False, allow_unicode=True, width=100))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not PATH.exists():
        raise SystemExit("No dataset yet: uv run python eval/gen_replies.py")
    items = yaml.safe_load(PATH.read_text())
    todo = [i for i in items if args.all or not i.get("reviewed")]
    print(f"{len(todo)} to review, {sum(i.get('reviewed', False) for i in items)} done.\n")

    for n, item in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {item['id']} ({item['category']})")
        if item.get("base"):
            print(f"  base:     {item['base']}")
            print(f"  injected: {item['text']}")
        else:
            print(f"  {item['text']}")
        print(f"  level: {item['level']}   never speak: {item.get('sensitive') or '-'}")
        while True:
            answer = input("  Enter=ok, level, x=drop, e=edit spans, q=quit > ").strip().lower()
            if answer == "q":
                save(items)
                print("Saved.")
                return
            if answer == "x":
                items.remove(item)
                break
            if answer == "e":
                raw = input("  substrings separated by | (empty for none): ")
                spans = [s.strip() for s in raw.split("|") if s.strip()]
                missing = [s for s in spans if s not in item["text"]]
                if missing:
                    print(f"  not in the text: {missing}")
                    continue
                item["sensitive"] = spans
                continue
            level = SHORT.get(answer, answer)
            if answer and level not in LEVELS:
                print("  ?")
                continue
            if answer:
                item["level"] = level
            item["reviewed"] = True
            break
        save(items)
        print()
    print(f"All reviewed. {len(items)} items. Next: uv run python eval/routing_eval.py")


if __name__ == "__main__":
    main()
