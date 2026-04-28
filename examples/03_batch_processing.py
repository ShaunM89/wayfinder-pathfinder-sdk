"""Example 3: Batch processing multiple (URL, task) pairs.

Since Pathfinder loads the model once at initialization, you can efficiently
process multiple pages by reusing the same SDK instance.

Note: A dedicated `rank_multiple()` API is planned for v0.2 (#32).
      Until then, loop over requests with a single Pathfinder instance.

Run:
    python examples/03_batch_processing.py
"""

from pathfinder_sdk import Pathfinder


# Simulated pre-extracted candidates for multiple pages
PAGES = {
    "https://docs.python.org/3/": [
        {"href": "/tutorial", "text": "Tutorial"},
        {"href": "/library", "text": "Library Reference"},
        {"href": "/whatsnew", "text": "What's New"},
    ],
    "https://docs.python.org/3/library/": [
        {"href": "/library/os.html", "text": "os — Miscellaneous operating system interfaces"},
        {"href": "/library/sys.html", "text": "sys — System-specific parameters and functions"},
        {"href": "/library/pathlib.html", "text": "pathlib — Object-oriented filesystem paths"},
        {"href": "/library/io.html", "text": "io — Core tools for working with streams"},
    ],
}

TASKS = [
    ("https://docs.python.org/3/", "Find the tutorial"),
    ("https://docs.python.org/3/library/", "Find filesystem path utilities"),
]


def main() -> None:
    sdk = Pathfinder(model="default", fetcher=None)

    results = []
    for url, task in TASKS:
        print(f"\nProcessing: {url}")
        print(f"Task: {task}")

        candidates = PAGES.get(url, [])
        result = sdk.rank_candidates(
            url=url,
            task_description=task,
            candidates=candidates,
            top_n=2,
        )

        print(f"Top result: {result.candidates[0].text} ({result.candidates[0].score:.3f})")
        results.append(result)

    print(f"\n=== Summary ===")
    print(f"Processed {len(results)} tasks")
    total_latency = sum(r.latency_ms for r in results)
    print(f"Total latency: {total_latency:.2f} ms")
    print(f"Average latency: {total_latency / len(results):.2f} ms")

    sdk.unload()


if __name__ == "__main__":
    main()
