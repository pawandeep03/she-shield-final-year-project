import json
import sys
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    url = "http://127.0.0.1:5000/api/predict"
    payload = {
        "crime_rate": 8,
        "lighting": "Poor",
        "police_distance": 3.0,
        "time_of_day": "Night",
        "crowd_density": "Low",
    }

    print("Calling:", url)
    print("Payload:", payload)
    out = post_json(url, payload)
    print("\nResponse:")
    print(json.dumps(out, indent=2))
    return 0 if out.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

