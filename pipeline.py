"""
Gemini -> Instagram pipeline.

Design: ONE deterministic run = ONE function call (run_pipeline). Scheduling is
a separate concern:  python pipeline.py            -> run once (cron / GitHub Actions call this)
                     python pipeline.py --serve    -> long-running APScheduler loop
"""
import argparse, base64, io, json, logging, os, sqlite3, time, traceback, uuid
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import cloudinary, cloudinary.uploader

load_dotenv()
E = os.environ.get
OUT = Path("out"); OUT.mkdir(exist_ok=True)
DB = sqlite3.connect("runs.db")
DB.execute("""CREATE TABLE IF NOT EXISTS runs(
    run_id TEXT PRIMARY KEY, ts REAL, status TEXT, caption TEXT, post_id TEXT, report TEXT)""")
log = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------- helpers ----------
class Post(BaseModel):                       # the contract between LLM and the rest of the code
    image_prompt: str
    caption: str = Field(max_length=1800)
    hashtags: list[str] = Field(max_length=15)


def tg(method, **kw):
    r = requests.post(f"https://api.telegram.org/bot{E('TG_TOKEN')}/{method}", json=kw, timeout=40)
    return r.json()


def graph(method, path, **params):
    """Graph API call that surfaces Meta's error body (raise_for_status alone hides it)."""
    url = f"https://graph.facebook.com/{E('GRAPH_VERSION', 'v23.0')}/{path}"
    params["access_token"] = E("IG_ACCESS_TOKEN")
    r = requests.request(method, url, params=params if method == "GET" else None,
                         data=params if method == "POST" else None, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Graph API {r.status_code}: {r.text}")
    return r.json()


class Rejected(Exception):
    pass


# ---------- stages: each reads/writes ctx and returns a small dict of diagnostics ----------
def gen_prompt(ctx):
    recent = [r[0] for r in DB.execute("SELECT caption FROM runs WHERE caption IS NOT NULL "
                                       "ORDER BY ts DESC LIMIT 10")]
    client = genai.Client(api_key=E("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model=E("TEXT_MODEL", "gemini-2.5-flash"),
        contents=(f"You run an Instagram page. Theme: {E('THEME')}.\n"
                  "Write ONE new post: a detailed, concrete image-generation prompt, a short "
                  f"engaging caption, and hashtags. Do NOT repeat these recent captions: {recent}"),
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=Post))
    ctx["post"] = Post.model_validate_json(resp.text)
    return {"prompt_chars": len(ctx["post"].image_prompt)}


def gen_image(ctx):
    client = genai.Client(api_key=E("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model=E("IMAGE_MODEL"),
        contents=ctx["post"].image_prompt + "\nPortrait composition, 4:5 aspect ratio.",
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]))
    raw = next((p.inline_data.data for p in resp.candidates[0].content.parts
                if getattr(p, "inline_data", None)), None)
    if raw is None:
        raise RuntimeError(f"No image returned (maybe safety block): {resp.candidates[0].finish_reason}")
    if isinstance(raw, str):
        raw = base64.b64decode(raw)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    # Instagram API: JPEG only, feed ratio must be within 4:5 .. 1.91:1 -> force 4:5 deterministically
    w, h = img.size
    target = 4 / 5
    if w / h > target:                       # too wide -> crop sides
        nw = int(h * target); img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:                                    # too tall -> crop top/bottom
        nh = int(w / target); img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    img = img.resize((1080, 1350), Image.LANCZOS)
    path = OUT / f"{ctx['run_id']}.jpg"
    img.save(path, "JPEG", quality=92)
    ctx["image_path"] = path
    return {"file": str(path), "kb": path.stat().st_size // 1024}


def host_image(ctx):
    res = cloudinary.uploader.upload(str(ctx["image_path"]), public_id=f"igbot/{ctx['run_id']}")
    ctx["image_url"] = res["secure_url"]
    return {"url": ctx["image_url"]}


def ask_approval(ctx):
    if E("REQUIRE_APPROVAL", "1") != "1":
        return {"approval": "skipped"}
    rid = ctx["run_id"]
    kb = {"inline_keyboard": [[{"text": "✅ Post", "callback_data": f"ok:{rid}"},
                               {"text": "❌ Reject", "callback_data": f"no:{rid}"}]]}
    with open(ctx["image_path"], "rb") as f:
        requests.post(f"https://api.telegram.org/bot{E('TG_TOKEN')}/sendPhoto",
                      data={"chat_id": E("TG_CHAT_ID"), "reply_markup": json.dumps(kb),
                            "caption": ctx["post"].caption[:900]}, files={"photo": f}, timeout=60)
    deadline, offset = time.time() + int(E("APPROVAL_TIMEOUT", "1800")), None
    while time.time() < deadline:
        kw = {"timeout": 25, **({"offset": offset} if offset else {})}
        for u in tg("getUpdates", **kw).get("result", []):
            offset = u["update_id"] + 1
            cb = u.get("callback_query")
            if cb and cb["data"].endswith(rid):
                tg("answerCallbackQuery", callback_query_id=cb["id"])
                if cb["data"].startswith("ok"):
                    return {"approval": "approved"}
                raise Rejected("rejected by human")
    raise TimeoutError("no approval before timeout")


def publish(ctx):
    ig = E("IG_USER_ID")
    caption = ctx["post"].caption + "\n\n" + " ".join(
        h if h.startswith("#") else f"#{h}" for h in ctx["post"].hashtags)
    if E("DRY_RUN", "1") == "1":
        return {"dry_run": True, "caption_preview": caption[:80]}
    quota = graph("GET", f"{ig}/content_publishing_limit", fields="quota_usage,config")
    cid = graph("POST", f"{ig}/media", image_url=ctx["image_url"], caption=caption)["id"]
    for _ in range(20):                      # poll container until Meta has fetched/processed it
        status = graph("GET", cid, fields="status_code")["status_code"]
        if status == "FINISHED":
            break
        if status in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"container status {status}")
        time.sleep(3)
    ctx["post_id"] = graph("POST", f"{ig}/media_publish", creation_id=cid)["id"]
    return {"post_id": ctx["post_id"], "quota": quota.get("data")}


# ---------- orchestration ----------
@dataclass
class StageResult:
    name: str
    ok: bool
    secs: float
    detail: dict = field(default_factory=dict)
    error: str | None = None


def run_stage(name, fn, ctx, retries=2):
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            out = fn(ctx) or {}
            return StageResult(name, True, time.time() - t0, {"attempts": attempt + 1, **out})
        except Exception as e:
            log.warning("stage %s attempt %d failed: %s", name, attempt + 1, e)
            if attempt == retries or isinstance(e, (Rejected, TimeoutError)):
                return StageResult(name, False, time.time() - t0, {"attempts": attempt + 1},
                                   "".join(traceback.format_exception_only(type(e), e)).strip())
            time.sleep(5 * 2 ** attempt)     # exponential backoff


def format_report(ctx, results):
    ok = all(r.ok for r in results) and len(results) == 5
    lines = [f"{'✅' if ok else '❌'} run {ctx['run_id']}  total {sum(r.secs for r in results):.0f}s"]
    for r in results:
        lines.append(f"{'✅' if r.ok else '❌'} {r.name:<9}{r.secs:5.1f}s  tries={r.detail.get('attempts')}"
                     + (f"\n    ↳ {r.error[:300]}" if r.error else ""))
    return "\n".join(lines)


def run_pipeline():
    ctx = {"run_id": uuid.uuid4().hex[:6]}
    stages = [("prompt", gen_prompt, 2), ("image", gen_image, 2), ("upload", host_image, 2),
              ("approve", ask_approval, 0), ("publish", publish, 1)]
    results = []
    for name, fn, retries in stages:
        r = run_stage(name, fn, ctx, retries)
        results.append(r)
        if not r.ok:
            break                            # fail fast, but still report below
    report = format_report(ctx, results)
    status = "ok" if report.startswith("✅") else "failed"
    DB.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)",
               (ctx["run_id"], time.time(), status,
                ctx["post"].caption if "post" in ctx else None, ctx.get("post_id"), report))
    DB.commit()
    log.info("\n%s", report)
    if E("TG_TOKEN"):
        tg("sendMessage", chat_id=E("TG_CHAT_ID"), text=report)
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="run on a cron schedule inside this process")
    args = ap.parse_args()
    if args.serve:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
        sched = BlockingScheduler()
        sched.add_job(run_pipeline, CronTrigger.from_crontab(E("POST_CRON", "0 9 * * *")),
                      max_instances=1, coalesce=True, misfire_grace_time=3600)
        log.info("scheduler started"); sched.start()
    else:
        raise SystemExit(0 if run_pipeline() == "ok" else 1)
