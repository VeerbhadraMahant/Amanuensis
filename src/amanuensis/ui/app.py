"""Local correction UI. Usage: uv run python -m amanuensis.ui.app  (binds to 127.0.0.1 only; audio never leaves the machine)."""
import sqlite3
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from amanuensis.config import load_language_tags, load_paths, load_variants
from amanuensis.registry import approvals
from amanuensis.store import db, lexicon, review
from amanuensis.text.spelling import check_spelling

STATIC = Path(__file__).parent / "static"


class CorrectBody(BaseModel):
    text: str
    tags: list[str]


class TagsBody(BaseModel):
    tags: list[str]


class SpellBody(BaseModel):
    text: str


class LexiconBody(BaseModel):
    canonical: str
    variants: list[str] = []
    kind: str


def create_app(conn: sqlite3.Connection, audio_dir: Path, tags: list[str], variants: dict[str, str]) -> FastAPI:
    # Endpoints are async on purpose: they run on the single event-loop thread, so the shared
    # sqlite connection is never used concurrently.
    app = FastAPI(title="Amanuensis correction UI")

    def guarded(fn, *args):  # KeyError -> 404, ValueError -> 422
        try:
            return fn(*args)
        except KeyError:
            raise HTTPException(404, "utterance not found")
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    async def config():
        return {"tags": tags}

    @app.get("/api/utterances")
    async def list_utterances(status: str | None = "raw", limit: int = 50, offset: int = 0):
        return review.list_utterances(conn, status or None, limit, offset)

    @app.get("/api/audio/{utt_id}")
    async def audio(utt_id: int):
        utt = review.get_utterance(conn, utt_id)
        path = (audio_dir / utt["audio_path"]).resolve() if utt else None
        if path is None or not path.is_file() or audio_dir.resolve() not in path.parents:
            raise HTTPException(404, "audio not found")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/utterances/{utt_id}/correct")
    async def correct(utt_id: int, body: CorrectBody):
        guarded(review.save_correction, conn, utt_id, body.text, body.tags, tags)
        return {"ok": True}

    @app.post("/api/utterances/{utt_id}/approve")
    async def approve(utt_id: int, body: TagsBody):
        guarded(review.approve_as_is, conn, utt_id, body.tags, tags)
        return {"ok": True}

    @app.post("/api/utterances/{utt_id}/reject")
    async def reject(utt_id: int):
        guarded(review.reject, conn, utt_id)
        return {"ok": True}

    @app.post("/api/spellcheck")
    async def spellcheck(body: SpellBody):
        return [asdict(f) for f in check_spelling(body.text, {**variants, **lexicon.variant_map(conn)})]

    @app.get("/api/lexicon")
    async def lexicon_list():
        return lexicon.list_entries(conn)

    @app.post("/api/lexicon")
    async def lexicon_add(body: LexiconBody):
        return {"id": guarded(lexicon.add_entry, conn, body.canonical, body.variants, body.kind)}

    @app.post("/api/lexicon/{entry_id}/update")
    async def lexicon_update(entry_id: int, body: LexiconBody):
        guarded(lexicon.update_entry, conn, entry_id, body.canonical, body.variants, body.kind)
        return {"ok": True}

    @app.post("/api/lexicon/{entry_id}/approve")
    async def lexicon_approve(entry_id: int):
        guarded(lexicon.set_approved, conn, entry_id, True)
        return {"ok": True}

    @app.post("/api/lexicon/{entry_id}/delete")
    async def lexicon_delete(entry_id: int):
        guarded(lexicon.delete_entry, conn, entry_id)
        return {"ok": True}

    @app.get("/api/approvals")
    async def approvals_pending():
        return approvals.pending(conn)

    @app.post("/api/approvals/{request_id}/approve")
    async def approvals_approve(request_id: int):
        guarded(approvals.approve, conn, request_id)
        return {"ok": True}

    @app.post("/api/approvals/{request_id}/decline")
    async def approvals_decline(request_id: int):
        guarded(approvals.decline, conn, request_id)
        return {"ok": True}

    @app.get("/api/stats")
    async def stats():
        return review.stats(conn)

    return app


def main() -> None:
    import uvicorn

    paths = load_paths()
    conn = db.connect(paths.db_path, check_same_thread=False)
    app = create_app(conn, paths.audio_dir, load_language_tags(), load_variants(paths.variants_file))
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
