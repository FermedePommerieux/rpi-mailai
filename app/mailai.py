#!/usr/bin/env python3
"""MailAI: automation for email classification.

This script orchestrates the entire processing pipeline of the application:

* Initialization and maintenance of the SQLite database that stores processed
  emails and their associated decisions.
* Synchronization of IMAP mailboxes to fetch the raw message content and keep
  track of their last server-side location.
* Training and inference of a classification model (Sentence Transformers +
  logistic regression) to suggest or apply automatic triage rules.
* Monitoring and scheduling utilities (statistics plus the continuous loop).

The objective of this extensive documentation is to keep the processing flow
explicit: every step is thoroughly commented to ease onboarding and future
debugging sessions.
"""

import json
import os, sys, time, yaml, sqlite3, email, re, random, hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from imapclient import IMAPClient
from mailparser import parse_from_bytes
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Process-local cache to avoid reloading the encoder multiple times during the
# same execution (especially for snapshot/maintenance runs).
_ENCODER_CACHE = {}

# === CONFIG PATHS ===
CFG_PATH = Path(os.environ.get("APP_CONFIG", "/config/config.yml"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MAIL_TYPES_DIR = Path(os.environ.get("MAIL_TYPES_DIR", "/config/account_types"))
DB_PATH = DATA_DIR / "db" / "mailai.sqlite"
MODEL_DIR = DATA_DIR / "models"

# Training and inference need a directory to store artefacts (models, encoder).
# We therefore create it during module import time to avoid races in parallel
# executions.
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === HELPERS ===
def load_cfg():
    """Load the main configuration for the application.

    The YAML configuration describes, among other things, the IMAP accounts to
    monitor, the model parameters, and the scheduling options.
    """

    # Ensure the file is present before attempting to parse it: without a
    # configuration the application cannot run at all.
    if not CFG_PATH.exists():
        print(f"[ERROR] config missing: {CFG_PATH}", file=sys.stderr)
        sys.exit(2)

    # Plain YAML loading into a Python dictionary.
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

def _looks_like_hash(value):
    """Detect whether a string looks like a hexadecimal SHA-256 hash."""

    if not value or not isinstance(value, str):
        return False
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def compute_mail_key(account, raw_identifier, salt):
    """Return a stable anonymised identifier for an email."""

    account = account or ""
    raw_identifier = raw_identifier or ""
    if isinstance(salt, bytes):
        salt_bytes = salt
    else:
        salt_bytes = str(salt or "").encode("utf-8")
    payload = f"{account}\0{raw_identifier}".encode("utf-8", "ignore")
    hasher = hashlib.sha256()
    if salt_bytes:
        hasher.update(salt_bytes)
    hasher.update(payload)
    return hasher.hexdigest()


def db_init(cfg=None):
    """Prepare the SQLite database and apply the minimal migrations.

    The function creates the required tables if they do not exist yet and then
    applies a few simple migrations (column additions). It returns a reusable
    `sqlite3.Connection` handle to the caller.
    """

    # Create the working directory if needed before opening the database.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cfg = cfg if isinstance(cfg, dict) else {}
    hash_salt = str(cfg.get("hash_salt") or "")

    # Main table: one record per unique email (account + Message-ID).
    c.execute(
        """CREATE TABLE IF NOT EXISTS mails (
        id INTEGER PRIMARY KEY,
        account TEXT, msgid TEXT, subject TEXT, body TEXT,
        folder TEXT, date TEXT, decision TEXT, confidence REAL,
        imap_uid TEXT, last_seen_at TEXT, auto_moved INTEGER DEFAULT 0,
        auto_moved_at TEXT,
        UNIQUE(account,msgid)
    )"""
    )

    # --- light migrations ---
    # Inspect the current schema and add missing columns on the fly.
    c.execute("PRAGMA table_info(mails)")
    cols = {row[1] for row in c.fetchall()}
    if "imap_uid" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN imap_uid TEXT")
    if "last_seen_at" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN last_seen_at TEXT")
    if "auto_moved" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN auto_moved INTEGER DEFAULT 0")
    if "auto_moved_at" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN auto_moved_at TEXT")
    if "embedding" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN embedding BLOB")
    if "embedding_dim" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN embedding_dim INTEGER")
    if "embedding_encoder" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN embedding_encoder TEXT")

    # Automatically purge plain text when an embedding is available to limit
    # the exposure of sensitive data in the database.
    c.execute(
        "UPDATE mails SET subject=NULL, body=NULL "
        "WHERE embedding IS NOT NULL AND (subject IS NOT NULL OR body IS NOT NULL)"
    )

    # Migration: anonymise message identifiers via a salted hash.
    c.execute(
        "SELECT id, account, msgid FROM mails "
        "WHERE msgid IS NOT NULL AND (LENGTH(msgid) != 64 OR msgid GLOB '*[^0-9a-f]*')"
    )
    updates = []
    for row_id, account, msgid in c.fetchall():
        if msgid is None:
            continue
        if _looks_like_hash(msgid):
            continue
        anon = compute_mail_key(account, msgid, hash_salt)
        if anon != msgid:
            updates.append((anon, row_id))
    if updates:
        c.executemany("UPDATE mails SET msgid=? WHERE id=?", updates)

    # Ensure an explicit unique index exists on (account, msgid).
    c.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_mails_account_msgid ON mails(account, msgid)"
    )

    # Table that tracks the history of completed training runs.
    c.execute(
        """CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY, version TEXT, trained_at TEXT
    )"""
    )

    conn.commit()
    return conn

def clean_text(s):
    """Quickly clean a text snippet to make encoding easier."""

    if not s:
        return ""

    # Collapse repeated whitespace and normalise line breaks.
    return re.sub(r"\s+", " ", s).strip()

def vectorize(texts, encoder):
    """Apply the SentenceTransformer encoder and return a numpy matrix."""

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    emb = encoder.encode(texts, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def get_encoder(enc_name):
    """Load and cache a SentenceTransformer encoder."""

    encoder = _ENCODER_CACHE.get(enc_name)
    if encoder is None:
        print(f"[encoder] loading {enc_name} ...")
        encoder = SentenceTransformer(enc_name)
        _ENCODER_CACHE[enc_name] = encoder
    return encoder


def compute_embedding_bytes(text, encoder):
    """Encode a text snippet and return the pair (bytes, dimension)."""

    vec = vectorize([text], encoder)
    if vec.size == 0:
        return None, None
    arr = vec[0]
    return arr.tobytes(), int(arr.shape[0])


def canonical_folder_name(name: str) -> str:
    """Normalise an IMAP folder name (upper case, `/` separator)."""

    if not name:
        return ""
    if isinstance(name, bytes):
        name = name.decode("utf-8", "ignore")
    cleaned = name.replace("\"", "").replace("'", "")
    cleaned = cleaned.replace(".", "/").replace("\\", "/")
    cleaned = re.sub(r"/+", "/", cleaned)
    return cleaned.strip().upper()


def load_account_mail_types(acc):
    """Load custom typing rules associated with an account."""

    if not acc:
        return {}

    # Look for an explicitly configured path first, otherwise fall back to the
    # account name.
    cfg_path = acc.get("mail_types_config")
    if not cfg_path:
        default_path = MAIL_TYPES_DIR / f"{acc['name']}.json"
        cfg_path = default_path if default_path.exists() else None
    if not cfg_path:
        return {}

    # Read the JSON file describing the rules. Resilience is key because a
    # missing or malformed file should not break the sync loop.
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        print(f"[WARN] invalid mail types JSON for {acc['name']}: {exc}")
        return {}

    # Build a quick index keyed by rule identifier to simplify later lookups.
    entries = payload.get("types", [])
    index = {e.get("key"): e for e in entries if e.get("key")}
    enabled = {k for k, v in index.items() if v.get("enabled", True)}
    return {"path": str(cfg_path), "entries": index, "enabled": enabled}


def _resolve_entry_folders(account_cfg, entry):
    """Determine the IMAP folders tied to a given typing rule."""

    folders = set()
    if not entry:
        return folders

    raw = entry.get("source_folders") or []
    if isinstance(raw, str):
        raw = [raw]
    for item in raw:
        if item:
            folders.add(item)

    # If the rule defines a target folder, include it to watch moves and detect
    # existing classifications.
    target = entry.get("target_folder")
    if not target:
        target = account_cfg.get("folders", {}).get("targets", {}).get(entry.get("key"))
    if target:
        folders.add(target)
    return folders


def _normalise_flags(flags):
    """Return a set of upper-case IMAP flags."""

    normalised = set()
    for flag in flags or ():
        if isinstance(flag, bytes):
            try:
                flag = flag.decode("utf-8")
            except Exception:
                flag = str(flag)
        if not isinstance(flag, str):
            flag = str(flag)
        normalised.add(flag.upper())
    return normalised


def _known_folders_for_account(cur, account_name):
    """Return folders previously observed for an account (excluding INBOX)."""

    cur.execute(
        "SELECT DISTINCT folder FROM mails WHERE account=? AND folder IS NOT NULL AND folder != ''",
        (account_name,),
    )
    folders = set()
    for (folder,) in cur.fetchall():
        canon = canonical_folder_name(folder)
        if not canon or canon in {"INBOX", "__DELETED__"}:
            continue
        folders.add(folder)
    return folders


def _guess_archive_candidates(cur, account_name, decision_folders):
    """Infer archive-like folders based on historical data."""

    cur.execute(
        """
        SELECT folder,
               SUM(CASE WHEN decision IS NULL THEN 1 ELSE 0) AS unlabeled,
               SUM(CASE WHEN auto_moved=1 THEN 1 ELSE 0) AS auto_moved,
               COUNT(*) AS total
        FROM mails
        WHERE account=? AND folder IS NOT NULL AND folder != ''
        GROUP BY folder
        """,
        (account_name,),
    )
    canonical = set()
    raw_names = set()
    for folder, unlabeled, auto_moved, total in cur.fetchall():
        canon = canonical_folder_name(folder)
        if not canon or canon in {"INBOX", "__DELETED__"}:
            continue
        if canon in decision_folders:
            continue
        total = total or 0
        if total == 0:
            continue
        unlabeled = unlabeled or 0
        auto_moved = auto_moved or 0
        ratio = unlabeled / total
        if total >= 2 and ratio >= 0.6 and auto_moved == 0:
            canonical.add(canon)
            raw_names.add(folder)
    return canonical, raw_names


def _is_probably_archived(folder, flags, decision_for_folder, archive_candidates):
    """Heuristically determine whether a message was archived manually."""

    canon = canonical_folder_name(folder)
    if not canon or canon in {"", "INBOX", "__DELETED__"}:
        return False
    if decision_for_folder:
        return False
    if canon in archive_candidates:
        return True
    flag_set = _normalise_flags(flags)
    if ("\\SEEN" in flag_set or "SEEN" in flag_set) and "\\FLAGGED" not in flag_set:
        return True
    return False

# === PIPELINE STEPS ===
def snapshot(cfg):
    """Perform an IMAP synchronisation and refresh the local database.

    This step collects emails from the monitored folders, updates metadata
    (subject, body, last seen folder) and keeps track of user-made decisions to
    feed the training pipeline.
    """

    conn = db_init(cfg)
    cur = conn.cursor()

    # These values act as a fallback if a message disappears from INBOX: they
    # are used to mark emails as deleted/spam automatically.
    delete_pref_keys = {"SPAM", "INDESIRABLE", "JUNK", "PROMOTION", "PROMOTIONS", "NEWSLETTER"}

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    enc_name = model_cfg.get("encoder")
    encoder_available = bool(enc_name)
    encoder = None

    hash_salt = str(cfg.get("hash_salt") or "")

    for acc in cfg.get("accounts", []):
        account_name = acc.get("name")
        if not account_name:
            continue

        now_iso = datetime.utcnow().isoformat()
        print(f"{datetime.now()} [{account_name}] snapshot...")

        # Fetch typing rules and prepare structures that map folders to
        # decisions.
        mail_types = load_account_mail_types(acc)
        entries = mail_types.get("entries", {})
        enabled_keys = {k for k in mail_types.get("enabled", set()) if entries.get(k)}
        folder_sources = defaultdict(set)
        folder_decisions = {}
        delete_fallback = None

        # Iterate over active rules to determine source folders to watch as well
        # as the deletion fallback (spam, promotions, ...).
        for key in enabled_keys:
            entry = entries.get(key) or {}
            sources = _resolve_entry_folders(acc, entry)
            for folder in sources:
                if not folder:
                    continue
                canon = canonical_folder_name(folder)
                if canon:
                    folder_sources[folder].add(key)
                    folder_decisions[canon] = key
            if not delete_fallback and key and key.upper() in delete_pref_keys:
                delete_fallback = key

        decision_folders = set(folder_decisions.keys())
        archive_candidates, archive_watch = _guess_archive_candidates(cur, account_name, decision_folders)
        known_folders = _known_folders_for_account(cur, account_name)
        imap = acc["imap"]
        pwd = Path(imap["password_file"]).read_text().strip()

        # Always watch INBOX and, for each rule, the associated folders (sources
        # + targets to detect pre-existing classifications).
        watch_folders = {"INBOX"}
        for folder in folder_sources:
            watch_folders.add(folder)
        # Add inferred archive folders and any location observed in previous
        # runs so that manual moves remain visible to the synchroniser.
        for folder in archive_watch:
            watch_folders.add(folder)
        for folder in known_folders:
            watch_folders.add(folder)

        seen = set()

        # IMAP connection: the context manager guarantees a clean logout.
        with IMAPClient(imap["host"], port=imap["port"], ssl=imap["ssl"]) as srv:
            srv.login(imap["user"], pwd)
            for folder in sorted(watch_folders):
                try:
                    # Select in read-only mode to avoid altering server-side flags.
                    srv.select_folder(folder, readonly=True)
                except Exception as exc:
                    print(f"[WARN] {account_name} select {folder}: {exc}")
                    continue
                try:
                    # Fetch every UID: no filter to avoid missing recent
                    # messages.
                    uids = srv.search()
                except Exception as exc:
                    print(f"[WARN] {account_name} search {folder}: {exc}")
                    continue
                if not uids:
                    continue
                try:
                    fetched = srv.fetch(uids, ["BODY.PEEK[]", "ENVELOPE", "FLAGS"])
                except Exception as exc:
                    print(f"[WARN] {account_name} fetch {folder}: {exc}")
                    continue

                for uid, data in fetched.items():
                    env = data.get(b"ENVELOPE")
                    raw_identifier = None
                    subject = ""
                    if env:
                        if env.message_id:
                            try:
                                raw_identifier = env.message_id.decode()
                            except Exception:
                                raw_identifier = str(env.message_id)
                        if env.subject:
                            try:
                                subject = env.subject.decode()
                            except Exception:
                                subject = str(env.subject)
                    if not raw_identifier:
                        # Fallback: compose a stable identifier based on the
                        # folder and the IMAP UID.
                        raw_identifier = f"{folder}:{uid}"

                    body = ""
                    raw_body = data.get(b"BODY[]") or data.get(b"BODY.PEEK[]")
                    try:
                        if raw_body:
                            mp = parse_from_bytes(raw_body)
                            body = mp.text_plain[0] if mp.text_plain else (mp.body or "")
                    except Exception:
                        # Parsing can be brittle (malformed emails); ignore the
                        # failure.
                        pass

                    # Cleaning avoids inserting huge, poorly normalised strings
                    # in the database.
                    body = clean_text(body)
                    subject = clean_text(subject)
                    combined_text = clean_text(f"{subject} {body}")

                    canon_folder = canonical_folder_name(folder)
                    decision_for_folder = folder_decisions.get(canon_folder)
                    flags = data.get(b"FLAGS")
                    archive_hit = _is_probably_archived(
                        folder,
                        flags,
                        decision_for_folder,
                        archive_candidates,
                    )

                    # Record that the message was seen during this snapshot.
                    mail_key = compute_mail_key(account_name, raw_identifier, hash_salt)
                    seen.add((account_name, mail_key))

                    cur.execute(
                        "SELECT id, folder, decision, auto_moved, embedding, embedding_encoder, embedding_dim "
                        "FROM mails WHERE account=? AND msgid=?",
                        (account_name, mail_key),
                    )
                    row = cur.fetchone()
                    updates = []
                    params = []
                    embedding_blob = None
                    embedding_dim = None
                    if row:
                        (
                            row_id,
                            prev_folder,
                            prev_decision,
                            prev_auto,
                            prev_embedding,
                            prev_encoder,
                            prev_dim,
                        ) = row
                        needs_embedding = False
                        if prev_folder != folder:
                            # Movement detected -> update folder, UID, and timestamps.
                            updates.append("folder=?")
                            params.append(folder)
                            updates.append("imap_uid=?")
                            params.append(str(uid))
                            updates.append("last_seen_at=?")
                            params.append(now_iso)
                            if prev_auto:
                                # If the message had been auto-moved, reset the
                                # flag to reflect the manual action.
                                updates.append("auto_moved=0")
                                updates.append("auto_moved_at=NULL")
                        else:
                            # No movement: simply refresh the timestamp and UID.
                            updates.append("last_seen_at=?")
                            params.append(now_iso)
                            updates.append("imap_uid=?")
                            params.append(str(uid))
                        if encoder_available and (
                            prev_embedding is None
                            or not prev_dim
                            or prev_encoder != enc_name
                        ):
                            needs_embedding = True
                        if not archive_hit:
                            if decision_for_folder:
                                if prev_decision != decision_for_folder:
                                    # Folder matches a rule -> update the stored decision.
                                    updates.append("decision=?")
                                    params.append(decision_for_folder)
                                    updates.append("confidence=?")
                                    params.append(1.0)
                                    updates.append("auto_moved=0")
                                    updates.append("auto_moved_at=?")
                                    params.append(None)
                            elif prev_decision is not None and canon_folder == "INBOX":
                                # Back to INBOX: consider the decision cancelled.
                                updates.append("decision=NULL")
                                updates.append("confidence=NULL")
                                updates.append("auto_moved=0")
                                updates.append("auto_moved_at=NULL")
                        if needs_embedding:
                            if encoder is None:
                                encoder = get_encoder(enc_name)
                            emb_bytes, emb_dim = compute_embedding_bytes(combined_text, encoder)
                            if emb_bytes is not None:
                                embedding_blob = sqlite3.Binary(emb_bytes)
                                embedding_dim = emb_dim
                            else:
                                print(
                                    f"[WARN] unable to compute embedding for {account_name} {mail_key}"
                                )
                        if embedding_blob is not None:
                            updates.append("embedding=?")
                            params.append(embedding_blob)
                            updates.append("embedding_dim=?")
                            params.append(embedding_dim)
                            updates.append("embedding_encoder=?")
                            params.append(enc_name)
                            updates.append("subject=NULL")
                            updates.append("body=NULL")
                        elif not encoder_available:
                            if subject:
                                updates.append("subject=?")
                                params.append(subject)
                            if body:
                                updates.append("body=?")
                                params.append(body)
                        if updates:
                            set_clause = ",".join(updates)
                            cur.execute(f"UPDATE mails SET {set_clause} WHERE id=?", (*params, row_id))
                    else:
                        if encoder_available:
                            if encoder is None:
                                encoder = get_encoder(enc_name)
                            emb_bytes, emb_dim = compute_embedding_bytes(combined_text, encoder)
                            if emb_bytes is not None:
                                embedding_blob = sqlite3.Binary(emb_bytes)
                                embedding_dim = emb_dim
                            else:
                                print(f"[WARN] unable to compute embedding for {account_name} {mail_key}")
                        subject_value = None
                        body_value = None
                        if embedding_blob is None:
                            subject_value = subject
                            body_value = body
                        # New message: insert a full row with the current metadata.
                        cur.execute(
                            "INSERT INTO mails (account,msgid,subject,body,folder,date,decision,confidence,imap_uid,last_seen_at,auto_moved,embedding,embedding_dim,embedding_encoder) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                account_name,
                                mail_key,
                                subject_value,
                                body_value,
                                folder,
                                now_iso,
                                decision_for_folder if decision_for_folder and not archive_hit else None,
                                1.0 if decision_for_folder and not archive_hit else None,
                                str(uid),
                                now_iso,
                                0,
                                embedding_blob,
                                embedding_dim,
                                enc_name if embedding_blob is not None else None,
                            ),
                        )

        # Handle messages that disappeared from INBOX (likely deleted or marked
        # as spam by the user): mark them using the configured fallback.
        if delete_fallback and seen:
            cur.execute(
                "SELECT id, msgid, folder FROM mails WHERE account=? AND folder='INBOX'",
                (account_name,),
            )
            for row_id, mail_key_db, folder in cur.fetchall():
                if (account_name, mail_key_db) in seen:
                    continue
                cur.execute(
                    "UPDATE mails SET folder=?, decision=?, confidence=?, imap_uid=NULL, last_seen_at=?, auto_moved=0 WHERE id=?",
                    (
                        "__deleted__",
                        delete_fallback,
                        1.0,
                        now_iso,
                        row_id,
                    ),
                )

    conn.commit()
    conn.close()

def retrain(cfg):
    """Retrain the classification model based on labelled data."""

    conn = db_init(cfg)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, embedding, embedding_dim, embedding_encoder, subject, body, decision "
        "FROM mails WHERE decision IS NOT NULL"
    )
    rows = cur.fetchall()
    if not rows:
        print("[WARN] no labeled data")
        return

    enc_name = (cfg.get("model") or {}).get("encoder")
    if not enc_name:
        print("[ERROR] missing encoder configuration")
        return

    encoder = get_encoder(enc_name)

    vectors = []
    labels = []
    missing_rows = []
    for row in rows:
        (
            row_id,
            emb_bytes,
            emb_dim,
            emb_encoder,
            subject,
            body,
            decision,
        ) = row
        vector = None
        if emb_bytes and emb_dim and emb_encoder == enc_name:
            arr = np.frombuffer(emb_bytes, dtype=np.float32)
            if arr.size == emb_dim:
                vector = arr
        if vector is None:
            text = clean_text(f"{(subject or '').strip()} {(body or '').strip()}")
            if text:
                missing_rows.append((row_id, text, decision))
            else:
                print(f"[WARN] skip id={row_id}: no embedding and empty text")
            continue
        vectors.append(vector)
        labels.append(decision)

    if missing_rows:
        texts = [item[1] for item in missing_rows]
        encoded = vectorize(texts, encoder)
        for (row_id, _text, decision), vec in zip(missing_rows, encoded):
            if vec.size == 0:
                print(f"[WARN] skip id={row_id}: unable to encode text")
                continue
            arr = np.asarray(vec, dtype=np.float32)
            vectors.append(arr)
            labels.append(decision)
            cur.execute(
                "UPDATE mails SET embedding=?, embedding_dim=?, embedding_encoder=?, subject=NULL, body=NULL WHERE id=?",
                (
                    sqlite3.Binary(arr.tobytes()),
                    int(arr.shape[0]),
                    enc_name,
                    row_id,
                ),
            )

    if not vectors:
        print("[WARN] no usable embeddings for training")
        conn.commit()
        conn.close()
        return

    # Stack embeddings and convert to float64 for sklearn.
    Xv = np.vstack(vectors).astype(np.float64)
    y = labels

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xv, y)

    # Persist to disk for future reuse.
    joblib.dump(clf, MODEL_DIR / "clf.joblib")
    joblib.dump(encoder, MODEL_DIR / "encoder.joblib")

    cur.execute(
        "INSERT INTO models(version,trained_at) VALUES (?,?)",
        (enc_name, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    print("[train] done")

def predict(cfg, auto_move=False):
    """Predict labels for pending emails and optionally move them."""

    conn = db_init(cfg)
    cur = conn.cursor()
    try:
        clf = joblib.load(MODEL_DIR / "clf.joblib")
        encoder = joblib.load(MODEL_DIR / "encoder.joblib")
    except Exception:
        print("[predict] no model, run retrain first")
        return

    cur.execute(
        "SELECT id, embedding, embedding_dim, embedding_encoder, subject, body, account, imap_uid "
        "FROM mails WHERE decision IS NULL AND (folder IS NULL OR UPPER(folder)='INBOX')"
    )
    rows = cur.fetchall()
    if not rows:
        conn.close()
        return

    enc_name = (cfg.get("model") or {}).get("encoder")
    expected_dim = clf.coef_.shape[1]

    prepared_rows = []
    vectors = []
    for row in rows:
        (
            row_id,
            emb_bytes,
            emb_dim,
            emb_encoder,
            subject,
            body,
            account,
            imap_uid,
        ) = row
        vector = None
        if (
            emb_bytes
            and emb_dim
            and emb_dim == expected_dim
            and (enc_name is None or emb_encoder == enc_name)
        ):
            arr = np.frombuffer(emb_bytes, dtype=np.float32)
            if arr.size == expected_dim:
                vector = arr
        if vector is None:
            text = clean_text(f"{(subject or '').strip()} {(body or '').strip()}")
            if not text:
                print(f"[predict] skip id={row_id}: missing embedding and text")
                continue
            vec = vectorize([text], encoder)
            if vec.size == 0:
                print(f"[predict] skip id={row_id}: encoder returned empty vector")
                continue
            arr = np.asarray(vec[0], dtype=np.float32)
            vector = arr
            cur.execute(
                "UPDATE mails SET embedding=?, embedding_dim=?, embedding_encoder=?, subject=NULL, body=NULL WHERE id=?",
                (
                    sqlite3.Binary(arr.tobytes()),
                    int(arr.shape[0]),
                    enc_name or emb_encoder,
                    row_id,
                ),
            )
        prepared_rows.append((row_id, account, imap_uid))
        vectors.append(vector)

    if not vectors:
        conn.commit()
        conn.close()
        return

    Xv = np.vstack(vectors).astype(np.float64)
    probs = clf.predict_proba(Xv)
    preds = clf.classes_[np.argmax(probs, axis=1)]

    accounts_map = {a["name"]: a for a in cfg.get("accounts", []) if a.get("name")}
    mail_types_cache = {}
    min_conf = float(cfg.get("model", {}).get("min_auto_move_confidence", 0.85))

    moves_by_account = defaultdict(list)
    for (row_meta, p, label) in zip(prepared_rows, probs, preds):
        id_, account, imap_uid = row_meta
        conf = float(np.max(p))
        decision = label
        acc = accounts_map.get(account)

        # Cache mail types to avoid hitting the filesystem for every message.
        mail_types = mail_types_cache.get(account)
        if mail_types is None:
            mail_types = load_account_mail_types(acc)
            mail_types_cache[account] = mail_types
        entries = mail_types.get("entries", {}) if mail_types else {}
        enabled = mail_types.get("enabled", set()) if mail_types else set()
        entry = entries.get(decision) if entries else None

        # Look up the target folder associated with the decision.
        target = None
        if entry:
            target = entry.get("target_folder") or (acc or {}).get("folders", {}).get("targets", {}).get(decision)
        elif acc:
            target = acc.get("folders", {}).get("targets", {}).get(decision)

        log_msg = f"[predict] {account} -> {decision} ({conf:.2f})"
        if not enabled:
            print(log_msg + " [no active rules]")
            continue
        if enabled and decision not in enabled:
            print(log_msg + " [disabled in config]")
            continue
        if entry and not entry.get("enabled", True):
            print(log_msg + " [rule disabled]")
            continue
        if not target:
            print(log_msg + " [no target folder]")
            continue

        print(log_msg)
        cur.execute(
            "UPDATE mails SET confidence=? WHERE id=?",
            (conf, id_),
        )

        # Optionally move the message if auto_move mode is enabled.
        if not (auto_move and acc and acc.get("mode", {}).get("auto_move", False) and conf > min_conf):
            continue
        if imap_uid is None:
            print(f"[WARN] missing UID for {account} message {id_}, skip move")
            continue
        try:
            uid_int = int(imap_uid)
        except Exception:
            print(f"[WARN] invalid UID '{imap_uid}' for {account}, skip move")
            continue
        moves_by_account[account].append((id_, uid_int, decision, target, conf))

    # Apply IMAP moves grouped by account to limit the number of connections.
    for account, items in moves_by_account.items():
        acc = accounts_map.get(account)
        if not acc:
            continue
        imap = acc.get("imap", {})
        if not imap:
            continue
        pwd = Path(imap["password_file"]).read_text().strip()
        with IMAPClient(imap["host"], port=imap["port"], ssl=imap["ssl"]) as srv:
            srv.login(imap["user"], pwd)
            srv.select_folder("INBOX")
            for row_id, uid, decision, target, conf in items:
                try:
                    srv.move([uid], target)
                except Exception as exc:
                    print(f"[move] {account} uid={uid} -> {target}: {exc}")
                    continue
                now_iso = datetime.utcnow().isoformat()
                cur.execute(
                    "UPDATE mails SET folder=?, decision=?, auto_moved=1, auto_moved_at=?, imap_uid=NULL, confidence=? WHERE id=?",
                    (target, decision, now_iso, conf, row_id),
                )
    conn.commit()
    conn.close()

def migrate_embeddings(cfg):
    """Maintenance helper: recompute missing embeddings and purge raw text."""

    conn = db_init(cfg)
    cur = conn.cursor()
    enc_name = (cfg.get("model") or {}).get("encoder")
    if not enc_name:
        print("[ERROR] missing encoder configuration")
        return

    encoder = get_encoder(enc_name)
    cur.execute(
        "SELECT id, subject, body, embedding, embedding_dim, embedding_encoder FROM mails"
    )
    rows = cur.fetchall()
    if not rows:
        print("[migrate] no data in mails table")
        return

    to_encode = []
    purge_ids = set()
    skipped = 0
    updated = 0

    for row in rows:
        (
            row_id,
            subject,
            body,
            emb_bytes,
            emb_dim,
            emb_encoder,
        ) = row
        text = clean_text(f"{(subject or '').strip()} {(body or '').strip()}")
        vector_valid = False
        if emb_bytes and emb_dim:
            arr = np.frombuffer(emb_bytes, dtype=np.float32)
            if arr.size == emb_dim and emb_encoder == enc_name:
                vector_valid = True
        if vector_valid:
            if subject or body:
                purge_ids.add(row_id)
            continue
        if text:
            to_encode.append((row_id, text))
        else:
            skipped += 1

    batch_size = 64
    for idx in range(0, len(to_encode), batch_size):
        batch = to_encode[idx : idx + batch_size]
        texts = [item[1] for item in batch]
        encoded = vectorize(texts, encoder)
        for (row_id, _), vec in zip(batch, encoded):
            if vec.size == 0:
                skipped += 1
                continue
            arr = np.asarray(vec, dtype=np.float32)
            cur.execute(
                "UPDATE mails SET embedding=?, embedding_dim=?, embedding_encoder=?, subject=NULL, body=NULL WHERE id=?",
                (
                    sqlite3.Binary(arr.tobytes()),
                    int(arr.shape[0]),
                    enc_name,
                    row_id,
                ),
            )
            updated += 1

    if purge_ids:
        cur.executemany(
            "UPDATE mails SET subject=NULL, body=NULL WHERE id=?",
            [(row_id,) for row_id in purge_ids],
        )

    conn.commit()
    conn.close()

    print(f"[migrate] embeddings updated: {updated}")
    if purge_ids:
        print(f"[migrate] purged text for {len(purge_ids)} rows")
    if skipped:
        print(f"[migrate] skipped rows without data: {skipped}")

def stats(cfg):
    """Display labelling statistics per account and auto-move mode."""

    conn = db_init(cfg)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT account,
               COUNT(*) AS total,
               SUM(CASE WHEN decision IS NOT NULL THEN 1 ELSE 0 END) AS labeled
        FROM mails
        GROUP BY account
        ORDER BY account
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("(no data in mails table)")
        return

    cur.execute(
        """
        SELECT account, decision, COUNT(*) AS cnt
        FROM mails
        WHERE decision IS NOT NULL
        GROUP BY account, decision
        """
    )
    decision_rows = cur.fetchall()
    conn.close()

    decisions_per_account = {}
    for account, decision, count in decision_rows:
        decisions_per_account.setdefault(account, {})[decision] = count

    auto_move_map = {
        a.get("name"): bool(a.get("mode", {}).get("auto_move", False))
        for a in cfg.get("accounts", [])
        if isinstance(a, dict) and a.get("name")
    }
    auto_summary = {
        True: {"total": 0, "labeled": 0, "decisions": {}},
        False: {"total": 0, "labeled": 0, "decisions": {}},
    }

    for account, total, labeled in rows:
        labeled = labeled or 0
        pending = (total or 0) - labeled
        percent = (labeled / total * 100) if total else 0.0
        auto_enabled = auto_move_map.get(account, False)
        auto_summary[auto_enabled]["total"] += total or 0
        auto_summary[auto_enabled]["labeled"] += labeled
        decisions = decisions_per_account.get(account, {})
        for decision, count in decisions.items():
            auto_summary[auto_enabled]["decisions"][decision] = (
                auto_summary[auto_enabled]["decisions"].get(decision, 0) + count
            )

        status = "enabled" if auto_enabled else "disabled"
        print(f"Account '{account}' (auto-move {status})")
        print(f"  Total messages : {total}")
        print(f"  Labeled        : {labeled} ({percent:.1f}%)")
        print(f"  Pending        : {pending}")
        if decisions:
            print("  Decisions      :")
            for decision, count in sorted(decisions.items(), key=lambda kv: (-kv[1], kv[0] or "")):
                label = decision if decision is not None else "<null>"
                print(f"    - {label}: {count}")
        else:
            print("  Decisions      : (none)")
        print()

    print("Summary by auto-move status:")
    for status_flag in (True, False):
        status = "enabled" if status_flag else "disabled"
        payload = auto_summary[status_flag]
        total = payload["total"]
        labeled = payload["labeled"]
        pending = total - labeled
        percent = (labeled / total * 100) if total else 0.0
        print(f"- Auto-move {status}: total={total}, labeled={labeled} ({percent:.1f}%), pending={pending}")
        decisions = payload["decisions"]
        if decisions:
            for decision, count in sorted(decisions.items(), key=lambda kv: (-kv[1], kv[0] or "")):
                label = decision if decision is not None else "<null>"
                print(f"    · {label}: {count}")
        else:
            print("    · (no labeled decisions)")

def loop(cfg):
    """Main loop: snapshot, prediction, periodic training."""

    scheduler_cfg = cfg.get("scheduler") or {}
    every = int(scheduler_cfg.get("poll_every_seconds", 600))
    retrain_every = int(scheduler_cfg.get("retrain_every_seconds", 86400))
    last_retrain = datetime.utcnow() - timedelta(seconds=retrain_every)
    while True:
        snapshot(cfg)
        predict(cfg, auto_move=True)
        if (datetime.utcnow() - last_retrain).total_seconds() >= retrain_every:
            retrain(cfg)
            last_retrain = datetime.utcnow()
        time.sleep(every)

# === CLI ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mailai.py [snapshot|predict|retrain|stats|migrate_embeddings|loop] [--auto] --config ...")
        sys.exit(1)
    cmd = sys.argv[1]
    cfg = load_cfg()
    if cmd == "snapshot":
        snapshot(cfg)
    elif cmd == "retrain":
        retrain(cfg)
    elif cmd == "predict":
        predict(cfg, auto_move="--auto" in sys.argv)
    elif cmd == "stats":
        stats(cfg)
    elif cmd == "migrate_embeddings":
        migrate_embeddings(cfg)
    elif cmd == "loop":
        loop(cfg)
    else:
        print("Unknown command", cmd)
        sys.exit(1)
