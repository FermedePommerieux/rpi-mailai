#!/usr/bin/env python3
import json
import os, sys, time, yaml, sqlite3, email, re, random
from datetime import datetime
from pathlib import Path
from imapclient import IMAPClient
from mailparser import parse_from_bytes
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# === CONFIG PATHS ===
CFG_PATH = Path(os.environ.get("APP_CONFIG", "/config/config.yml"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MAIL_TYPES_DIR = Path(os.environ.get("MAIL_TYPES_DIR", "/config/account_types"))
DB_PATH = DATA_DIR / "db" / "mailai.sqlite"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === HELPERS ===
def load_cfg():
    if not CFG_PATH.exists():
        print(f"[ERROR] config missing: {CFG_PATH}", file=sys.stderr)
        sys.exit(2)
    with open(CFG_PATH,"r") as f: return yaml.safe_load(f)

def db_init():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS mails (
        id INTEGER PRIMARY KEY,
        account TEXT, msgid TEXT, subject TEXT, body TEXT,
        folder TEXT, date TEXT, decision TEXT, confidence REAL,
        UNIQUE(account,msgid)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY, version TEXT, trained_at TEXT
    )""")
    conn.commit()
    return conn

def clean_text(s):
    if not s: return ""
    return re.sub(r"\s+"," ", s).strip()

def vectorize(texts, encoder):
    emb = encoder.encode(texts, show_progress_bar=False)
    return np.array(emb)


def load_account_mail_types(acc):
    if not acc:
        return {}
    cfg_path = acc.get("mail_types_config")
    if not cfg_path:
        default_path = MAIL_TYPES_DIR / f"{acc['name']}.json"
        cfg_path = default_path if default_path.exists() else None
    if not cfg_path:
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        print(f"[WARN] invalid mail types JSON for {acc['name']}: {exc}")
        return {}
    entries = payload.get("types", [])
    index = {e.get("key"): e for e in entries if e.get("key")}
    enabled = {k for k, v in index.items() if v.get("enabled", True)}
    return {"path": str(cfg_path), "entries": index, "enabled": enabled}

# === PIPELINE STEPS ===
def snapshot(cfg):
    conn = db_init()
    cur = conn.cursor()
    for acc in cfg.get("accounts", []):
        print(f"{datetime.now()} [{acc['name']}] snapshot...")
        imap = acc["imap"]; pwd = Path(imap["password_file"]).read_text().strip()
        with IMAPClient(imap["host"], port=imap["port"], ssl=imap["ssl"]) as srv:
            srv.login(imap["user"], pwd)
            srv.select_folder("INBOX")
            uids = srv.search()
            for uid, data in srv.fetch(uids, ["BODY[]","ENVELOPE"]).items():
                msgid = data[b"ENVELOPE"].message_id.decode() if data[b"ENVELOPE"].message_id else str(uid)
                subject = data[b"ENVELOPE"].subject.decode() if data[b"ENVELOPE"].subject else ""
                body = ""
                try:
                    mp = parse_from_bytes(data[b"BODY[]"])
                    body = mp.text_plain[0] if mp.text_plain else (mp.body or "")
                except Exception: pass
                cur.execute("INSERT OR IGNORE INTO mails (account,msgid,subject,body,folder,date) VALUES (?,?,?,?,?,?)",
                    (acc["name"], msgid, clean_text(subject), clean_text(body), "INBOX", datetime.now().isoformat()))
    conn.commit(); conn.close()

def retrain(cfg):
    conn = db_init(); cur = conn.cursor()
    cur.execute("SELECT subject,body,decision FROM mails WHERE decision IS NOT NULL")
    rows = cur.fetchall()
    if not rows:
        print("[WARN] no labeled data"); return
    X = [clean_text(r[0]+" "+r[1]) for r in rows]
    y = [r[2] for r in rows]
    enc_name = cfg["model"]["encoder"]
    print(f"[train] loading encoder {enc_name} ...")
    encoder = SentenceTransformer(enc_name)
    Xv = vectorize(X, encoder)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xv,y)
    joblib.dump(clf, MODEL_DIR/"clf.joblib")
    joblib.dump(encoder, MODEL_DIR/"encoder.joblib")
    cur.execute("INSERT INTO models(version,trained_at) VALUES (?,?)",
                (enc_name, datetime.now().isoformat()))
    conn.commit(); conn.close()
    print("[train] done")

def predict(cfg, auto_move=False):
    conn = db_init(); cur = conn.cursor()
    try:
        clf = joblib.load(MODEL_DIR/"clf.joblib")
        encoder = joblib.load(MODEL_DIR/"encoder.joblib")
    except Exception:
        print("[predict] no model, run retrain first"); return
    cur.execute("SELECT id,subject,body,account FROM mails WHERE decision IS NULL")
    rows = cur.fetchall()
    if not rows: return
    X = [clean_text(r[1]+" "+r[2]) for r in rows]
    Xv = vectorize(X, encoder)
    probs = clf.predict_proba(Xv)
    preds = clf.classes_[np.argmax(probs, axis=1)]
    accounts_map = {a["name"]: a for a in cfg.get("accounts", [])}
    mail_types_cache = {}
    min_conf = float(cfg.get("model", {}).get("min_auto_move_confidence", 0.85))

    for (id_,_,_,account),p,label in zip(rows,probs,preds):
        conf = np.max(p)
        decision = label
        cur.execute("UPDATE mails SET decision=?,confidence=? WHERE id=?",(decision,float(conf),id_))
        acc = accounts_map.get(account)
        mail_types = mail_types_cache.get(account)
        if mail_types is None:
            mail_types = load_account_mail_types(acc)
            mail_types_cache[account] = mail_types

        entries = mail_types.get("entries", {}) if mail_types else {}
        enabled = mail_types.get("enabled", set()) if mail_types else set()
        entry = entries.get(decision)

        log_msg = f"[predict] {account} -> {decision} ({conf:.2f})"
        if enabled and decision not in enabled:
            print(log_msg + " [désactivé dans config]")
            continue
        if entry and not entry.get("enabled", True):
            print(log_msg + " [règle désactivée]")
            continue

        print(log_msg)

        if not (auto_move and acc and acc.get("mode", {}).get("auto_move", False) and conf > min_conf):
            continue

        imap = acc["imap"]; pwd = Path(imap["password_file"]).read_text().strip()
        target = None
        if entry:
            target = entry.get("target_folder") or acc.get("folders", {}).get("targets", {}).get(decision)
        else:
            target = acc.get("folders", {}).get("targets", {}).get(decision)

        if not target:
            continue

        with IMAPClient(imap["host"], port=imap["port"], ssl=imap["ssl"]) as srv:
            srv.login(imap["user"], pwd)
            try:
                srv.move([id_], target)
            except Exception as e:
                print(f"[move] {e}")
    conn.commit(); conn.close()

def loop(cfg):
    every = int(cfg["scheduler"].get("poll_every_seconds",600))
    while True:
        snapshot(cfg)
        predict(cfg, auto_move=True)
        time.sleep(every)

# === CLI ===
if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: mailai.py [snapshot|predict|retrain|loop] --config ..."); sys.exit(1)
    cmd = sys.argv[1]
    cfg = load_cfg()
    if cmd=="snapshot": snapshot(cfg)
    elif cmd=="retrain": retrain(cfg)
    elif cmd=="predict": predict(cfg, auto_move="--auto" in sys.argv)
    elif cmd=="loop": loop(cfg)
    else:
        print("Unknown command",cmd); sys.exit(1)
