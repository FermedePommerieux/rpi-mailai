#!/usr/bin/env python3
"""MailAI: automatisation de la classification des emails.

Ce script regroupe l'ensemble de la chaîne de traitement de l'application :

* Initialisation/maintenance de la base SQLite qui stocke les emails vus et
  leurs décisions associées.
* Synchronisation des boîtes IMAP pour récupérer le contenu brut des
  messages et garder une trace de leur dernière position côté serveur.
* Entraînement et utilisation d'un modèle de classification (Sentence
  Transformers + régression logistique) pour suggérer ou appliquer des règles
  de tri automatique.
* Outils de monitoring (statistiques) et de scheduling (boucle continue).

L'objectif de cette passe de documentation est de rendre le flot de
traitement explicitement lisible : chaque étape est abondamment commentée pour
faciliter l'onboarding ou le debug futur.
"""

import json
import os, sys, time, yaml, sqlite3, email, re, random
from collections import defaultdict
from datetime import datetime, timedelta
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

# L'entraînement et les prédictions nécessitent la présence d'un dossier pour
# stocker les artefacts (modèles, encodeur). On crée donc le dossier dès le
# chargement du module pour éviter les races lors d'exécutions parallèles.
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === HELPERS ===
def load_cfg():
    """Charge la configuration principale de l'application.

    La configuration YAML décrit entre autres les comptes IMAP à suivre, les
    paramètres du modèle et la planification.
    """

    # On s'assure que le fichier est présent avant de tenter de le parser :
    # sans configuration l'application ne peut pas fonctionner.
    if not CFG_PATH.exists():
        print(f"[ERROR] config missing: {CFG_PATH}", file=sys.stderr)
        sys.exit(2)

    # Chargement brut du YAML -> dictionnaire Python.
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

def db_init():
    """Prépare la base SQLite et effectue les migrations minimales.

    On crée les tables nécessaires si elles n'existent pas encore puis on
    applique quelques migrations simples (ajout de colonnes). La fonction
    renvoie un handle `sqlite3.Connection` réutilisable par l'appelant.
    """

    # Création du répertoire de travail si besoin avant d'ouvrir la base.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Table principale : un enregistrement par email unique (compte + Message-ID).
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

    # --- migrations légères ---
    # On inspecte la structure courante puis on ajoute les colonnes manquantes.
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

    # Table pour suivre l'historique des entraînements réalisés.
    c.execute(
        """CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY, version TEXT, trained_at TEXT
    )"""
    )

    conn.commit()
    return conn

def clean_text(s):
    """Nettoie rapidement un texte pour l'encoder plus facilement."""

    if not s:
        return ""

    # Suppression des espaces multiples et normalisation des sauts de ligne.
    return re.sub(r"\s+", " ", s).strip()

def vectorize(texts, encoder):
    """Applique l'encodeur SentenceTransformer et renvoie une matrice numpy."""

    emb = encoder.encode(texts, show_progress_bar=False)
    return np.array(emb)


def canonical_folder_name(name: str) -> str:
    """Normalise le nom d'un dossier IMAP (majuscules, séparateur /)."""

    if not name:
        return ""
    if isinstance(name, bytes):
        name = name.decode("utf-8", "ignore")
    cleaned = name.replace("\"", "").replace("'", "")
    cleaned = cleaned.replace(".", "/").replace("\\", "/")
    cleaned = re.sub(r"/+", "/", cleaned)
    return cleaned.strip().upper()


def load_account_mail_types(acc):
    """Charge les règles de typage personnalisées associées à un compte."""

    if not acc:
        return {}

    # On recherche d'abord un chemin défini explicitement dans la config du
    # compte, sinon on tente un fallback basé sur le nom du compte.
    cfg_path = acc.get("mail_types_config")
    if not cfg_path:
        default_path = MAIL_TYPES_DIR / f"{acc['name']}.json"
        cfg_path = default_path if default_path.exists() else None
    if not cfg_path:
        return {}

    # Lecture du fichier JSON qui décrit les règles. La robustesse est de mise
    # car un fichier manquant ou mal formé ne doit pas faire planter la synchro.
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        print(f"[WARN] invalid mail types JSON for {acc['name']}: {exc}")
        return {}

    # Indexation rapide par clé pour simplifier les lookups plus tard.
    entries = payload.get("types", [])
    index = {e.get("key"): e for e in entries if e.get("key")}
    enabled = {k for k, v in index.items() if v.get("enabled", True)}
    return {"path": str(cfg_path), "entries": index, "enabled": enabled}


def _resolve_entry_folders(account_cfg, entry):
    """Détermine les dossiers IMAP liés à une règle de typage donnée."""

    folders = set()
    if not entry:
        return folders

    raw = entry.get("source_folders") or []
    if isinstance(raw, str):
        raw = [raw]
    for item in raw:
        if item:
            folders.add(item)

    # Si la règle définit un dossier cible, on l'ajoute aussi pour surveiller
    # les mouvements et détecter les classifications existantes.
    target = entry.get("target_folder")
    if not target:
        target = account_cfg.get("folders", {}).get("targets", {}).get(entry.get("key"))
    if target:
        folders.add(target)
    return folders


def _collect_archive_roots(account_cfg):
    """Construit la liste des dossiers considérés comme archives."""

    roots = set()
    folders_cfg = account_cfg.get("folders", {}) if account_cfg else {}
    archive_values = []
    for key in ("archive", "archives", "archive_root", "archive_roots"):
        val = folders_cfg.get(key)
        if isinstance(val, list):
            archive_values.extend(val)
        elif val:
            archive_values.append(val)
    for item in archive_values:
        if item:
            roots.add(canonical_folder_name(item))
    return roots


def _is_archive_folder(folder, archive_roots):
    """Détermine si un dossier correspond à l'archive (d'où exclusion auto)."""

    canon = canonical_folder_name(folder)
    if not canon:
        return False
    if canon in archive_roots:
        return True
    parts = [p for p in canon.split("/") if p]
    if any(p.startswith("ARCHIVE") or p.endswith("ARCHIVE") for p in parts):
        return True
    for root in archive_roots:
        if root and (canon == root or canon.startswith(f"{root}/")):
            return True
    return False

# === PIPELINE STEPS ===
def snapshot(cfg):
    """Réalise une synchronisation IMAP et met à jour la base locale.

    Cette étape collecte les emails des dossiers surveillés, met à jour les
    métadonnées (sujet, corps, dernier dossier vu) et tient à jour les
    décisions prises manuellement par l'utilisateur afin d'alimenter l'apprentissage.
    """

    conn = db_init()
    cur = conn.cursor()

    # Ces valeurs servent de fallback si un message disparaît d'INBOX : on les
    # utilise pour marquer les mails comme supprimés / spam automatiquement.
    delete_pref_keys = {"SPAM", "INDESIRABLE", "JUNK", "PROMOTION", "PROMOTIONS", "NEWSLETTER"}

    for acc in cfg.get("accounts", []):
        account_name = acc.get("name")
        if not account_name:
            continue

        now_iso = datetime.utcnow().isoformat()
        print(f"{datetime.now()} [{account_name}] snapshot...")

        # On récupère les règles de typage et on prépare des structures pour
        # suivre la correspondance dossier -> décision.
        mail_types = load_account_mail_types(acc)
        entries = mail_types.get("entries", {})
        enabled_keys = {k for k in mail_types.get("enabled", set()) if entries.get(k)}
        folder_sources = defaultdict(set)
        folder_decisions = {}
        delete_fallback = None

        # On parcourt les règles actives pour déterminer les dossiers sources à
        # surveiller ainsi que le fallback suppression (spam, promotions...).
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

        archive_roots = _collect_archive_roots(acc)
        imap = acc["imap"]
        pwd = Path(imap["password_file"]).read_text().strip()

        # On surveille systématiquement l'INBOX et, pour chaque règle, les
        # dossiers associés (sources + cibles pour détecter les classifications).
        watch_folders = {"INBOX"}
        for folder in folder_sources:
            watch_folders.add(folder)

        seen = set()

        # Connexion IMAP : le contexte assure la déconnexion propre.
        with IMAPClient(imap["host"], port=imap["port"], ssl=imap["ssl"]) as srv:
            srv.login(imap["user"], pwd)
            for folder in sorted(watch_folders):
                try:
                    # On se met en lecture seule pour éviter de modifier les flags côté serveur.
                    srv.select_folder(folder, readonly=True)
                except Exception as exc:
                    print(f"[WARN] {account_name} select {folder}: {exc}")
                    continue
                try:
                    # Récupération de tous les UID : on ne filtre pas pour ne
                    # manquer aucun message récent.
                    uids = srv.search()
                except Exception as exc:
                    print(f"[WARN] {account_name} search {folder}: {exc}")
                    continue
                if not uids:
                    continue
                try:
                    fetched = srv.fetch(uids, ["BODY.PEEK[]", "ENVELOPE"])
                except Exception as exc:
                    print(f"[WARN] {account_name} fetch {folder}: {exc}")
                    continue

                for uid, data in fetched.items():
                    env = data.get(b"ENVELOPE")
                    msgid = None
                    subject = ""
                    if env:
                        if env.message_id:
                            try:
                                msgid = env.message_id.decode()
                            except Exception:
                                msgid = str(env.message_id)
                        if env.subject:
                            try:
                                subject = env.subject.decode()
                            except Exception:
                                subject = str(env.subject)
                    if not msgid:
                        # Fallback: on compose un identifiant stable basé sur
                        # le dossier et l'UID IMAP.
                        msgid = f"{folder}:{uid}"

                    body = ""
                    raw_body = data.get(b"BODY[]") or data.get(b"BODY.PEEK[]")
                    try:
                        if raw_body:
                            mp = parse_from_bytes(raw_body)
                            body = mp.text_plain[0] if mp.text_plain else (mp.body or "")
                    except Exception:
                        # Parsing parfois fragile (emails mal formés) : on ignore l'erreur.
                        pass

                    # Le nettoyage permet d'éviter d'insérer des chaînes
                    # énormes et mal normalisées dans la base.
                    body = clean_text(body)
                    subject = clean_text(subject)

                    canon_folder = canonical_folder_name(folder)
                    decision_for_folder = folder_decisions.get(canon_folder)
                    archive_hit = _is_archive_folder(folder, archive_roots)

                    # Trace que le message a été vu durant ce snapshot.
                    seen.add((account_name, msgid))

                    cur.execute(
                        "SELECT id, folder, decision, auto_moved FROM mails WHERE account=? AND msgid=?",
                        (account_name, msgid),
                    )
                    row = cur.fetchone()
                    updates = []
                    params = []
                    if row:
                        row_id, prev_folder, prev_decision, prev_auto = row
                        if prev_folder != folder:
                            # Déplacement détecté -> on met à jour dossier, UID et timestamps.
                            updates.append("folder=?")
                            params.append(folder)
                            updates.append("imap_uid=?")
                            params.append(str(uid))
                            updates.append("last_seen_at=?")
                            params.append(now_iso)
                            if prev_auto:
                                # Si le message avait été auto-déplacé on reset
                                # le flag pour refléter l'action manuelle.
                                updates.append("auto_moved=0")
                                updates.append("auto_moved_at=NULL")
                        else:
                            # Pas de déplacement : on rafraîchit simplement l'horodatage et l'UID.
                            updates.append("last_seen_at=?")
                            params.append(now_iso)
                            updates.append("imap_uid=?")
                            params.append(str(uid))
                        if subject:
                            updates.append("subject=?")
                            params.append(subject)
                        if body:
                            updates.append("body=?")
                            params.append(body)
                        if not archive_hit:
                            if decision_for_folder:
                                if prev_decision != decision_for_folder:
                                    # Le dossier surveillé correspond à une règle -> on met à jour la décision.
                                    updates.append("decision=?")
                                    params.append(decision_for_folder)
                                    updates.append("confidence=?")
                                    params.append(1.0)
                                    updates.append("auto_moved=0")
                                    updates.append("auto_moved_at=?")
                                    params.append(None)
                            elif prev_decision is not None and canon_folder == "INBOX":
                                # Retour dans l'INBOX : on considère que la
                                # décision est annulée.
                                updates.append("decision=NULL")
                                updates.append("confidence=NULL")
                                updates.append("auto_moved=0")
                                updates.append("auto_moved_at=NULL")
                        if updates:
                            set_clause = ",".join(updates)
                            cur.execute(f"UPDATE mails SET {set_clause} WHERE id=?", (*params, row_id))
                    else:
                        # Nouveau message : on insère une ligne complète avec les métadonnées courantes.
                        cur.execute(
                            "INSERT INTO mails (account,msgid,subject,body,folder,date,decision,confidence,imap_uid,last_seen_at,auto_moved) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                account_name,
                                msgid,
                                subject,
                                body,
                                folder,
                                now_iso,
                                decision_for_folder if decision_for_folder and not archive_hit else None,
                                1.0 if decision_for_folder and not archive_hit else None,
                                str(uid),
                                now_iso,
                                0,
                            ),
                        )

        # Traitement des messages qui ont disparu de l'INBOX (probablement
        # supprimés ou classés en spam côté client) : on les marque avec le
        # fallback défini.
        if delete_fallback and seen:
            cur.execute(
                "SELECT id, msgid, folder FROM mails WHERE account=? AND folder='INBOX'",
                (account_name,),
            )
            for row_id, msgid, folder in cur.fetchall():
                if (account_name, msgid) in seen:
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
    """Réentraîne le modèle de classification à partir des données étiquetées."""

    conn = db_init()
    cur = conn.cursor()
    cur.execute("SELECT subject,body,decision FROM mails WHERE decision IS NOT NULL")
    rows = cur.fetchall()
    if not rows:
        print("[WARN] no labeled data")
        return

    # Préparation des features texte (concat sujet + corps) avec nettoyage.
    X = [clean_text(f"{(r[0] or '').strip()} {(r[1] or '').strip()}") for r in rows]
    y = [r[2] for r in rows]

    enc_name = cfg["model"]["encoder"]
    print(f"[train] loading encoder {enc_name} ...")
    encoder = SentenceTransformer(enc_name)

    # Passage dans l'encodeur SentenceTransformer pour obtenir des embeddings.
    Xv = vectorize(X, encoder)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xv, y)

    # Sauvegarde sur disque pour réutilisation future.
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
    """Effectue des prédictions sur les emails en attente et optionnellement les déplace."""

    conn = db_init()
    cur = conn.cursor()
    try:
        clf = joblib.load(MODEL_DIR / "clf.joblib")
        encoder = joblib.load(MODEL_DIR / "encoder.joblib")
    except Exception:
        print("[predict] no model, run retrain first")
        return

    cur.execute(
        "SELECT id,subject,body,account,imap_uid FROM mails WHERE decision IS NULL AND (folder IS NULL OR UPPER(folder)='INBOX')"
    )
    rows = cur.fetchall()
    if not rows:
        conn.close()
        return

    # Conversion en embeddings pour le modèle.
    X = [clean_text((r[1] or "") + " " + (r[2] or "")) for r in rows]
    Xv = vectorize(X, encoder)
    probs = clf.predict_proba(Xv)
    preds = clf.classes_[np.argmax(probs, axis=1)]

    accounts_map = {a["name"]: a for a in cfg.get("accounts", []) if a.get("name")}
    mail_types_cache = {}
    min_conf = float(cfg.get("model", {}).get("min_auto_move_confidence", 0.85))

    moves_by_account = defaultdict(list)
    for (row, p, label) in zip(rows, probs, preds):
        id_, _, _, account, imap_uid = row
        conf = float(np.max(p))
        decision = label
        acc = accounts_map.get(account)

        # Cache des mail types pour éviter de recharger sur disque à chaque message.
        mail_types = mail_types_cache.get(account)
        if mail_types is None:
            mail_types = load_account_mail_types(acc)
            mail_types_cache[account] = mail_types
        entries = mail_types.get("entries", {}) if mail_types else {}
        enabled = mail_types.get("enabled", set()) if mail_types else set()
        entry = entries.get(decision) if entries else None

        # Recherche du dossier cible associé à la décision.
        target = None
        if entry:
            target = entry.get("target_folder") or (acc or {}).get("folders", {}).get("targets", {}).get(decision)
        elif acc:
            target = acc.get("folders", {}).get("targets", {}).get(decision)

        log_msg = f"[predict] {account} -> {decision} ({conf:.2f})"
        if not enabled:
            print(log_msg + " [aucune règle active]")
            continue
        if enabled and decision not in enabled:
            print(log_msg + " [désactivé dans config]")
            continue
        if entry and not entry.get("enabled", True):
            print(log_msg + " [règle désactivée]")
            continue
        if not target:
            print(log_msg + " [pas de dossier cible]")
            continue

        print(log_msg)
        cur.execute(
            "UPDATE mails SET confidence=? WHERE id=?",
            (conf, id_),
        )

        # Optionnellement on déplace le message si le mode auto_move est actif.
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

    # Application des déplacements IMAP regroupés par compte pour limiter les connexions.
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

def stats(cfg):
    """Affiche des statistiques de labeling par compte et par mode auto-move."""

    conn = db_init()
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
    """Boucle principale : snapshot, prédiction, entraînement périodique."""

    every = int(cfg["scheduler"].get("poll_every_seconds", 600))
    retrain_every = int(cfg["scheduler"].get("retrain_every_seconds", 86400))
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
        print("Usage: mailai.py [snapshot|predict|retrain|loop] --config ...")
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
    elif cmd == "loop":
        loop(cfg)
    else:
        print("Unknown command", cmd)
        sys.exit(1)
