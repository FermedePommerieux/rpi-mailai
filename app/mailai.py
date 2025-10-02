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

# Cache process-local pour éviter de recharger plusieurs fois l'encodeur dans
# une même exécution (notamment pour snapshot/maintenance).
_ENCODER_CACHE = {}

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

def _looks_like_hash(value):
    """Détecte si une chaîne ressemble à un hash hexadécimal SHA-256."""

    if not value or not isinstance(value, str):
        return False
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def compute_mail_key(account, raw_identifier, salt):
    """Retourne un identifiant anonymisé stable pour un email."""

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
    """Prépare la base SQLite et effectue les migrations minimales.

    On crée les tables nécessaires si elles n'existent pas encore puis on
    applique quelques migrations simples (ajout de colonnes). La fonction
    renvoie un handle `sqlite3.Connection` réutilisable par l'appelant.
    """

    # Création du répertoire de travail si besoin avant d'ouvrir la base.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cfg = cfg if isinstance(cfg, dict) else {}
    hash_salt = str(cfg.get("hash_salt") or "")

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
    if "embedding" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN embedding BLOB")
    if "embedding_dim" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN embedding_dim INTEGER")
    if "embedding_encoder" not in cols:
        c.execute("ALTER TABLE mails ADD COLUMN embedding_encoder TEXT")

    # Purge automatique du texte lorsque l'embedding est présent afin de
    # limiter l'exposition des données sensibles dans la base.
    c.execute(
        "UPDATE mails SET subject=NULL, body=NULL "
        "WHERE embedding IS NOT NULL AND (subject IS NOT NULL OR body IS NOT NULL)"
    )

    # Migration : anonymisation des identifiants de messages via hash salé.
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

    # S'assure qu'un index unique explicite existe sur (account, msgid).
    c.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_mails_account_msgid ON mails(account, msgid)"
    )

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

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    emb = encoder.encode(texts, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def get_encoder(enc_name):
    """Charge et met en cache un encodeur SentenceTransformer."""

    encoder = _ENCODER_CACHE.get(enc_name)
    if encoder is None:
        print(f"[encoder] loading {enc_name} ...")
        encoder = SentenceTransformer(enc_name)
        _ENCODER_CACHE[enc_name] = encoder
    return encoder


def compute_embedding_bytes(text, encoder):
    """Encode un texte et renvoie le couple (bytes, dimension)."""

    vec = vectorize([text], encoder)
    if vec.size == 0:
        return None, None
    arr = vec[0]
    return arr.tobytes(), int(arr.shape[0])


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

    conn = db_init(cfg)
    cur = conn.cursor()

    # Ces valeurs servent de fallback si un message disparaît d'INBOX : on les
    # utilise pour marquer les mails comme supprimés / spam automatiquement.
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
                        # Fallback: on compose un identifiant stable basé sur
                        # le dossier et l'UID IMAP.
                        raw_identifier = f"{folder}:{uid}"

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
                    combined_text = clean_text(f"{subject} {body}")

                    canon_folder = canonical_folder_name(folder)
                    decision_for_folder = folder_decisions.get(canon_folder)
                    archive_hit = _is_archive_folder(folder, archive_roots)

                    # Trace que le message a été vu durant ce snapshot.
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
                        if encoder_available and (
                            prev_embedding is None
                            or not prev_dim
                            or prev_encoder != enc_name
                        ):
                            needs_embedding = True
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
                        # Nouveau message : on insère une ligne complète avec les métadonnées courantes.
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

        # Traitement des messages qui ont disparu de l'INBOX (probablement
        # supprimés ou classés en spam côté client) : on les marque avec le
        # fallback défini.
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
    """Réentraîne le modèle de classification à partir des données étiquetées."""

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

    # Empilement des embeddings et conversion vers float64 pour sklearn.
    Xv = np.vstack(vectors).astype(np.float64)
    y = labels

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

def migrate_embeddings(cfg):
    """Maintenance : recalcule les embeddings manquants et purge le texte brut."""

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
    """Affiche des statistiques de labeling par compte et par mode auto-move."""

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
