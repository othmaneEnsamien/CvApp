"""
Application Streamlit pour le traitement avancé de CVs candidats.

Fonctions principales :
- Upload de CVs (PDF, DOCX, TXT).
- Extraction automatique du texte des CVs.
- Analyse simple du profil (nom, email, téléphone, compétences).
- Matching par rapport à un profil de poste (job description).
- Classement des candidats (Shortlist / À revoir / Rejet).
- Envoi d'emails automatiques et personnalisables aux candidats.

Les commentaires sont pensés pour un développeur Java :
- Un CV est représenté par un dict Python (équivalent à un Map<String, Object>).
- Une liste de CVs est un List[dict].
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import pdfplumber
import streamlit as st
from rapidfuzz import fuzz
from docx import Document
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import mimetypes


# =========================
# Modèle de données métier
# =========================


@dataclass
class ParsedCV:
    """
    Représentation structurée d'un CV.

    En Java, on ferait une classe avec des getters/setters.
    Ici, @dataclass génère automatiquement constructeur + repr.
    """

    filename: str
    # Bytes du fichier original (pour téléchargement après analyse).
    # En Java : byte[].
    file_bytes: bytes
    # Type MIME (ex: 'application/pdf').
    mime_type: str
    raw_text: str
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    # Liste de compétences éventuellement détectées (optionnelle, dépend du poste)
    skills: List[str]
    years_experience: Optional[int]


class _LocalUploadedFile:
    """
    Adaptateur très simple pour traiter un fichier du système comme
    un UploadedFile Streamlit (pour réutiliser le même pipeline).

    En Java, ce serait une classe wrapper autour d'un InputStream + métadonnées.
    """

    def __init__(self, path: str):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._data = f.read()
        self.type = mimetypes.guess_type(self.name)[0] or "application/octet-stream"

    def getbuffer(self):
        return memoryview(self._data)


# ======================
# Extraction texte de CV
# ======================


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extrait le texte d'un PDF (CV) via pdfplumber.

    :param uploaded_file: objet UploadedFile Streamlit (binaire).
    :return: texte brut du PDF.
    """
    buffer = io.BytesIO(uploaded_file.getbuffer())
    text_parts: List[str] = []
    with pdfplumber.open(buffer) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(uploaded_file) -> str:
    """
    Extrait le texte d'un DOCX via python-docx.
    """
    buffer = io.BytesIO(uploaded_file.getbuffer())
    doc = Document(buffer)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_from_txt(uploaded_file) -> str:
    """
    Extrait le texte d'un fichier texte simple.
    """
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def extract_text_from_cv(uploaded_file) -> str:
    """
    Routeur qui appelle la bonne fonction d'extraction selon l'extension.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if filename.endswith((".docx", ".doc")):
        return extract_text_from_docx(uploaded_file)
    if filename.endswith((".txt",)):
        return extract_text_from_txt(uploaded_file)
    # Fallback : tentative lecture texte brut.
    return extract_text_from_txt(uploaded_file)


# =============================
# Parsing / analyse d'un CV brut
# =============================


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"\+?\d[\d\s/().-]{7,}")


def parse_basic_fields(text: str) -> Dict[str, Optional[str]]:
    """
    Essaie d'extraire les champs simples : email, téléphone, nom.

    - email : première adresse trouvée par regex.
    - téléphone : premier pattern compatible trouvé.
    - nom : heuristique simple -> première ligne avec plusieurs mots en lettres.
    """
    email_match = EMAIL_REGEX.search(text)
    email = email_match.group(0) if email_match else None

    phone_match = PHONE_REGEX.search(text)
    phone = phone_match.group(0) if phone_match else None

    # Heuristique pour le nom : on prend la première ligne du CV qui ressemble à un nom.
    name = None
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped.split()) >= 2 and any(c.isalpha() for c in stripped):
            name = stripped
            break

    return {"name": name, "email": email, "phone": phone}


def estimate_years_experience(text: str) -> Optional[int]:
    """
    Estimation très approximative du nombre d'années d'expérience.

    Heuristiques multilingues (français + anglais) :
      - Rechercher des patterns comme 'X ans', 'X années', 'X years', 'X year',
        éventuellement suivis de 'experience' / 'd'expérience'.
      - Retourner la valeur max trouvée (supposée être l'expérience totale).
    """
    text_low = text.lower()

    # On cherche d'abord des formes explicites de type "X ans d'expérience" / "X years of experience".
    exp_strong: List[int] = []
    for m in re.finditer(
        r"(\d{1,2})\s*\+?\s*(?:ans|année|années|years|year)\b",
        text_low,
    ):
        try:
            val = int(m.group(1))
        except ValueError:
            continue

        if not (0 < val < 50):
            continue

        start = max(0, m.start() - 40)
        end = min(len(text_low), m.end() + 60)
        ctx = text_low[start:end]

        # Exclure des indices d'âge (souvent "âge: 35 ans", "age 35 years").
        if any(k in ctx for k in ["âge", "age", "date de naissance", "born", "birth"]):
            continue

        # Garder en priorité les occurrences clairement liées à l'expérience.
        if any(
            k in ctx
            for k in [
                "expérience",
                "experience",
                "d'expérience",
                "of experience",
                "years of exp",
                "ans d'exp",
            ]
        ):
            exp_strong.append(val)

    if exp_strong:
        return max(exp_strong)

    # Fallback (plus faible) : "X ans" ou "X years" sans le mot "experience".
    # On l'autorise uniquement si aucune forme forte n'a été détectée, pour éviter
    # de confondre avec un âge ou des pourcentages.
    exp_weak: List[int] = []
    for m in re.finditer(
        r"(\d{1,2})\s*\+?\s*(?:ans|année|années|years|year)\b",
        text_low,
    ):
        try:
            val = int(m.group(1))
        except ValueError:
            continue

        if not (0 < val < 50):
            continue

        start = max(0, m.start() - 40)
        end = min(len(text_low), m.end() + 60)
        ctx = text_low[start:end]

        if any(k in ctx for k in ["âge", "age", "date de naissance", "born", "birth"]):
            continue

        # Si le contexte ressemble à une performance ("35%"), on ignore.
        if "%" in ctx:
            continue

        exp_weak.append(val)

    if exp_weak:
        return max(exp_weak)

    return None


def parse_cv(uploaded_file) -> ParsedCV:
    """
    Pipeline complet :
      - Extraction texte
      - Parsing des champs simples
      - Extraction compétences
      - Estimation années d'expérience
    """
    file_bytes = bytes(uploaded_file.getbuffer())
    mime_type = getattr(uploaded_file, "type", "") or "application/octet-stream"
    raw_text = extract_text_from_cv(uploaded_file)
    basics = parse_basic_fields(raw_text)
    # Les compétences seront déterminées plus tard en fonction du poste (plus dynamique).
    skills: List[str] = []
    years = estimate_years_experience(raw_text)

    return ParsedCV(
        filename=uploaded_file.name,
        file_bytes=file_bytes,
        mime_type=mime_type,
        raw_text=raw_text,
        name=basics["name"],
        email=basics["email"],
        phone=basics["phone"],
        skills=skills,
        years_experience=years,
    )


def parse_cv_from_path(path: str) -> ParsedCV:
    """
    Variante de parse_cv pour un fichier déjà présent sur le disque.

    :param path: chemin absolu ou relatif vers un fichier CV (PDF, DOCX, TXT).
    """
    local_file = _LocalUploadedFile(path)
    return parse_cv(local_file)


# ============================
# Matching avec profil de poste
# ============================


def parse_job_profile(
    text: str, explicit_skills: Optional[List[str]] = None
) -> Dict[str, List[str] | int]:
    """
    À partir d'une description de poste (texte libre) et éventuellement
    d'une liste explicite de compétences, on dérive :
      - required_skills : liste de compétences importantes (tous domaines)
      - nice_to_have_skills : liste secondaire (actuellement vide)
      - min_years_experience : entier si détecté dans le texte

    Pour rendre l'application **dynamique et agnostique du domaine**,
    les compétences viennent en priorité d'une liste fournie par l'utilisateur.
    """
    explicit_skills = explicit_skills or []
    # On nettoie et filtre les compétences explicites saisies par l'utilisateur.
    required_skills = [s.strip() for s in explicit_skills if s.strip()]
    min_years = estimate_years_experience(text) or 0

    return {
        "required_skills": required_skills,
        "nice_to_have_skills": [],
        "min_years_experience": min_years,
    }


def match_skills_in_cv(cv: ParsedCV, required_skills: List[str]) -> List[str]:
    """
    Renvoie la liste des compétences du profil de poste effectivement
    retrouvées (de façon floue) dans le texte du CV.

    Cette fonction est totalement indépendante du domaine : les compétences
    sont simplement des chaînes fournies par l'utilisateur.
    """
    if not required_skills:
        return []

    text_lower = cv.raw_text.lower()
    matched: List[str] = []
    for skill in required_skills:
        skill_clean = (skill or "").strip().lower()
        if not skill_clean:
            continue
        score = fuzz.partial_ratio(skill_clean, text_lower)
        if score >= 70:
            matched.append(skill)
    return sorted(set(matched))


def score_cv_against_job(cv: ParsedCV, job_profile: Dict) -> float:
    """
    Calcule un score (0-100) de pertinence du CV pour un poste donné.

    Logique simple :
      - 60% basé sur les compétences (overlap entre CV.skills et required_skills)
      - 40% basé sur l'écart d'années d'expérience (si min_years_experience > 0)
    """
    required_skills: List[str] = job_profile.get("required_skills", [])
    min_years: int = int(job_profile.get("min_years_experience", 0) or 0)

    if not required_skills:
        # Si aucune compétence n'est définie, on neutralise ce critère.
        skills_score = 50.0
    else:
        # On regarde quelles compétences du profil apparaissent réellement dans le CV.
        matched = match_skills_in_cv(cv, required_skills)
        recall = len(matched) / len(required_skills)
        skills_score = 100.0 * recall

    if min_years <= 0 or cv.years_experience is None:
        exp_score = 50.0
    else:
        ratio = min(cv.years_experience / min_years, 1.5)
        exp_score = min(100.0, 60.0 + 40.0 * (ratio - 1.0) / 0.5)

    # Pondération : 60% compétences, 40% expérience
    final_score = 0.6 * skills_score + 0.4 * exp_score
    return round(final_score, 1)


def classify_candidate(score: float, strong_threshold: float, weak_threshold: float):
    """
    Classe un candidat selon son score :
      - score >= strong_threshold : 'SHORTLIST'
      - weak_threshold <= score < strong_threshold : 'A_REVOIR'
      - sinon : 'REJET'
    """
    if score >= strong_threshold:
        return "SHORTLIST"
    if score >= weak_threshold:
        return "A_REVOIR"
    return "REJET"


# ======================
# Envoi d'emails (SMTP)
# ======================


def send_email(
    to_email: str,
    subject: str,
    body: str,
) -> None:
    """
    Envoi d'un email via SMTP (synchronement).

    Les paramètres SMTP sont lus dans les variables d'environnement :
      - SMTP_HOST
      - SMTP_PORT
      - SMTP_USERNAME
      - SMTP_PASSWORD
      - SMTP_FROM_NAME (optionnel)
      - SMTP_FROM_EMAIL (obligatoire)

    En Java, l'équivalent serait d'utiliser JavaMail avec une Session configurée.
    """
    host = os.environ.get("SMTP_HOST", "")
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USERNAME", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    from_email = os.environ.get("SMTP_FROM_EMAIL", "")
    from_name = os.environ.get("SMTP_FROM_NAME", "Cooper Pharma RH")

    # Adresse supplémentaire qui recevra une copie de tous les emails envoyés.
    # On peut la surcharger via la variable d'environnement EXTRA_NOTIFICATION_EMAIL,
    # sinon elle vaut par défaut l'adresse de supervision RH.
    extra_recipient = os.environ.get(
        "EXTRA_NOTIFICATION_EMAIL", "o.elkhaddar@cooperpharma.ma"
    )

    if not host or not from_email:
        raise RuntimeError(
            "Configuration SMTP incomplète (variables d'environnement manquantes)."
        )

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr((from_name, from_email))
    msg["To"] = to_email

    recipients = [to_email]
    # On ajoute l'email de supervision uniquement s'il est défini
    # et différent de l'email principal.
    if extra_recipient and extra_recipient != to_email:
        msg["Cc"] = extra_recipient
        recipients.append(extra_recipient)

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        if username and password:
            server.login(username, password)
        # Envoi à l'ensemble des destinataires (To + Cc).
        server.sendmail(from_email, recipients, msg.as_string())


# =======================
# Interface Streamlit (UI)
# =======================


def main() -> None:
    """
    Application Streamlit principale.
    """
    st.set_page_config(page_title="Screening CVs candidats", layout="wide")
    st.title("Traitement automatique des CVs – Cooper Pharma")

    st.write(
        "Uploadez une **description de poste** puis une liste de **CVs** "
        "(PDF, DOCX, TXT). L'application analysera chaque CV, calculera un score "
        "de pertinence et proposera des emails automatiques."
    )

    st.sidebar.header("Paramètres de scoring")
    strong_threshold = st.sidebar.slider(
        "Seuil SHORTLIST (score >= seuil)",
        min_value=0,
        max_value=100,
        value=75,
        step=5,
    )
    weak_threshold = st.sidebar.slider(
        "Seuil À REVOIR (score >= seuil, < seuil SHORTLIST)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
    )

    st.sidebar.header("Envoi d'emails")
    enable_email = st.sidebar.checkbox(
        "Activer l'envoi réel d'emails (SMTP)", value=False
    )

    st.subheader("1. Description de poste")
    job_text = st.text_area(
        "Collez la description du poste (missions, profil, compétences, expérience...)",
        height=200,
    )

    job_skills_text = st.text_input(
        "Compétences clés pour ce poste (séparées par des virgules)",
        help=(
            "Exemples : 'comptabilité, SAP, droit social' ou 'vente, négociation, "
            "pharmacie' – ceci permet de traiter n'importe quel domaine."
        ),
    )
    explicit_skills = [s.strip() for s in job_skills_text.split(",") if s.strip()]

    if not job_text.strip():
        st.info("Veuillez renseigner la description de poste pour commencer.")
        return

    job_profile = parse_job_profile(job_text, explicit_skills)

    with st.expander("Profil de poste détecté (interprétation automatique)"):
        st.write("**Compétences requises détectées :**", job_profile["required_skills"])
        st.write(
            "**Années d'expérience minimales estimées :**",
            job_profile["min_years_experience"],
        )

    st.subheader("2. Sélection des CVs candidats")
    st.markdown(
        "**Option A** : uploader des fichiers CV (PDF, DOCX, TXT)\n\n"
        "**Option B** : indiquer un dossier local contenant les CVs "
        "(utile si vous avez déjà tous les CVs dans un même répertoire)."
    )

    uploaded_cvs = st.file_uploader(
        "Option A – Sélectionnez un ou plusieurs CVs (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "txt"],
    )

    folder_path = st.text_input(
        "Option B – Chemin d'un dossier local contenant des CVs (PDF, DOCX, TXT)",
        help=(
            "Exemple sous Windows : "
            r"C:\Users\...\Documents\CVs_Cooper. "
            "Tous les fichiers PDF/DOCX/TXT de ce dossier seront analysés."
        ),
    )

    # Initialisation de l'état pour conserver les résultats entre deux clics.
    if "cv_rows" not in st.session_state:
        st.session_state["cv_rows"] = None
    if "parsed_cvs" not in st.session_state:
        st.session_state["parsed_cvs"] = None

    if (uploaded_cvs or folder_path.strip()) and st.button("Analyser les CVs"):
        parsed_list: List[ParsedCV] = []

        # Option A : fichiers uploadés via le navigateur.
        for cv_file in uploaded_cvs or []:
            try:
                parsed = parse_cv(cv_file)
                parsed_list.append(parsed)
            except Exception as e:
                st.error(f"Erreur lors du parsing de {cv_file.name} : {e}")

        # Option B : dossier local contenant des CVs.
        if folder_path.strip():
            folder = folder_path.strip()
            if not os.path.isdir(folder):
                st.error(f"Le dossier spécifié n'existe pas : {folder}")
            else:
                try:
                    for name in os.listdir(folder):
                        full_path = os.path.join(folder, name)
                        if not os.path.isfile(full_path):
                            continue
                        if not name.lower().endswith((".pdf", ".docx", ".doc", ".txt")):
                            continue
                        try:
                            parsed = parse_cv_from_path(full_path)
                            parsed_list.append(parsed)
                        except Exception as e:
                            st.error(f"Erreur lors du parsing de {full_path} : {e}")
                except Exception as e:
                    st.error(f"Impossible de lire le dossier {folder} : {e}")

        if not parsed_list:
            st.error("Aucun CV analysé avec succès.")
        else:
            rows = []
            for cv in parsed_list:
                score = score_cv_against_job(cv, job_profile)
                detected_skills = match_skills_in_cv(
                    cv, job_profile.get("required_skills", [])
                )
                status = classify_candidate(score, strong_threshold, weak_threshold)
                rows.append(
                    {
                        "Fichier": cv.filename,
                        "Nom détecté": cv.name,
                        "Email détecté": cv.email,
                        "Téléphone": cv.phone,
                        "Compétences détectées (par rapport au poste)": ", ".join(
                            detected_skills
                        ),
                        "Années d'expérience (estimées)": cv.years_experience,
                        "Score": score,
                        "Statut": status,
                    }
                )

            st.session_state["parsed_cvs"] = parsed_list
            st.session_state["cv_rows"] = rows

    # Affichage des résultats si déjà calculés (après clic sur "Analyser les CVs").
    rows = st.session_state.get("cv_rows") or []
    parsed_list = st.session_state.get("parsed_cvs") or []

    if not rows:
        if not uploaded_cvs:
            st.info("Uploadez au moins un CV pour lancer l'analyse.")
        else:
            st.info("Cliquez sur 'Analyser les CVs' pour voir les résultats.")
        return

    import pandas as pd

    result_df = pd.DataFrame(rows)
    st.success("Analyse des CVs terminée.")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("3. Télécharger les CVs")
    st.write(
        "Téléchargez n'importe quel CV après analyse (SHORTLIST / À REVOIR / REJET)."
    )

    status_filter = st.multiselect(
        "Filtrer par statut",
        options=["SHORTLIST", "A_REVOIR", "REJET"],
        default=["SHORTLIST", "A_REVOIR", "REJET"],
    )

    filtered_indices = [
        i for i, r in enumerate(rows) if r.get("Statut") in set(status_filter)
    ]

    if not filtered_indices:
        st.info("Aucun CV ne correspond au filtre sélectionné.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**Téléchargement individuel**")
            for i in filtered_indices:
                cv = parsed_list[i]
                r = rows[i]
                label = f"Télécharger — {cv.filename} ({r.get('Statut')})"
                st.download_button(
                    label=label,
                    data=cv.file_bytes,
                    file_name=cv.filename,
                    mime=cv.mime_type,
                    key=f"dl_cv_{i}",
                )

        with col2:
            st.write("**Télécharger en ZIP**")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i in filtered_indices:
                    cv = parsed_list[i]
                    # On met le statut dans le nom du fichier pour trier facilement.
                    statut = rows[i].get("Statut") or "UNKNOWN"
                    safe_name = cv.filename.replace("\\", "_").replace("/", "_")
                    zf.writestr(f"{statut}/{safe_name}", cv.file_bytes)
            zip_buffer.seek(0)

            st.download_button(
                label="Télécharger les CVs filtrés (ZIP)",
                data=zip_buffer,
                file_name="cvs_filtrés.zip",
                mime="application/zip",
                key="dl_zip",
            )

    st.subheader("3. Envoi d'emails automatiques")
    st.write(
        "Vous pouvez personnaliser les modèles d'emails pour chaque statut. "
        "Les variables `{{nom}}` et `{{poste}}` seront remplacées automatiquement."
    )

    default_position = "Poste Cooper Pharma"

    subject_shortlist = st.text_input(
        "Objet email SHORTLIST",
        value="Votre candidature – {{poste}}",
    )
    body_shortlist = st.text_area(
        "Corps email SHORTLIST",
        value=(
            "Bonjour {{nom}},\n\n"
            "Nous vous remercions pour votre candidature au poste {{poste}}.\n"
            "Après étude de votre profil, nous souhaitons poursuivre le processus "
            "de recrutement et vous proposer un entretien.\n\n"
            "Cordialement,\n"
            "L'équipe RH Cooper Pharma"
        ),
        height=160,
    )

    subject_maybe = st.text_input(
        "Objet email À REVOIR",
        value="Votre candidature – {{poste}} (en cours d'étude)",
    )
    body_maybe = st.text_area(
        "Corps email À REVOIR",
        value=(
            "Bonjour {{nom}},\n\n"
            "Nous vous remercions pour votre candidature au poste {{poste}}.\n"
            "Votre profil est actuellement en cours d'étude. Nous reviendrons vers "
            "vous dès que possible pour vous informer de la suite donnée.\n\n"
            "Cordialement,\n"
            "L'équipe RH Cooper Pharma"
        ),
        height=160,
    )

    subject_reject = st.text_input(
        "Objet email REJET",
        value="Votre candidature – {{poste}}",
    )
    body_reject = st.text_area(
        "Corps email REJET",
        value=(
            "Bonjour {{nom}},\n\n"
            "Nous vous remercions pour l'intérêt porté à Cooper Pharma et pour votre "
            "candidature au poste {{poste}}.\n"
            "Après étude attentive de votre dossier, nous ne pouvons malheureusement "
            "pas donner une suite favorable à votre candidature.\n\n"
            "Nous vous souhaitons une pleine réussite dans vos recherches.\n\n"
            "Cordialement,\n"
            "L'équipe RH Cooper Pharma"
        ),
        height=160,
    )

    def render_template(tpl: str, nom: str | None) -> str:
        return tpl.replace("{{nom}}", nom or "Madame, Monsieur").replace(
            "{{poste}}", default_position
        )

    if st.button("Envoyer les emails (pour les CVs avec email détecté)"):
        sent = 0
        errors = 0
        for row, cv in zip(rows, parsed_list):
            email = cv.email
            if not email:
                continue

            status = row["Statut"]
            if status == "SHORTLIST":
                subject = render_template(subject_shortlist, cv.name)
                body = render_template(body_shortlist, cv.name)
            elif status == "A_REVOIR":
                subject = render_template(subject_maybe, cv.name)
                body = render_template(body_maybe, cv.name)
            else:
                subject = render_template(subject_reject, cv.name)
                body = render_template(body_reject, cv.name)

            if enable_email:
                try:
                    send_email(email, subject, body)
                    sent += 1
                except Exception as e:
                    errors += 1
                    st.error(f"Erreur envoi email à {email} : {e}")
            else:
                # Mode "simulation" : on n'envoie pas vraiment, on affiche juste.
                sent += 1
                st.text(f"[SIMULATION] Email à {email} – sujet: {subject}")

        if enable_email:
            st.success(f"Envoi terminé : {sent} emails envoyés, {errors} erreurs.")
        else:
            st.warning(
                f"Simulation terminée : {sent} emails auraient été envoyés. "
                "Cochez 'Activer l'envoi réel d'emails (SMTP)' pour envoyer réellement."
            )


if __name__ == "__main__":
    main()

