# GitHub + Streamlit Cloud Deployment Guide

## STEP 1 — Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `sri-lanka-weather-ensemble`
3. Set to **Public** (required for Streamlit Cloud free tier)
4. Do NOT initialise with README (you already have one)
5. Click **Create repository**

---

## STEP 2 — Push Project to GitHub

Open PowerShell inside `weather_project/`:

```powershell
git init
git add .
git commit -m "Initial commit — Ensemble Learning Assignment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sri-lanka-weather-ensemble.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

> **Note:** The `.pkl` model files are large (~30MB total).
> If GitHub rejects them, install Git LFS:
> ```powershell
> git lfs install
> git lfs track "models/*.pkl"
> git add .gitattributes
> git commit -m "Track model files with LFS"
> git push
> ```

---

## STEP 3 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **New app**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/sri-lanka-weather-ensemble`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**

Deployment takes 3–5 minutes.
Your live URL will be: `https://YOUR_USERNAME-sri-lanka-weather-ensemble-app-XXXXX.streamlit.app`

---

## STEP 4 — Add URL to README and Notebook

In `README.md`, add at the top:
```markdown
🌐 **Live Dashboard:** https://YOUR_STREAMLIT_URL.streamlit.app
📁 **GitHub:** https://github.com/YOUR_USERNAME/sri-lanka-weather-ensemble
```

In the notebook, add a markdown cell at the top:
```markdown
**Live Dashboard:** https://YOUR_STREAMLIT_URL.streamlit.app
**GitHub:** https://github.com/YOUR_USERNAME/sri-lanka-weather-ensemble
```

---

## STEP 5 — If Models Are Too Large for GitHub

Streamlit Cloud needs the `.pkl` files to load models.
If they exceed GitHub's 100MB file limit, use this workaround:

**Option A — Train on startup (slow but works):**
Add this to the top of `app.py`:
```python
import os
if not os.path.exists("models/catboost.pkl"):
    os.system("python src/train.py")
```

**Option B — Host models on Google Drive and download at startup.**

---

## STEP 6 — Submission Checklist

Before Monday midnight:
- [ ] GitHub repo is public and accessible
- [ ] `app.py`, `src/`, `models/`, `notebooks/`, `requirements.txt` all pushed
- [ ] Streamlit Cloud URL is live and loads without error
- [ ] Both URLs added to README.md and notebook
- [ ] Video recorded and under 3 minutes
- [ ] All files submitted to the university portal
