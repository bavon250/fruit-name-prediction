# Fruit Detection App

A Streamlit web app for detecting fruit in uploaded images or webcam pictures using a trained RF-DETR checkpoint.

## Project files

- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes large model weights and private files.
- `.streamlit/config.toml` — allows model uploads up to 250 MB.
- `weights.pt` — your trained model file (keep this locally; do not commit it to GitHub).

## 1. Run locally in VS Code

Install Python 3.10 or newer. Open this folder in VS Code, then run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows CMD**
```bat
.venv\Scripts\activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## 2. Load your model

In the app's left sidebar, upload your `weights.pt` file. Alternatively, for local use only, place `weights.pt` in the same folder as `app.py`; the app will find it automatically.

The uploaded checkpoint is loaded as an RF-DETR Small model. Your checkpoint's architecture must match that model class. If the app reports an architecture or checkpoint error, the model class used during training must be confirmed and the loader adjusted.

If class labels are not included in the checkpoint, enter the class names in the sidebar in class-ID order, separated by commas.

## 3. Upload the source code to GitHub

Create a new empty GitHub repository. In the terminal, from this project folder, run:

```bash
git init
git add .
git commit -m "Add fruit detection Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the GitHub URL with your own repository URL.

**Do not force-add `weights.pt`.** GitHub blocks ordinary files larger than 100 MB. Keep the weights private and upload them through the app when it runs, or store them in an appropriate external model-storage service.

## 4. Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at https://share.streamlit.io/.
3. Select **Create app** and choose your repository.
4. Set the main file path to `app.py`.
5. Deploy.
6. When the app opens, upload your `weights.pt` file using the sidebar.

Large model downloads and CPU memory use may exceed the limits of a free hosting instance. If deployment runs out of memory or times out, use a machine with more RAM or a GPU.

## Notes

- The app does not need your Roboflow API key when you upload the trained checkpoint directly.
- Never commit API keys, passwords, or private tokens.
- RF-DETR inference API reference: https://rfdetr.roboflow.com/latest/reference/rfdetr/
