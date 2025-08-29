# Music RAG Dataset

This folder holds **local, human-readable data** for a music Retrieval-Augmented Generation (RAG) corpus.
It is organized by **song–artist** so you can easily browse and extend sources (YouTube, Reddit, Genius, etc.).

> **Tracked in Git:** this README and `manifest.json`  
> **Ignored by Git:** `songs/` raw data and digestions (see repo `.gitignore`)

---

## Layout

music-rag/
├─ README.md
├─ manifest.json # (optional) dataset-level info
└─ songs/
    └─ <Artist> - <Song>/
    ├─ meta.json
    ├─ sources/
    │ └─ youtube_<videoId>.comments.jsonl
    └─ digestion/
        ├─ summary.md
        └─ summary.json

- **sources/** → raw data pulled from YouTube (later Reddit, Genius, etc.)  
- **digestion/** → “music vibe digestion” (GPT summary of comments) in markdown + JSON  
- **meta.json** → basic info about the song and source IDs

---

## Notes

- Raw comments are saved as `.jsonl` (one comment per line).  
- Digestion is saved in both `.md` (human-readable) and `.json` (structured metadata).  
- `songs/` can grow large, so it is gitignored. Only this README and schema/manifest files are tracked in Git.  
- Typical dataset size for ~1000 songs is only a few MBs, so local storage is fine.  