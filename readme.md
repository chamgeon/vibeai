# VibeAI

VibeAI is a web application that generates music playlists based on the **vibe of a user-uploaded photo**.  
It uses AI to analyze mood, suggest songs, and export personalized playlists to **YouTube** — with full editing support.


## Features

- 📷 **Upload a Photo** — Send in a photo that reflects your mood or moment.
- 🧠 **Vibe Extraction** — Uses OpenAI to interpret the aesthetic, emotion, or context of the image.
- 🎶 **Playlist Generation** — Creates a custom music playlist that matches the vibe.
- 🛠️ **Edit Your Playlist** — Add/remove tracks before finalizing.
- 📤 **Export to YouTube** — Save your playlist directly to your YouTube account.


## Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **APIs Used:**
  - [OpenAI API](https://platform.openai.com/) — for image-to-text and mood analysis
  - [Spotify API](https://developer.spotify.com/) — for music metadata and recommendations
  - [YouTube Data API](https://developers.google.com/youtube/v3) — to create and export playlists
