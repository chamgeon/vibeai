document.addEventListener('DOMContentLoaded', function () {

  const playlist = document.getElementById("playlist");

  Sortable.create(playlist, {
    handle: ".drag-handle", // Only allow dragging by handle
    animation: 150,          // Smooth animation
    onEnd: function () {
      // Update the visible index numbers
      document.querySelectorAll("#playlist li").forEach((li, idx) => {
        const indexDiv = li.querySelector(".index");
        if (indexDiv) {
          indexDiv.textContent = idx + 1;
        }
      });
    }
  });

  // Handle remove buttons
  document.querySelectorAll('.remove-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const index = btn.getAttribute('data-index');
      const item = document.querySelector(`#playlist li[data-index='${index}']`);
      if (item) item.remove();
    });
  });

  // Update playlsit and confirm
  window.saveAndConfirm = function () {
    const listItems = document.querySelectorAll("#playlist li");

    const updatedTracks = Array.from(listItems).map(li => ({
      song: li.getAttribute("data-song"),
      artist: li.getAttribute("data-artist"),
      uri: li.getAttribute("data-uri"),
      cover_url: li.getAttribute("data-cover-url"),
      vibe: li.getAttribute("data-vibe")
    }));

    // Show loading
    document.getElementById("loading-overlay").style.display = "flex";

    // ✅ Open the new tab immediately
    const win = window.open("", "_blank");

    fetch('/finalize_playlist', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tracks: updatedTracks })
    })
    .then(res => res.json())
    .then(data => {
      if (data.playlist_url) {
        win.location.href = data.playlist_url; // ✅ Redirect in the already-opened window
      } else {
        alert("Failed to create playlist.");
        win.close();
      }
      document.getElementById("loading-overlay").style.display = "none";
    })
    .catch(err => {
      console.error(err);
      alert("Error finalizing playlist.");
      win.close();
      document.getElementById("loading-overlay").style.display = "none";
    });
  };
});
