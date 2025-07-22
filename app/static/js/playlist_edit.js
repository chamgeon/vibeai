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


  // Update playlsit and login to spotify
  window.saveAndRedirectToLogin = function () {
    const listItems = document.querySelectorAll("#playlist li");

    const updatedTracks = Array.from(listItems).map(li => ({
      song: li.getAttribute("data-song"),
      artist: li.getAttribute("data-artist"),
      uri: li.getAttribute("data-uri"),
      cover_url: li.getAttribute("data-cover-url"),
      vibe: li.getAttribute("data-vibe")
    }));

    fetch('/save-tracks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tracks: updatedTracks })
    })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        window.location.href = "/spotify-login";
      } else {
        alert("Failed to create playlist.");
      }
    })
    .catch(err => {
      console.error(err);
      alert(err.message || "Error finalizing playlist.");
    });
  };
});
