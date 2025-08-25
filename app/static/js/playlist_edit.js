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


  // Update playlsit and login to platform
  window.saveAndRedirectToLogin = async function (platform) {
    const btn = document.querySelector(".save-btn");
    try {
      // 1) Check quota first
      btn?.setAttribute("disabled", "true");

      const r = await fetch("/quota-status", { cache: "no-store" });
      if (!r.ok) throw new Error("Failed to check quota");
      const { ratio } = await r.json();

      if (ratio != null && ratio >= 0.9) {
        alert("We're near today's YouTube quota limit. Please try again after the daily reset (midnight PT).");
        return; // stop here
      }

      // 2) Proceed with your original logic
      const listItems = document.querySelectorAll("#playlist li");
      const updatedTracks = Array.from(listItems).map(li => ({
        song: li.getAttribute("data-song"),
        artist: li.getAttribute("data-artist"),
        uri: li.getAttribute("data-uri"),
        cover_url: li.getAttribute("data-cover-url"),
        vibe: li.getAttribute("data-vibe"),
      }));

      document.getElementById("loading-overlay").style.display = "flex";

      const res = await fetch("/save-tracks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tracks: updatedTracks }),
      });

      const data = await res.json();
      if (data.success && data.pl_id) {
        window.location.href = `/${platform}-login?pl_id=${encodeURIComponent(data.pl_id)}`;
      } else {
        alert("Failed to create playlist.");
      }
    } catch (err) {
      console.error(err);
      alert(err.message || "Error finalizing playlist.");
    } finally {
      btn?.removeAttribute("disabled");
      // If you showed the overlay but exited early due to quota, make sure it's hidden:
      const overlay = document.getElementById("loading-overlay");
      if (overlay) overlay.style.display = "none";
    }
  };
});
