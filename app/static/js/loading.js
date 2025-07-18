document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("playlist-form");
  const loader = document.getElementById("loading-overlay");

  form.addEventListener("submit", function () {
    loader.style.display = "flex";  // Show loading screen
  });
});