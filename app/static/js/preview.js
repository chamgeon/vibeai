document.addEventListener("DOMContentLoaded", function () {
  const dropZone = document.getElementById('drop-zone');
  const dropText = document.getElementById('drop-text');
  const imageInput = document.getElementById('image-input');
  const preview = document.getElementById('image-preview');

  function previewFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      preview.src = reader.result;
      dropZone.classList.add('previewing');
    };
    reader.readAsDataURL(file);
  }

  dropZone.addEventListener('click', () => imageInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      imageInput.files = e.dataTransfer.files;
      previewFile(file);
    }
  });

  imageInput.addEventListener('change', function () {
    const file = this.files[0];
    if (file) previewFile(file);
  });
});