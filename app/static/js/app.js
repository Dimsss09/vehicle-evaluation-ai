const form = document.querySelector("#predict-form");

if (form) {
  form.addEventListener("submit", () => {
    const button = form.querySelector("button[type='submit']");
    button.textContent = "Memproses...";
    button.disabled = true;
  });
}
