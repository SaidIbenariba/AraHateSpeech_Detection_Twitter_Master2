document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const btn = document.getElementById("btn");
  const loading = document.getElementById("loading");
  const textarea = form.querySelector("textarea[name='tweet']");

  // ✅ تأكيد الإخفاء عند فتح الصفحة
  loading.classList.add("hidden");

  form.addEventListener("submit", (e) => {
    const value = (textarea.value || "").trim();

    // ✅ منع الإرسال إذا فارغ
    if (!value) {
      e.preventDefault();
      alert("المرجو كتابة نص قبل الضغط على تحليل.");
      return;
    }

    // ✅ إظهار spinner عند الضغط على تحليل
    btn.disabled = true;
    loading.classList.remove("hidden");
  });
});
