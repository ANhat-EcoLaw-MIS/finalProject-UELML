/* StyleRec: ghi sự kiện gợi ý (impression / click), thêm giỏ bằng fetch + toast, cuộn mục gợi ý.
   Không có JS trang vẫn chạy: form thêm giỏ submit bình thường, sự kiện add_to_cart do server tự ghi. */
(() => {
  "use strict";
  const ENDPOINT = "/api/events";
  const queue = [];
  const seen = new WeakSet();

  const payload = (card, event) => ({
    event,
    article_id: card.dataset.article,
    surface: card.dataset.surface || "",
    shelf: card.dataset.shelf || "",
    position: Number(card.dataset.pos) || "",
  });

  function send(events, beacon) {
    if (!events.length) return;
    const body = JSON.stringify({ events });
    if (beacon && navigator.sendBeacon) {
      navigator.sendBeacon(ENDPOINT, new Blob([body], { type: "application/json" }));
    } else {
      fetch(ENDPOINT, { method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true })
        .catch(() => {});
    }
  }
  const flush = (beacon) => send(queue.splice(0, queue.length), beacon);

  // --- impression: mỗi thẻ gợi ý một lần, khi hiện >= 50% trên màn hình; gửi gộp
  const cards = document.querySelectorAll("[data-rec]");
  if ("IntersectionObserver" in window && cards.length) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !seen.has(e.target)) {
          seen.add(e.target);
          queue.push(payload(e.target, "impression"));
          io.unobserve(e.target);
        }
      }
      if (queue.length >= 20) flush(false);
    }, { threshold: 0.5 });
    cards.forEach((c) => io.observe(c));
    setInterval(() => flush(false), 3000);
    addEventListener("pagehide", () => flush(true));
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") flush(true); });
  }

  // --- click vào thẻ gợi ý (ảnh / tên) -> ghi click bằng sendBeacon (không chặn chuyển trang)
  document.addEventListener("click", (ev) => {
    const link = ev.target.closest("[data-rec] a");
    if (!link) return;
    const card = link.closest("[data-rec]");
    flush(true);
    send([payload(card, "click")], true);
  });

  // --- toast
  const toast = document.querySelector(".toast");
  let timer;
  function showToast(msg) {
    if (!toast) return;
    toast.textContent = msg;
    toast.hidden = false;
    clearTimeout(timer);
    timer = setTimeout(() => { toast.hidden = true; }, 2600);
  }

  // --- thêm giỏ bằng fetch (server tự ghi add_to_cart kèm surface/shelf/position)
  document.addEventListener("submit", async (ev) => {
    const form = ev.target.closest("[data-add-form]");
    if (!form) return;
    ev.preventDefault();
    const btn = form.querySelector("button[type=submit]");
    if (btn) btn.disabled = true;
    try {
      const res = await fetch(form.action, {
        method: "POST", body: new FormData(form),
        headers: { "X-Requested-With": "fetch", Accept: "application/json" },
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Lỗi");
      document.querySelectorAll("[data-cart-count]").forEach((el) => { el.textContent = data.cart_count; });
      showToast(`Đã thêm ${data.name}${data.size ? " · size " + data.size : ""} vào giỏ`);
    } catch (e) {
      form.submit();   // fallback: submit thường
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  // --- nút cuộn mục gợi ý
  document.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-rail]");
    if (!b) return;
    const rail = b.closest(".shelf").querySelector(".rail");
    rail.scrollBy({ left: Number(b.dataset.rail) * rail.clientWidth * 0.9, behavior: "smooth" });
  });
})();
