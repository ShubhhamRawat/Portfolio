const header = document.querySelector("[data-header]");
const nav = document.querySelector("[data-nav]");
const navToggle = document.querySelector("[data-nav-toggle]");
const navBackdrop = document.querySelector("[data-nav-backdrop]");
const year = document.querySelector("[data-year]");

const updateHeader = () => {
  header.classList.toggle("is-scrolled", window.scrollY > 10);
};

year.textContent = new Date().getFullYear();
updateHeader();

window.addEventListener("scroll", updateHeader, { passive: true });

const setNavOpen = (isOpen) => {
  nav.classList.toggle("is-open", isOpen);
  navToggle.classList.toggle("is-open", isOpen);
  navBackdrop.classList.toggle("is-open", isOpen);
  document.body.classList.toggle("nav-open", isOpen);
  navToggle.setAttribute("aria-expanded", String(isOpen));
  navToggle.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
};

navToggle.addEventListener("click", () => {
  setNavOpen(!nav.classList.contains("is-open"));
});

nav.addEventListener("click", (event) => {
  if (nav.classList.contains("is-open") && !event.target.closest("a")) {
    event.stopPropagation();
  }
});

nav.addEventListener("click", (event) => {
  const link = event.target.closest("a");

  if (link) {
    const targetId = link.getAttribute("href");

    if (targetId && targetId.startsWith("#")) {
      event.preventDefault();
      setNavOpen(false);

      window.setTimeout(() => {
        document.querySelector(targetId)?.scrollIntoView({ behavior: "smooth" });
      }, 220);

      return;
    }

    setNavOpen(false);
  }
});

navBackdrop.addEventListener("click", () => {
  setNavOpen(false);
});

document.addEventListener(
  "click",
  (event) => {
    if (!nav.classList.contains("is-open")) {
      return;
    }

    if (nav.contains(event.target) || navToggle.contains(event.target)) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    setNavOpen(false);
  },
  true
);

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && nav.classList.contains("is-open")) {
    setNavOpen(false);
  }
});
