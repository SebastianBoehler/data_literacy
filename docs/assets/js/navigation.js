import { qs, qsa } from "./dom.js";

export function setupNavigation() {
  const navLinks = qsa(".section-nav a");
  const sections = navLinks
    .map((link) => {
      const target = link.getAttribute("href");
      return target ? qs(target) : null;
    })
    .filter(Boolean);

  function setActiveLink(activeLink) {
    navLinks.forEach((navLink) => {
      const isActive = navLink === activeLink;
      navLink.classList.toggle("active", isActive);
      if (isActive) {
        navLink.setAttribute("aria-current", "true");
      } else {
        navLink.removeAttribute("aria-current");
      }
    });
  }

  function syncActiveSection() {
    const offset = 148;
    let activeIndex = 0;

    sections.forEach((section, index) => {
      if (section.getBoundingClientRect().top <= offset) {
        activeIndex = index;
      }
    });

    if (navLinks[activeIndex]) {
      setActiveLink(navLinks[activeIndex]);
    }
  }

  if (navLinks[0]) {
    setActiveLink(navLinks[0]);
  }

  let ticking = false;
  const onScroll = () => {
    if (ticking) {
      return;
    }

    ticking = true;
    window.requestAnimationFrame(() => {
      syncActiveSection();
      ticking = false;
    });
  };

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", syncActiveSection);
  syncActiveSection();
}
