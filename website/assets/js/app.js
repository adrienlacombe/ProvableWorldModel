// ProvableWorldModel explainer interactivity: mobile nav, scroll reveals, the
// animated end-to-end flow, the live Freivalds challenge game, and the tier tabs.
// The Freivalds widget does real integer arithmetic, so the equality it shows is
// genuine: v.x equals r.z exactly when z = W.x, and breaks the moment z is forged.

(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // --- mobile nav ---
  const toggle = $("#navToggle");
  const links = $(".nav-links");
  if (toggle && links) {
    const setNav = (open) => {
      links.classList.toggle("open", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    };
    toggle.addEventListener("click", () => setNav(!links.classList.contains("open")));
    $$(".nav-links a").forEach((a) => a.addEventListener("click", () => setNav(false)));
  }
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // --- reveal on scroll (interactive boards and card grids only; headings and
  // hero content are visible immediately, never gated on a scroll transition) ---
  const revealTargets = $$(".card, .flow-step, .crate, .start-card, .tiers, .compare, .frei");
  revealTargets.forEach((el) => el.classList.add("reveal"));
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      }),
      { threshold: 0.12 },
    );
    revealTargets.forEach((el) => io.observe(el));
  } else {
    revealTargets.forEach((el) => el.classList.add("in"));
  }

  // --- animated end-to-end flow ---
  const flow = $("#flow");
  if (flow) {
    const steps = $$(".flow-step", flow);
    const verdict = $(".flow-verdict", flow);
    const vTitle = $("#verdictTitle");
    const vText = $("#verdictText");
    const tamper = $("#tamperToggle");
    let timer = null;

    const clear = () => {
      if (timer) { clearInterval(timer); timer = null; }
      steps.forEach((s) => s.classList.remove("is-active", "is-done"));
      verdict.classList.remove("reject");
    };
    const setVerdict = () => {
      const forged = tamper && tamper.checked;
      verdict.classList.toggle("reject", forged);
      if (forged) {
        vTitle.textContent = "Reject";
        vText.textContent = "One matmul output was forged, so v.x != r.z. The verifier returns FreivaldsCheckFailed.";
      } else {
        vTitle.textContent = "Accept";
        vText.textContent = "All checks pass. The trajectory, costs, and selected action are exactly the committed quantized inference.";
      }
    };
    const play = () => {
      clear();
      setVerdict();
      if (reduceMotion) {
        steps.forEach((s) => s.classList.add("is-done"));
        steps[steps.length - 1].classList.add("is-active");
        return;
      }
      let i = 0;
      timer = setInterval(() => {
        if (i > 0) steps[i - 1].classList.add("is-done");
        if (i >= steps.length) { clearInterval(timer); timer = null; return; }
        steps[i].classList.remove("is-done");
        steps[i].classList.add("is-active");
        i += 1;
      }, 650);
    };
    $("#flowPlay")?.addEventListener("click", play);
    $("#flowReset")?.addEventListener("click", clear);
    tamper?.addEventListener("change", setVerdict);
    setVerdict();
  }

  // --- live Freivalds challenge game (real integer math) ---
  const board = $("#freiBoard");
  if (board) {
    const W = [[2, -1], [1, 3], [0, 2]]; // 3 x 2 fixed weights
    const x = [3, 4];
    const rows = W.length, cols = x.length;
    const honestZ = W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0)); // [2,15,8]
    let z = honestZ.slice();
    let r = null;
    let tampered = false;

    const cell = (v, cls = "") => {
      const d = document.createElement("span");
      d.className = "cell " + cls;
      d.textContent = String(v);
      return d;
    };
    const fillMatrix = (el, grid) => {
      $$(".cell", el).forEach((c) => c.remove());
      el.style.gridTemplateColumns = `repeat(${grid[0].length}, auto)`;
      grid.forEach((row) => row.forEach((v) => el.appendChild(cell(v))));
    };
    const fillVector = (el, vec, tamperedIdx = -1) => {
      $$(".cell", el).forEach((c) => c.remove());
      el.style.gridTemplateColumns = "auto";
      vec.forEach((v, i) => el.appendChild(cell(v, i === tamperedIdx ? "tampered" : "")));
    };

    const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
    const rTW = (rv) => W[0].map((_, j) => rv.reduce((s, _w, i) => s + rv[i] * W[i][j], 0));

    const matW = $("#matW"), vecX = $("#vecX"), vecZ = $("#vecZ"), vecR = $("#vecR");
    const vx = $("#vxVal"), rz = $("#rzVal"), eq = $("#freiEq"), verdict = $("#freiVerdict");

    const renderStatic = () => {
      fillMatrix(matW, W);
      fillVector(vecX, x);
      fillVector(vecZ, z, tampered ? 0 : -1);
    };
    const renderCheck = () => {
      if (!r) {
        vx.textContent = "?"; rz.textContent = "?";
        eq.textContent = "=?="; eq.className = "frei-eq";
        verdict.textContent = "Draw a challenge to start.";
        verdict.className = "frei-verdict";
        return;
      }
      const v = rTW(r);
      const lhs = dot(v, x);
      const rhs = dot(r, z);
      vx.textContent = String(lhs);
      rz.textContent = String(rhs);
      const ok = lhs === rhs;
      eq.textContent = ok ? "==" : "!=";
      eq.className = "frei-eq " + (ok ? "ok" : "no");
      verdict.className = "frei-verdict " + (ok ? "ok" : "no");
      verdict.textContent = ok
        ? "ACCEPT. v.x == r.z, so the committed z really is W.x."
        : "REJECT. v.x != r.z, the forged accumulator is caught. FreivaldsCheckFailed.";
    };

    const draw = () => {
      r = Array.from({ length: rows }, () => 1 + Math.floor(Math.random() * 9));
      fillVector(vecR, r);
      $$(".cell", vecR).forEach((c) => { c.classList.add("flash"); setTimeout(() => c.classList.remove("flash"), 350); });
      renderCheck();
    };
    const doTamper = () => {
      tampered = true;
      z = honestZ.slice();
      z[0] += 1;
      renderStatic();
      renderCheck();
    };
    const reset = () => {
      tampered = false;
      z = honestZ.slice();
      renderStatic();
      renderCheck();
    };

    $("#freiDraw")?.addEventListener("click", draw);
    $("#freiTamper")?.addEventListener("click", doTamper);
    $("#freiReset")?.addEventListener("click", reset);
    renderStatic();
    fillVector(vecR, ["?", "?", "?"]);
    renderCheck();
  }

  // --- tier tabs (ARIA tabs pattern: click + arrow keys, roving tabindex) ---
  const tabs = $$(".tier-tab");
  if (tabs.length) {
    const select = (tab) => {
      const tier = tab.dataset.tier;
      tabs.forEach((t) => {
        const on = t === tab;
        t.classList.toggle("is-active", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.setAttribute("tabindex", on ? "0" : "-1");
      });
      $$(".tier-panel").forEach((p) => p.classList.toggle("is-active", p.dataset.tier === tier));
    };
    tabs.forEach((tab, i) => {
      tab.setAttribute("tabindex", tab.classList.contains("is-active") ? "0" : "-1");
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", (e) => {
        const delta = { ArrowRight: 1, ArrowLeft: -1, Home: -i, End: tabs.length - 1 - i }[e.key];
        if (delta === undefined) return;
        e.preventDefault();
        const next = tabs[(i + delta + tabs.length) % tabs.length];
        select(next);
        next.focus();
      });
    });
  }
})();
