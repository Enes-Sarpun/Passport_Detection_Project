// HeroLanding — Pasaport OCR-YOLO için dark dramatic giriş ekranı.
// 3 CloudFront video (blob preloading + crossfade) + CSS fallback katmanları.
// "taramaya başla" → props.onStart() → #console-section'a smooth scroll.
import { useCallback, useEffect, useRef, useState } from 'react';
import './Hero.css';

// 3 CloudFront video URL'i (orijinal prompt'tan)
const VIDEO_URLS = [
  'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260629_030107_874273ea-684a-4e90-bb96-8fdfde48d53d.mp4',
  'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260629_032424_3c9c2a9d-807b-4482-80e6-dd6d9dfd4545.mp4',
  'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260627_094019_4214ea73-b963-46a4-8327-61489192de99.mp4',
];

// 3 tema — video/css katmanlarına karşılık gelir
const SLIDES = [
  { id: 0, tag: '01', label: 'TEMA 1' },
  { id: 1, tag: '02', label: 'TEMA 2' },
  { id: 2, tag: '03', label: 'TEMA 3' },
];

// Ground-truth'tan kalibre edilmiş gerçek proje rakamları
const TRUST_POINTS = [
  { k: 'Karakter Doğruluğu', v: '%98.3' },
  { k: 'Alan Doğruluğu',     v: '%96.9' },
  { k: 'Güvenilirlik AUC',   v: '0.916' },
  { k: 'Hata Yakalama',      v: '%100'  },
];

// Yalnızca çalışan 2 nav linki
const NAV_LINKS = [
  { idx: '01', label: 'Genel Bakış', target: 'top'     },
  { idx: '02', label: 'Konsol',      target: 'console' },
];

const MRZ_TEXT = [
  'P<TURMRZ<OCR<YOLO<<<<<<<<<<<<<<<<<<<<<<<<<<<',
  'ZD000078<7TUR8501019M1801145<<<<<<<<<<<<<<04',
  'ICAO9303<DOGRULAMA<GUVEN<KONSOL<TESPIT<<<<<<',
].join('   ·   ');
const MRZ_WATERMARK = `${MRZ_TEXT}   ${MRZ_TEXT}   ${MRZ_TEXT}`;

// IntersectionObserver reveal hook'u
function useReveal(threshold = 0.15) {
  const ref = useRef(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setShown(true); io.disconnect(); } },
      { threshold }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [threshold]);
  return [ref, shown];
}

// Canlı saat: "TUR HH:MM:SS"
function Clock() {
  const [now, setNow] = useState('');
  useEffect(() => {
    const fmt = new Intl.DateTimeFormat('tr-TR', {
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    });
    const tick = () => setNow(fmt.format(new Date()));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return <span className="hero-clock mono">TUR {now}</span>;
}

// Videoları blob olarak arka planda yükle
function usePreloadedVideos(urls) {
  const [srcMap, setSrcMap] = useState({}); // index → objectURL | original URL
  useEffect(() => {
    const revoke = [];
    urls.forEach((url, i) => {
      fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error('not ok');
          return r.blob();
        })
        .then((blob) => {
          const objUrl = URL.createObjectURL(blob);
          revoke.push(objUrl);
          setSrcMap((prev) => ({ ...prev, [i]: objUrl }));
        })
        .catch(() => {
          // Erişilemezse orijinal URL'i kullan — CORS/auth sorununda CSS fallback devreye girer
          setSrcMap((prev) => ({ ...prev, [i]: url }));
        });
    });
    return () => revoke.forEach((u) => URL.revokeObjectURL(u));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return srcMap;
}

export default function HeroLanding({ onStart }) {
  const [active, setActive] = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);
  const [revealRef, revealed] = useReveal(0.15);
  const videoRefs = useRef([]);
  const srcMap = usePreloadedVideos(VIDEO_URLS);

  // Video elemanlarına src ata ve oynat
  useEffect(() => {
    videoRefs.current.forEach((el, i) => {
      if (!el || !srcMap[i]) return;
      if (el.src !== srcMap[i]) {
        el.src = srcMap[i];
        el.load();
      }
      el.play().catch(() => { /* autoplay policy — muted olduğu için çalışmalı */ });
    });
  }, [srcMap]);

  // Aktif tema değişince o videoyu baştan oynat
  useEffect(() => {
    const el = videoRefs.current[active];
    if (!el || !srcMap[active]) return;
    el.currentTime = 0;
    el.play().catch(() => {});
  }, [active, srcMap]);

  // Nav link tıklamaları
  const handleNavClick = useCallback((e, target) => {
    e.preventDefault();
    if (target === 'console') {
      const appRoot = document.querySelector('.app-root');
      const section = document.getElementById('console-section');
      if (appRoot && section) {
        appRoot.scrollTo({ top: section.offsetTop, behavior: 'smooth' });
      }
    } else {
      const appRoot = document.querySelector('.app-root');
      if (appRoot) appRoot.scrollTo({ top: 0, behavior: 'smooth' });
    }
    setMenuOpen(false);
  }, []);

  // Video bitti → sonraki temaya geç (0→1→2→0)
  const handleVideoEnd = useCallback((i) => {
    setActive((prev) => {
      // Yalnızca aktif video tetiklemeli
      if (prev !== i) return prev;
      return (prev + 1) % VIDEO_URLS.length;
    });
  }, []);

  // onStart → console section'a scroll
  const handleStart = useCallback(() => {
    const appRoot = document.querySelector('.app-root');
    const section = document.getElementById('console-section');
    if (appRoot && section) {
      appRoot.scrollTo({ top: section.offsetTop, behavior: 'smooth' });
    }
    if (onStart) onStart();
  }, [onStart]);

  return (
    <div className="hero" data-accent="green">
      {/* ── ARKA PLAN ── */}
      <div className="hero-bg" aria-hidden="true">
        {/* Video katmanları */}
        {VIDEO_URLS.map((_, i) => (
          <video
            key={i}
            ref={(el) => { videoRefs.current[i] = el; }}
            className={`hero-bg__video ${active === i ? 'is-active' : ''}`}
            muted
            autoPlay
            playsInline
            // loop kaldırıldı — video bitince onEnded tetiklensin
            onEnded={() => handleVideoEnd(i)}
            aria-hidden="true"
          />
        ))}

        {/* CSS fallback katmanları (video yüklenemezse görünür) */}
        {SLIDES.map((s) => (
          <div
            key={s.id}
            className={`hero-bg__layer hero-bg__layer--${s.id} ${
              // Yalnızca o videonun src'si henüz gelmemişse göster
              !srcMap[s.id] && active === s.id ? 'is-active' : ''
            }`}
          />
        ))}

        <div className="hero-bg__overlay" />
        <div className="hero-bg__mrz mono" aria-hidden="true">{MRZ_WATERMARK}</div>
      </div>

      {/* ── NAVBAR ── */}
      <header className="hero-nav" role="banner">
        <nav className="hero-nav__left" aria-label="Ana gezinme">
          {NAV_LINKS.map(({ idx, label, target }) => (
            <a
              key={label}
              className="hero-nav__link"
              href={target === 'console' ? '#console-section' : '#'}
              aria-label={label}
              onClick={(e) => handleNavClick(e, target)}
            >
              <span className="hero-nav__idx">{idx} /</span>
              <span className="hero-nav__label">{label}</span>
            </a>
          ))}
        </nav>

        <div className="hero-nav__right">
          <span className="hero-nav__brand">Passport&lt;OCR-YOLO</span>
          <Clock />
        </div>

        <button
          className="hero-nav__toggle"
          onClick={() => setMenuOpen((v) => !v)}
          aria-label={menuOpen ? 'Menüyü kapat' : 'Menüyü aç'}
          aria-expanded={menuOpen}
        >
          {menuOpen ? 'Kapat ×' : 'Menü'}
        </button>

        <div className={`hero-nav__panel ${menuOpen ? 'is-open' : ''}`} role="navigation">
          <div className="hero-nav__panel-inner">
            {NAV_LINKS.map(({ label }) => (
              <a
                key={label}
                className="hero-nav__panel-link"
                href="#"
                onClick={() => setMenuOpen(false)}
              >
                {label}
              </a>
            ))}
          </div>
        </div>
      </header>

      {/* ── ANA İÇERİK ── */}
      <main className="hero-main" ref={revealRef}>
        {/* Üst: katman seçici + hazır durumu */}
        <section className="hero-upper" aria-label="Modül seçici">
          <div className="hero-switch" role="tablist" aria-label="Pipeline adımları">
            {SLIDES.map((s) => (
              <button
                key={s.id}
                role="tab"
                aria-selected={active === s.id}
                className={`hero-switch__btn ${active === s.id ? 'is-active' : ''}`}
                onClick={() => { setActive(s.id); setMenuOpen(false); }}
              >
                {s.tag} / {s.label}
              </button>
            ))}
          </div>

          <div className="hero-avail" role="status" aria-live="polite">
            <span className="hero-dot" />
            <span>Doğrulamaya hazır</span>
          </div>
        </section>

        {/* Alt: dev başlık + pitch + CTA */}
        <section className="hero-lower" aria-label="Giriş mesajı">
          <div className={`hero-name ${revealed ? 'reveal-up' : ''}`} aria-label="Passport Detection System">
            Passport<br />Detection<span className="hero-name__dot">.</span>
          </div>

          <div
            className={`hero-pitch ${revealed ? 'reveal-right' : ''}`}
            style={{ animationDelay: revealed ? '0.08s' : '0s' }}
          >
            <p className="hero-pitch__text">
              Pasaport ve kimlik belgelerindeki MRZ bölgesini{' '}
              <strong>YOLO</strong> ile tespit eder, <strong>Tesseract OCR-B</strong>{' '}
              ile okur ve <strong>ICAO 9303</strong>'e göre doğrular — her alan
              kontrol edilir, düşük güvenli okumalar otomatik olarak manuel
              incelemeye yönlendirilir.
            </p>

            <div className="hero-trust" aria-label="Performans metrikleri">
              {TRUST_POINTS.map((p) => (
                <div key={p.k} className="hero-trust__item">
                  <span className="hero-trust__v mono">{p.v}</span>
                  <span className="hero-trust__k">{p.k}</span>
                </div>
              ))}
            </div>

            <button
              className="hero-cta"
              onClick={handleStart}
              aria-label="Pasaport taramasını başlat"
            >
              taramaya başla ↓
            </button>
          </div>
        </section>
      </main>

      {/* Aşağı kaydır işareti */}
      <div className="hero-scroll-hint" aria-hidden="true" onClick={handleStart}>
        <span className="hero-scroll-hint__line" />
        <span className="hero-scroll-hint__text">kaydır</span>
      </div>

      {/* Hero-konsol geçiş blur fade */}
      <div className="hero-bottom-fade" aria-hidden="true" />
    </div>
  );
}
