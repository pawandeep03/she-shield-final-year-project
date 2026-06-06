// SHESHIELD frontend logic:
// - Slider UI updates
// - Real-time prediction via POST /api/predict (no page reload)
// - Falls back to normal form POST if API fails

function updateSliderUI(val) {
  const sliderVal = document.getElementById('sliderVal');
  const slider = document.getElementById('crimeSlider');
  if (!sliderVal || !slider) return;

  sliderVal.textContent = String(val);
  const color = val <= 3 ? '#27ae60' : val <= 6 ? '#f39c12' : '#e74c3c';
  slider.style.accentColor = color;
  sliderVal.style.color = color;
}

function showResult(result) {
  const placeholder = document.getElementById('resultPlaceholder');
  const card = document.getElementById('resultCard');
  const summary = document.getElementById('inputSummary');

  if (placeholder) placeholder.style.display = 'none';
  if (card) card.style.display = 'block';
  if (summary) summary.style.display = 'block';

  const riskCode = result.risk_code || 'medium';
  if (card) {
    card.classList.remove('result-low', 'result-medium', 'result-high');
    card.classList.add(`result-${riskCode}`);
  }

  const setText = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text ?? '';
  };

  setText('resultEmoji', result.emoji);
  setText('resultLevel', result.prediction);
  setText('resultMessage', result.message);

  // Input summary
  setText('sumArea', result._ui?.area_name);
  setText('sumCrime', `${result._ui?.crime_rate}/10`);
  setText('sumLighting', result._ui?.lighting);
  setText('sumTime', result._ui?.time_of_day);
  setText('sumCrowd', result._ui?.crowd_density);
  setText('sumPolice', `${result._ui?.police_distance} km`);

  setText('resultArea', result._ui?.area_name);
}

async function predictViaApi(payload) {
  const resp = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  return await resp.json();
}

async function sendContactViaApi(payload) {
  const resp = await fetch('/api/contact', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const out = await resp.json().catch(() => ({}));
  if (!resp.ok || out.status !== 'success') {
    throw new Error(out.message || `HTTP ${resp.status}`);
  }
  return out;
}

function init() {
  const form = document.getElementById('predictForm');
  const slider = document.getElementById('crimeSlider');
  const hint = document.getElementById('apiHint');
  const contactForm = document.getElementById('contactForm');
  const contactStatus = document.getElementById('contactStatus');

  if (slider) {
    updateSliderUI(slider.value);
    slider.addEventListener('input', (e) => updateSliderUI(e.target.value));
  }

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (hint) hint.style.display = 'block';

    // Build payload for API
    const fd = new FormData(form);
    const payload = {
      crime_rate: Number(fd.get('crime_rate') || 5),
      lighting: String(fd.get('lighting') || 'Good'),
      police_distance: Number(fd.get('police_distance') || 1.0),
      time_of_day: String(fd.get('time_of_day') || 'Day'),
      crowd_density: String(fd.get('crowd_density') || 'Medium'),
    };

    const areaName = String(fd.get('area_name') || 'Unknown Area');

    try {
      const out = await predictViaApi(payload);
      out._ui = { ...payload, area_name: areaName };

      if (out.status !== 'success') {
        throw new Error(out.message || 'Prediction failed');
      }

      showResult(out);
    } catch (err) {
      // Fallback: submit normal form POST /predict (server-rendered)
      console.warn('API predict failed, falling back to form submit:', err);
      form.submit();
    } finally {
      if (hint) hint.style.display = 'none';
    }
  });

  // Smooth scroll for nav links
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      document.querySelector(this.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
    });
  });

  // Active nav highlight on scroll
  window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section[id]');
    const links = document.querySelectorAll('.nav-link');
    let current = '';
    sections.forEach((s) => {
      if (window.scrollY >= s.offsetTop - 100) current = s.getAttribute('id');
    });
    links.forEach((l) => {
      l.classList.toggle('active', l.getAttribute('href') === '#' + current);
    });
  });

  // Contact form (demo)
  if (contactForm) {
    contactForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (contactStatus) {
        contactStatus.style.display = 'block';
        contactStatus.className = 'contact-status';
        contactStatus.textContent = 'Sending…';
      }

      const fd = new FormData(contactForm);
      const payload = {
        name: String(fd.get('name') || '').trim(),
        email: String(fd.get('email') || '').trim(),
        message: String(fd.get('message') || '').trim(),
      };

      try {
        const out = await sendContactViaApi(payload);
        if (contactStatus) {
          contactStatus.classList.add('ok');
          contactStatus.textContent = out.message || 'Message saved successfully.';
        }
        contactForm.reset();
      } catch (err) {
        if (contactStatus) {
          contactStatus.classList.add('err');
          contactStatus.textContent = err?.message || 'Failed to send message.';
        }
      }
    });
  }
}

document.addEventListener('DOMContentLoaded', init);
