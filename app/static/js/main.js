/* The AI Skincare Progress Tracker — main.js */

// Auto-dismiss flash messages after 5s
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .4s ease';
      el.style.opacity    = '0';
      setTimeout(() => el.remove(), 400);
    }, 5000);
  });

  // Animate bar fills on dashboard load
  document.querySelectorAll('.bar-fill').forEach(bar => {
    const target = bar.style.width;
    bar.style.width = '0';
    requestAnimationFrame(() => {
      setTimeout(() => { bar.style.width = target; }, 100);
    });
  });

  // Animate SVG ring on dashboard
  document.querySelectorAll('.ring-svg circle:last-child').forEach(circle => {
    const target = parseFloat(circle.getAttribute('stroke-dashoffset') || '0');
    circle.setAttribute('stroke-dashoffset', '314');
    circle.style.transition = 'stroke-dashoffset 1s ease';
    setTimeout(() => circle.setAttribute('stroke-dashoffset', target), 200);
  });
});
