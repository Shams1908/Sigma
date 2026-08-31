/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: 'var(--color-background)',
        panel: {
          DEFAULT: 'var(--color-panel)',
          raised: 'var(--color-panel-raised)',
        },
        grid: 'var(--color-border)',
        signal: {
          cyan: 'var(--color-signal-cyan)',
          green: 'var(--color-signal-green)',
          amber: 'var(--color-signal-amber)',
          red: 'var(--color-signal-red)',
        },
        primary: 'var(--color-text-primary)',
        muted: 'var(--color-text-muted)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      borderColor: {
        DEFAULT: 'var(--color-border)',
      },
      borderRadius: {
        DEFAULT: '4px',
        sm: '4px',
        md: '6px',
        lg: '6px',
      },
    },
  },
  plugins: [],
}
