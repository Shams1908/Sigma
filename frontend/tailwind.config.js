/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Custom gradient utilities
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
      
      // SIGMA Brand Colors
      colors: {
        'sigma-purple': {
          DEFAULT: '#6d28d9',
          50: '#faf5ff',
          100: '#f3e8ff',
          200: '#e9d5ff',
          300: '#d8b4fe',
          400: '#c084fc',
          500: '#a855f7',
          600: '#9333ea',
          700: '#6d28d9',
          800: '#5b21b6',
          900: '#4c1d95',
          950: '#2e1065',
        },
        'sigma-teal': {
          DEFAULT: '#14b8a6',
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
          950: '#042f2e',
        },
        
        // Surface colors (solid materials)
        'sigma-surface': {
          darkest: '#000000',
          darker: '#0A0A0A',
          dark: '#111111',
          medium: '#1a1a1a',
        },
        
        // Border colors
        'sigma-border': {
          DEFAULT: '#222222',
          hover: '#333333',
          active: '#444444',
        },
      },
      
      // Custom box shadows with glows
      boxShadow: {
        'glow-purple': '0 0 40px rgba(109, 40, 217, 0.3)',
        'glow-purple-lg': '0 0 80px rgba(109, 40, 217, 0.4)',
        'glow-teal': '0 0 40px rgba(20, 184, 166, 0.3)',
        'glow-teal-lg': '0 0 80px rgba(20, 184, 166, 0.4)',
        'glow-red': '0 0 40px rgba(239, 68, 68, 0.3)',
        'glow-blue': '0 0 40px rgba(59, 130, 246, 0.3)',
        'glow-white': '0 0 60px rgba(255, 255, 255, 0.4)',
      },
      
      // Animation durations
      transitionDuration: {
        '400': '400ms',
        '600': '600ms',
        '800': '800ms',
      },
      
      // Custom animation curves
      transitionTimingFunction: {
        'sigma': 'cubic-bezier(0.22, 1, 0.36, 1)',
      },
      
      // Custom keyframe animations
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'slide-down': {
          '0%': { transform: 'translateY(-20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'scale-in': {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        'pulse-slow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
      },
      
      animation: {
        'fade-in': 'fade-in 0.5s ease-out',
        'slide-up': 'slide-up 0.5s ease-out',
        'slide-down': 'slide-down 0.5s ease-out',
        'scale-in': 'scale-in 0.4s ease-out',
        'pulse-slow': 'pulse-slow 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
