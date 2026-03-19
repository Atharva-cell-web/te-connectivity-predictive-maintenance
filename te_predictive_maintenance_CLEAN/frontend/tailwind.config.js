/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'industrial-bg': '#121212',
        'industrial-card': '#1E1E1E',
        'industrial-text': '#E0E0E0',
        'safe': '#4CAF50',
        'warning': '#FFC107',
        'danger': '#FF5252',
      },
    },
  },
  plugins: [],
}