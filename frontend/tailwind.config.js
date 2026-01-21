/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        energy: {
          orange: "#F59E0B",
          orangeSoft: "#FBBF24",
        },
        grid: {
          dark: "#1F1F1F",
          panel: "#2A2A2A",
          border: "#3A3A3A",
          muted: "#9CA3AF",
          text: "#E5E7EB",
        },
        success: "#22C55E",
        danger: "#EF4444",
      },

      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        mono: ["JetBrains Mono", "ui-monospace"],
      },

      boxShadow: {
        panel: "0 0 0 1px rgba(255,255,255,0.04)",
        glow: "0 0 20px rgba(245,158,11,0.25)",
      },

      backgroundImage: {
        "grid-gradient":
          "linear-gradient(180deg, #1F1F1F 0%, #161616 100%)",
      },
    },
  },
  plugins: [],
};
