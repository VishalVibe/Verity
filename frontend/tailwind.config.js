/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {
      colors: {
        supported: {
          bg: '#EAF3DE',
          text: '#27500A',
          border: '#3B6D11',
        },
        contradicted: {
          bg: '#FCEBEB',
          text: '#791F1F',
          border: '#A32D2D',
        },
        unsupported: {
          bg: '#FAEEDA',
          text: '#633806',
          border: '#854F0B',
        },
        brand: '#534AB7',
      }
    },
  },
  plugins: [],
}