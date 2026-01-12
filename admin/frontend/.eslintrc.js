module.exports = {
  extends: [
    'react-app',
    'react-app/jest'
  ],
  rules: {
    'no-restricted-globals': ['error', 'isFinite', 'isNaN'].filter((global) =>
      global !== 'confirm'
    ),
  },
};