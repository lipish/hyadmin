// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

console.log('API Base URL:', API_BASE_URL);
console.log('REACT_APP_API_URL env:', process.env.REACT_APP_API_URL);

export const apiConfig = {
  baseURL: API_BASE_URL,
};

// Helper function to build full API URLs
export const getApiUrl = (endpoint: string): string => {
  const fullUrl = `${apiConfig.baseURL}${endpoint}`;
  console.log('API URL:', fullUrl);
  return fullUrl;
};