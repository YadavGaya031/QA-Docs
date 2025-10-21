const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function register(username, email, password) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({username, email, password})
  });
  return res.json();
}

export async function login(username, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({username, password})
  });
  return res.json();
}

export function getAuthHeader() {
  const token = localStorage.getItem("token");
  return token ? { "Authorization": `Bearer ${token}` } : {};
}

export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    headers: {...getAuthHeader()},
    body: form
  });
  return res.json();
}

export async function ingest() {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: {...getAuthHeader(), "Content-Type":"application/json"}
  });
  return res.json();
}

export async function ask(query) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: {...getAuthHeader(), "Content-Type":"application/json"},
    body: JSON.stringify({query})
  });
  return res.json();
}
