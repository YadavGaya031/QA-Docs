import React, { useState } from "react";
import { register } from "../services/api";
import "./Auth.css";

export default function Register({ onRegister }) {
  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setMessage(""); // Clear message on input change
  };

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      const data = await register(form.username, form.email, form.password);

      if (data.user_id) {
        setMessage("✅ Registration successful! Please log in.");
        setForm({ username: "", email: "", password: "" }); // Clear form
        if (onRegister) onRegister(); // Optional: auto-navigate to login
      } else {
        setMessage(data.detail || "❌ Registration failed.");
      }
    } catch (err) {
      console.error(err);
      setMessage("❌ Something went wrong during registration.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="register-form">
      <h2>Register</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="reg-username">Username</label>
          <input
            id="reg-username"
            type="text"
            name="username"
            placeholder="Choose a username"
            value={form.username}
            onChange={handleChange}
            required
            disabled={loading}
          />
        </div>
        <div className="form-group">
          <label htmlFor="reg-email">Email</label>
          <input
            id="reg-email"
            type="email"
            name="email"
            placeholder="Enter your email"
            value={form.email}
            onChange={handleChange}
            required
            disabled={loading}
          />
        </div>
        <div className="form-group">
          <label htmlFor="reg-password">Password</label>
          <input
            id="reg-password"
            type="password"
            name="password"
            placeholder="Choose a password"
            value={form.password}
            onChange={handleChange}
            required
            disabled={loading}
          />
        </div>
        {message && (
          <p className={`message ${message.includes('✅') ? 'success' : 'error'}`}>
            {message}
          </p>
        )}
        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? 'Registering...' : 'Register'}
        </button>
      </form>
    </div>
  );
}
