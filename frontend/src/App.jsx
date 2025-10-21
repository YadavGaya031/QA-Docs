// frontend/src/App.jsx
import React, { useState, useEffect } from "react";
import Login from "./components/Login";
import Register from "./components/Register";
import Dashboard from "./components/Dashboard";
import "./App.css";

function App() {
  const [page, setPage] = useState("login");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is already logged in
    const token = localStorage.getItem("token");
    if (token) {
      setPage("dashboard");
    }
    setLoading(false);
  }, []);

  const handleLogin = () => setPage("dashboard");
  const handleRegister = () => setPage("login"); // go to login after register
  const handleLogout = () => {
    localStorage.removeItem("token");
    setPage("login");
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>QA Document Assistant</h1>
        {page === "dashboard" && (
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        )}
      </header>
      <main className="app-main">
        {page === "login" && (
          <div className="auth-container">
            <Login onLogin={handleLogin} />
            <p className="auth-switch">
              Don't have an account?{" "}
              <button
                className="link-btn"
                onClick={() => setPage("register")}
              >
                Register
              </button>
            </p>
          </div>
        )}

        {page === "register" && (
          <div className="auth-container">
            <Register onRegister={handleRegister} />
            <p className="auth-switch">
              Already have an account?{" "}
              <button
                className="link-btn"
                onClick={() => setPage("login")}
              >
                Login
              </button>
            </p>
          </div>
        )}

        {page === "dashboard" && <Dashboard />}
      </main>
    </div>
  );
}

export default App;
