import React, { useContext } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/Header';
import HomePage from './pages/HomePage';
// import LoginPage from './pages/LoginPage';
// import RegisterPage from './pages/RegisterPage';
import { AuthContext } from './context/AuthContext';
import ChatPage from './pages/ChatPage';
import { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
  body, html {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    font-family: 'Inter', sans-serif;
    background: #0f1117;
    color: #e6e8ef;
    overflow-y: auto;
    overflow-x: hidden;
  }

  #root {
    min-height: 100vh;
    width: 100%;
    display: flex;
    flex-direction: column;
  }

  main.container {
    flex: 1;
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 24px;
    box-sizing: border-box;
  }

  .card {
    background: #161a23;
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #232734;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    max-width: 700px;
    width: 100%;
    margin: 0 auto;
    transition: background 0.2s ease, border 0.2s ease;
  }

  .card:hover {
    background: #1a1f2b;
    border-color: #2c3242;
  }

  .gradientText {
    font-weight: 600;
    font-size: 2.6rem;
    margin-bottom: 1.2rem;
    letter-spacing: 1px;
    color: #ffffff;
  }

  .infoText {
    margin-top: 1.5rem;
    font-size: 1rem;
    line-height: 1.6;
    color: #aab0c0;
  }

  .divider {
    height: 1px;
    width: 100%;
    background: #2a2f3c;
    margin: 2rem 0;
  }

  @media (max-width: 768px) {
    .gradientText {
      font-size: 2.2rem;
    }
    .infoText {
      font-size: 0.95rem;
    }
  }

  @media (max-width: 576px) {
    .gradientText {
      font-size: 1.9rem;
    }
    .infoText {
      font-size: 0.9rem;
    }
  }
`;




function App() {
  // const { isAuth, loading } = useContext(AuthContext);

  // if (loading) {
  //   return <div>Loading...</div>;
  // }

  return (
    <>
      <GlobalStyle />
      <Header />
      <main className="container mt-5">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/chat" element={<ChatPage />} />
        </Routes>
      </main>
    </>
  );
}

export default App;
