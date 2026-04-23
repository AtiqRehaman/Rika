import React, { useContext, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import Container from "react-bootstrap/Container";
import { AuthContext } from "../context/AuthContext";
import styled, { createGlobalStyle } from "styled-components";
import rikaLogo from '../rika_logo.png';

const GlobalStyle = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Serif+JP:wght@600&display=swap');

  body {
    font-family: 'Inter', sans-serif;
    background: #0b0e14;
    color: #f1f5f9;
  }
`;

const StyledCollapse = styled(Navbar.Collapse)`
  background-color: transparent;
`;

const StyledNavbar = styled(Navbar)`
  background: #0b0e14;
  border-bottom: 1px solid #1e2430;
  padding: 1rem 3rem;
`;

const StyledNavbarBrandText = styled.span`
  font-family: 'Noto Serif JP', serif;
  font-size: 2.5rem;
  letter-spacing: 3px;
  color: #f1f5f9;
`;

const StyledNavLink = styled(Nav.Link)`
  color: #9aa4b2 !important;
  margin: 0 20px;
  font-weight: 500;
  font-size: 0.95rem;
  position: relative;
  transition: 0.2s ease;
  text-decoration: none;

  &:hover {
    color: #ffffff !important;
  }

  &:after {
    content: "";
    position: absolute;
    bottom: -6px;
    left: 0;
    width: 0%;
    height: 2px;
    background: #6366f1;
    transition: width 0.3s ease;
  }

  &:hover:after {
    width: 100%;
  }
`;

const StyledButton = styled.button`
  background: transparent;
  border: 1px solid #6366f1;
  color: #6366f1;
  padding: 6px 16px;
  border-radius: 6px;
  font-weight: 500;
  transition: 0.2s ease;

  &:hover {
    background: #6366f1;
    color: #ffffff;
  }
`;

const CenteredUsername = styled.div`
  color: #9aa4b2;
  font-size: 0.9rem;
  margin-right: 24px;
`;

const BrandContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
`;

const BrandLogo = styled.img`
  width: 42px;
  height: 42px;
  border-radius: 8px;
  object-fit: cover;
`;


function Header() {
  const { user, isAuth, logout } = useContext(AuthContext);
  const navigate = useNavigate();
  const [expanded, setExpanded] = useState(false);

  const handleLogout = async () => {
    await logout();
    setExpanded(false);
    navigate("/login");
  };

  const handleSelect = () => {
    setExpanded(false);
  };

  return (
    <>
      <GlobalStyle />
      <StyledNavbar expand="md" fixed="top" navColour={true} expanded={expanded}>
        <Container>
          <Navbar.Brand href="/" className="d-flex">
            <BrandContainer>
              <BrandLogo src={rikaLogo} alt="Rika Logo" />
              <StyledNavbarBrandText>Rika</StyledNavbarBrandText>
            </BrandContainer>

          </Navbar.Brand>
          
          <Navbar.Toggle aria-controls="responsive-navbar-nav" onClick={() => setExpanded(!expanded)} />
          <StyledCollapse id="responsive-navbar-nav">
            <Nav className="ms-auto" defaultActiveKey="#home" onSelect={handleSelect}>
              <Nav.Item>
                <StyledNavLink as={Link} to="/" onClick={handleSelect}>
                  Home
                </StyledNavLink>
              </Nav.Item>

              {!isAuth && (
                <>
                  <Nav.Item>
                    <StyledNavLink as={Link} to="/login" onClick={handleSelect}>
                      Login
                    </StyledNavLink>
                  </Nav.Item>
                  <Nav.Item>
                    <StyledNavLink as={Link} to="/register" onClick={handleSelect}>
                      Register
                    </StyledNavLink>
                  </Nav.Item>
                </>
              )}
              
              {isAuth && user && (
                <>
                  <Nav.Item>
                    <StyledNavLink as={Link} to="/chat" onClick={handleSelect}>
                      Chat with 🤖
                    </StyledNavLink>
                  </Nav.Item>
                  <Nav.Item>
                    <CenteredUsername>{user.username}</CenteredUsername>
                  </Nav.Item> 
                  <Nav.Item>
                    <StyledButton onClick={handleLogout}>
                      Logout
                    </StyledButton>
                  </Nav.Item>
                </>
              )}
            </Nav>
          </StyledCollapse>
        </Container>
      </StyledNavbar>
    </>
  );
}

export default Header;
